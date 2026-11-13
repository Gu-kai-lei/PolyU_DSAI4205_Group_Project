"""Real local fitting/reload and observation regression checks."""
import numpy as np
import pytest
import torch


def test_real_cpu_fit_reload_and_resume(tmp_path):
    import v2_training as t
    from v2_runtime import AttemptStore,WallBudget,CANDIDATES
    from mvp import TextCNN
    from safetensors.torch import load_file
    x=np.random.default_rng(42).normal(size=(12,5,8)).astype('float32')
    m=np.ones((12,5),bool);y=np.arange(12)%3
    cfg={**CANDIDATES['C1'],'filters':4,'max_epochs':2,'patience':2}
    store=AttemptStore(tmp_path/'run','local_fixture');budget=WallBudget(tmp_path/'budget.json',60);budget.start()
    args=(store,'tiny',{'purpose':'local_test'},x[:9],y[:9],m[:9],x[9:],y[9:],m[9:],cfg,42,'cpu',budget)
    a=t.run_fit(*args);b=t.run_fit(*args)
    assert b['resumed'] and a['folder']==b['folder']
    model=TextCNN(8,filters=4,dropout=cfg['dropout']);model.load_state_dict(load_file(str(a['folder']/'weights.safetensors')))
    np.testing.assert_allclose(t.probabilities(model,x[9:],m[9:],'cpu'),a['probabilities'],atol=1e-7,rtol=0)
    assert np.isfinite(a['record']['history'][0]['gradient_norm'])
    assert a['record']['history'][0]['peak_cuda_bytes']==0
    budget.pause()


def test_explicit_backend_validation_does_not_silently_fallback():
    import v2_training as t
    assert t.validate_device('cpu')=='cpu'
    with pytest.raises(ValueError):t.validate_device('typo')
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError):t.validate_device('cuda')


@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Requires direct access to local Apple GPU')
def test_mps_observation_preserves_rng_and_cpu_reload(tmp_path):
    import v2_training as t
    from mvp import TextCNN
    from safetensors.torch import save_file,load_file
    torch.manual_seed(42);torch.mps.manual_seed(42)
    model=TextCNN(8,filters=4,dropout=.5).to('mps').train()
    x=np.random.default_rng(43).normal(size=(9,5,8)).astype('float32');m=np.ones((9,5),bool)
    class Noisy(TextCNN):
        def forward(self,x,mask=None):
            logits=super().forward(x,mask)
            return logits+torch.rand_like(logits)*.01
    noisy=Noisy(8,filters=4,dropout=.5).to('mps').train()
    rng=torch.mps.get_rng_state().clone()
    t.probabilities(noisy,x,m,'mps')
    assert torch.equal(rng,torch.mps.get_rng_state()) and noisy.training
    p=t.probabilities(model,x,m,'mps')
    assert torch.equal(rng,torch.mps.get_rng_state()) and model.training
    save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(tmp_path/'weights.safetensors'))
    cpu=TextCNN(8,filters=4,dropout=.5);cpu.load_state_dict(load_file(str(tmp_path/'weights.safetensors')))
    np.testing.assert_allclose(t.probabilities(cpu,x,m,'cpu'),p,atol=5e-5,rtol=5e-4)
    from v2_runtime import AttemptStore,WallBudget,CANDIDATES
    budget=WallBudget(tmp_path/'budget.json',60);budget.start()
    cfg={**CANDIDATES['C1'],'filters':4,'max_epochs':2}
    result=t.run_fit(AttemptStore(tmp_path/'fit','mps_fixture'),'tiny',{},x[:6],np.arange(6)%3,m[:6],
        x[6:],np.arange(3),m[6:],cfg,42,'mps',budget)
    assert len(result['record']['history'])==2 and (result['folder']/'weights.safetensors').exists()
    reloaded=TextCNN(8,filters=4,dropout=cfg['dropout'])
    reloaded.load_state_dict(load_file(str(result['folder']/'weights.safetensors')))
    cpu_p=t.probabilities(reloaded,x[6:],m[6:],'cpu')
    np.testing.assert_allclose(cpu_p,result['probabilities'],atol=5e-5,rtol=5e-4)
    np.testing.assert_array_equal(cpu_p.argmax(1),result['probabilities'].argmax(1))
    budget.pause()


def test_local_session_preserves_run_identity_and_rejects_backend_mix(tmp_path):
    from run_local import select_run
    first=select_run(tmp_path,None,'cpu')
    assert select_run(tmp_path,None,'cpu')==first
    assert select_run(tmp_path,first,'cpu')==first
    with pytest.raises(ValueError):select_run(tmp_path,None,'mps')


def test_local_session_rejects_saved_path_traversal(tmp_path):
    from run_local import select_run
    from v2_runtime import write_json
    write_json(tmp_path/'local_session.json',{'run_id':'../outside','device':'cpu'})
    with pytest.raises(ValueError):select_run(tmp_path,None,'cpu')


def test_environment_manifest_is_preserved_on_rejected_resume(tmp_path):
    from run_local import record_environment
    from v2_runtime import read_json
    original={'hardware':{'backend':'cpu'},'packages':[{'name':'torch','version':'2.11'}]}
    record_environment(tmp_path,original)
    with pytest.raises(ValueError):record_environment(tmp_path,{**original,'packages':[]})
    assert read_json(tmp_path/'evidence/local_environment.json')==original


def test_terminating_launcher_stops_its_child(tmp_path):
    import os,signal,subprocess,sys,time
    from pathlib import Path
    from run_local import run_command
    root=Path(__file__).resolve().parents[1];pidfile=tmp_path/'child.pid'
    child="import os,time; from pathlib import Path; Path("+repr(str(pidfile))+").write_text(str(os.getpid())); time.sleep(60)"
    parent="from run_local import run_command; from pathlib import Path; run_command("+repr([sys.executable,'-c',child])+",Path("+repr(str(tmp_path/'output.log'))+"),65)"
    p=subprocess.Popen([sys.executable,'-c',parent],cwd=root,env={**os.environ,'PYTHONPATH':str(root/'Member5_file')},stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        end=time.monotonic()+8
        while not pidfile.exists() and time.monotonic()<end:time.sleep(.05)
        assert pidfile.exists(),'Child did not start'
        child_pid=int(pidfile.read_text());p.send_signal(signal.SIGTERM);p.wait(timeout=8)
        with pytest.raises(ProcessLookupError):os.kill(child_pid,0)
    finally:
        if p.poll() is None:p.kill();p.wait()
