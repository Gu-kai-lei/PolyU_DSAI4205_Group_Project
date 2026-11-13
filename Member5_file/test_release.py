"""Release behavior: ordinary Jupyter, offline evidence and provider-free imports."""
import ast
import json
from pathlib import Path
import shutil
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def test_published_python_and_notebooks_have_no_cloud_imports():
    bad=[]
    for path in (ROOT/'Member5_file').glob('*'):
        if path.suffix=='.py':sources=[path.read_text()]
        elif path.suffix=='.ipynb':sources=[''.join(c['source']) for c in json.loads(path.read_text())['cells'] if c['cell_type']=='code']
        else:continue
        for source in sources:
            for node in ast.walk(ast.parse(source)):
                modules=[a.name for a in node.names] if isinstance(node,ast.Import) else [node.module or ''] if isinstance(node,ast.ImportFrom) else []
                if any(m.split('.')[0] in ['google','googleapiclient','google_auth_httplib2','httplib2'] for m in modules):bad.append(str(path.name))
    assert bad==[],bad


def test_committed_evidence_is_recomputed_and_corruption_rejected(tmp_path):
    from release import verify_evidence
    source=ROOT/'Member5_file/evidence/local_refinement_20261113'
    result=verify_evidence(source)
    assert result['verified'] and not result['full_training_archive_checked']
    target=tmp_path/'evidence';shutil.copytree(source,target)
    import pandas as pd
    file=target/'selected_seed42_oof.csv';f=pd.read_csv(file)
    f.loc[0,['prob_0','prob_1','prob_2']]=[1.,0.,0.]
    f.loc[0,'label']=(int(f.loc[0,'label'])+1)%3
    f.to_csv(file,index=False)
    with pytest.raises(ValueError):verify_evidence(target)


@pytest.mark.parametrize('subdirectory,notebook',[('','Member5_TextCNN.ipynb'),('Member5_file','Member5_TextCNN.ipynb'),('Member5_file','Member5_MVP.ipynb')])
def test_generic_notebook_restart_run_all_without_embedding_cache(tmp_path,monkeypatch,subdirectory,notebook):
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    monkeypatch.delenv('DATA_ROOT',raising=False)
    project=tmp_path/'project';project.mkdir()
    for name in ['src','splits','results','Member5_file']:
        shutil.copytree(ROOT/name,project/name,ignore=shutil.ignore_patterns('__pycache__','.ipynb_checkpoints'))
    (project/'data/clean_v2').mkdir(parents=True)
    shutil.copy2(ROOT/'data/clean_v2/clean_v2.csv',project/'data/clean_v2/clean_v2.csv')
    assert not (project/'data/emb').exists()
    kernel=tmp_path/'kernels/release-test';kernel.mkdir(parents=True)
    (kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'Release test','language':'python'}))
    manager=KernelManager(kernel_name='release-test',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel.parent)]))
    n=nbformat.read(project/'Member5_file'/notebook,as_version=4)
    try:executed=NotebookClient(n,km=manager,timeout=120,resources={'metadata':{'path':str(project/subdirectory)}}).execute()
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    assert not any(o.output_type=='error' for c in executed.cells if c.cell_type=='code' for o in c.outputs)
    text='\n'.join(o.get('text','') for c in executed.cells if c.cell_type=='code' for o in c.outputs)
    assert ('Evidence verification passed' if notebook=='Member5_TextCNN.ipynb' else 'Review mode complete') in text


def test_runtime_can_import_without_unix_only_lock_module(tmp_path):
    import subprocess
    script="""import builtins
original=builtins.__import__
def portable(name,*args,**kwargs):
    if name=='fcntl':raise ImportError('Unix-only lock module unavailable')
    return original(name,*args,**kwargs)
builtins.__import__=portable
from v2_runtime import exclusive_run
from pathlib import Path
with exclusive_run(Path('"""+str(tmp_path)+"""')):pass
"""
    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT/'Member5_file',capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_fresh_embedding_manifest_binds_new_bytes_to_same_frozen_model(tmp_path):
    from diagnose_refine import select_embedding_manifest
    import hashlib
    reference=json.loads((ROOT/'Member5_file/evidence/embedding_manifest.json').read_text())
    ref=tmp_path/'reference.json';ref.write_text(json.dumps(reference))
    cache=tmp_path/'emb';cache.mkdir();actual={**reference,'files':{}}
    for name in reference['files']:
        value=b'regenerated floating-point cache';(cache/name).write_bytes(value)
        actual['files'][name]=hashlib.sha256(value).hexdigest()
    manifest=cache/'m5_transformer_manifest.json';manifest.write_text(json.dumps(actual))
    selected,source=select_embedding_manifest(cache,ref,reference['data_sha256'],reference['split_sha256'])
    assert selected['files']==actual['files'] and source=='local_cache_manifest'
    actual['frozen']=False;manifest.write_text(json.dumps(actual))
    with pytest.raises(ValueError):select_embedding_manifest(cache,ref,reference['data_sha256'],reference['split_sha256'])


def test_mvp_outputs_are_scoped_to_reproduction_data_root(tmp_path):
    import subprocess,os
    script="from pathlib import Path; from mvp import OUT,DATA_ROOT; assert OUT.is_relative_to(DATA_ROOT), (OUT,DATA_ROOT)"
    result=subprocess.run([sys.executable,'-c',script],cwd=ROOT/'Member5_file',env={**os.environ,'DATA_ROOT':str(tmp_path)},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_encoding_rejects_unversioned_existing_arrays(tmp_path):
    from mvp import guard_embedding_output
    (tmp_path/'sbert_multi.npy').write_bytes(b'existing team cache')
    with pytest.raises(FileExistsError):guard_embedding_output(tmp_path)
    assert (tmp_path/'sbert_multi.npy').read_bytes()==b'existing team cache'


def test_class_tradeoff_csv_cannot_disagree_with_verified_predictions(tmp_path):
    from release import verify_evidence
    import pandas as pd
    target=tmp_path/'evidence';shutil.copytree(ROOT/'Member5_file/evidence/local_refinement_20261113',target)
    p=target/'class_tradeoffs.csv';frame=pd.read_csv(p);frame.loc[0,'delta']+=.1;frame.to_csv(p,index=False)
    with pytest.raises(ValueError):verify_evidence(target)


def test_fasttext_training_refuses_to_replace_existing_team_arrays(tmp_path):
    from mvp import guard_fasttext_output
    p=tmp_path/'fasttext_mean_fold0.npy';p.write_bytes(b'existing team vectors')
    with pytest.raises(FileExistsError):guard_fasttext_output(tmp_path)
    assert p.read_bytes()==b'existing team vectors'
