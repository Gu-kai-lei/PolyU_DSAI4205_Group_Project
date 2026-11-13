"""Run the approved study entirely on this computer; no cloud/auth/storage calls."""
from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Member5_file'))
from v2_runtime import read_json,write_json,safe_name


def select_run(root,requested,device,*,persist=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);session=root/'local_session.json'
    previous=read_json(session) if session.exists() else None
    if requested is None and previous:
        if previous['device']!=device:raise ValueError('Backend changed; provide a different explicit run ID')
        return safe_name(previous['run_id'])
    run_id=safe_name(requested or ('m5_local_'+datetime.now(ZoneInfo('Asia/Hong_Kong')).strftime('%Y%m%d_%H%M%S')))
    if previous and previous['run_id']==run_id and previous['device']!=device:raise ValueError('Cannot mix devices in one run')
    if persist:write_json(session,dict(run_id=run_id,device=device))
    return run_id


def record_environment(root,value):
    root=Path(root);original=root/'evidence/local_environment.json'
    if original.exists():
        if read_json(original)!=value:raise ValueError('Local environment differs; preserving original provenance')
    else:write_json(original,value)
    write_json(root/f'evidence/launch_environment_{time.time_ns()}.json',value)


def run_command(command,log,timeout):
    """Forward termination to this launcher's child and wait before returning."""
    previous=signal.getsignal(signal.SIGTERM)
    def terminated(signum,frame):raise KeyboardInterrupt('Local launcher received termination')
    signal.signal(signal.SIGTERM,terminated)
    proc=None;started=time.monotonic()
    try:
        with Path(log).open('w') as handle:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=handle,stderr=subprocess.STDOUT,env=os.environ.copy())
            while proc.poll() is None:
                if time.monotonic()-started>timeout:raise TimeoutError('Local stage exceeded its outer protection timeout')
                time.sleep(.25)
            if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,command)
    except BaseException:
        if proc is not None and proc.poll() is None:
            if os.name=='nt':proc.terminate()
            else:proc.send_signal(signal.SIGINT)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        raise
    finally:signal.signal(signal.SIGTERM,previous)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-id');parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--stages',default='audit,benchmark,diagnose,run,verify')
    args=parser.parse_args()
    stages=args.stages.split(',')
    if not stages or any(s not in ['audit','benchmark','diagnose','run','verify','report'] for s in stages):parser.error('Unknown local stage')
    os.environ.setdefault('DATA_ROOT',str(ROOT/'data'));os.environ['MPLBACKEND']='Agg'
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/local_artifacts/matplotlib'))
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
    from v2_training import validate_device,hardware_info
    device=validate_device(args.device)
    artifact_root=Path(os.environ['DATA_ROOT'])/'local_artifacts/m5_runs'
    run_id=select_run(artifact_root,args.run_id,device,persist=False);run_root=artifact_root/run_id
    run_root.mkdir(parents=True,exist_ok=True)
    import importlib.metadata as md
    environment=dict(
        interpreter=sys.executable,hardware=hardware_info(device),
        packages=sorted([{'name':p.metadata['Name'],'version':p.version} for p in md.distributions()],key=lambda p:p['name'].lower()))
    # Audit validates source/runtime signatures before changing the session or provenance.
    if stages[0]!='audit':stages.insert(0,'audit')
    print(f'Local study {run_id}; backend={device}; results={run_root}',flush=True)
    for stage in stages:
        command=[sys.executable,'-u',str(ROOT/'Member5_file/diagnose_refine.py'),stage,
                 '--run-id',run_id,'--device',device,'--artifact-root',str(artifact_root),'--budget-hours','2']
        log=run_root/f'console_{stage}_{time.time_ns()}.log'
        print(f'[{stage}] {log}',flush=True)
        used=read_json(run_root/'budget.json')['used_seconds'] if (run_root/'budget.json').exists() else 0
        try:run_command(command,log,max(120,7200-used+120))
        except BaseException:
            print(log.read_text()[-8000:],flush=True);raise
        if stage=='audit':
            record_environment(run_root,environment)
            select_run(artifact_root,run_id,device)
        print(f'[{stage}] completed',flush=True)
    status=read_json(run_root/'status.json') if (run_root/'status.json').exists() else {'complete':False}
    print(json.dumps(dict(run_id=run_id,complete=status.get('complete',False),result_directory=str(run_root)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
