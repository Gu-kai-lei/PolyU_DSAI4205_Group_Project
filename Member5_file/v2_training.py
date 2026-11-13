"""Member5-owned trainer with observational metrics and immutable fit attempts.

No code runs training on import. Explicit CPU/MPS/CUDA backends support local fitting.
Shared src/training.py and the original TextCNN remain untouched.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import random
import time
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, recall_score

from v2_runtime import BudgetExceeded, append_event, read_json, sha256, write_json


def validate_device(device):
    if device=='auto':
        device='cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device not in ['cpu','mps','cuda']:raise ValueError('Unknown training backend')
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('Requested CUDA device is unavailable')
    if device=='mps' and not torch.backends.mps.is_available():raise RuntimeError('Requested Apple MPS device is unavailable; use direct local execution or select CPU explicitly')
    return device


def hardware_info(device):
    result=dict(backend=device,platform=platform.platform(),machine=platform.machine(),threads=2,
                gpu=None,capability=None,cudnn=torch.backends.cudnn.version())
    if device=='cuda':result.update(gpu=torch.cuda.get_device_name(0),capability=list(torch.cuda.get_device_capability(0)))
    elif device=='mps':
        name=subprocess.run(['sysctl','-n','machdep.cpu.brand_string'],capture_output=True,text=True,timeout=5)
        result['gpu']=name.stdout.strip() if name.returncode==0 else 'Apple Metal GPU'
    return result


def synchronize(device):
    if device=='cuda':torch.cuda.synchronize()
    elif device=='mps':torch.mps.synchronize()


@contextmanager
def observing(model):
    modes=[(m,m.training) for m in model.modules()]
    cpu=torch.random.get_rng_state();cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    mps=torch.mps.get_rng_state() if torch.backends.mps.is_available() else None
    numpy_state=np.random.get_state();python_state=random.getstate()
    try:
        model.eval()
        with torch.no_grad():yield
    finally:
        for m,mode in modes:m.training=mode
        torch.random.set_rng_state(cpu)
        if cuda is not None:torch.cuda.set_rng_state_all(cuda)
        if mps is not None:torch.mps.set_rng_state(mps)
        np.random.set_state(numpy_state);random.setstate(python_state)


def probabilities(model,x,mask,device,check=lambda:None):
    output=[]
    with observing(model):
        for start in range(0,len(x),64):
            check()
            xb=torch.as_tensor(np.asarray(x[start:start+64]),dtype=torch.float32,device=device)
            mb=torch.as_tensor(np.asarray(mask[start:start+64]),dtype=torch.bool,device=device)
            output.append(torch.softmax(model(xb,mb),dim=1).cpu().numpy())
    p=np.concatenate(output).astype('float64')
    if p.shape!=(len(x),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5):
        raise ValueError('Malformed model probabilities')
    return p


def observe(model,x,y,mask,device,check=lambda:None):
    p=probabilities(model,x,mask,device,check)
    pred=p.argmax(1);y=np.asarray(y,dtype=int)
    return dict(n=len(y),accuracy=float(accuracy_score(y,pred)),
        macro_f1=float(f1_score(y,pred,labels=[0,1,2],average='macro',zero_division=0)),
        cross_entropy=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-300,1)).mean()),
        recall=recall_score(y,pred,labels=[0,1,2],average=None,zero_division=0).tolist())


def learning_subsets(frame):
    """Nested prefixes of shuffled whole groups, stratified by majority label."""
    strata={c:[] for c in range(3)}
    for group,part in frame.groupby('group_id',sort=True):
        counts=part.label.value_counts();major=min(counts[counts==counts.max()].index)
        strata[int(major)].append(group)
    rng=np.random.default_rng(42)
    for groups in strata.values():rng.shuffle(groups)
    output={}
    for fraction in [.25,.5,1.]:
        selected={g for groups in strata.values() for g in groups[:max(1,int(np.ceil(len(groups)*fraction)))]}
        ix=np.flatnonzero(frame.group_id.isin(selected).to_numpy()).tolist()
        if set(frame.iloc[ix].label)!={0,1,2}:raise ValueError('Learning subset lacks a class')
        output[fraction]=ix
    return output


def run_fit(store,key,metadata,x,y,mask,xv,yv,mv,cfg,seed,device,budget,*,
            deadline=None,max_updates=None,target_f1=None):
    """Completed fit resume only; interrupted fits restart with a new retained attempt."""
    device=validate_device(device)
    from mvp import TextCNN
    from src.training import _batches,seed_everything
    from safetensors.torch import save_file
    torch.set_num_threads(2)
    meta={**metadata,'cfg':cfg,'seed':seed,'device':device,
          'max_updates':max_updates,'target_f1':target_f1}
    existing=store.completed(key,meta)
    if existing is not None:
        return dict(folder=existing,record=read_json(existing/'record.json'),
                    probabilities=np.load(existing/'predictions.npy'),resumed=True)
    budget.check();folder=store.begin(key,meta);history=[];started=time.monotonic()
    model=None;best_state=None;best_score=-1.;best_epoch=0;stale=0;updates=0;target_met=False
    def check():
        budget.check()
        if deadline is not None and time.monotonic()>=deadline:
            raise BudgetExceeded('Diagnostic phase/time limit reached; inconclusive')
    try:
        seed_everything(seed)
        if device=='mps':torch.mps.manual_seed(seed)
        model=TextCNN(x.shape[-1],filters=cfg['filters'],dropout=cfg['dropout']).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
        generator=torch.Generator().manual_seed(seed)
        loss_fn=nn.CrossEntropyLoss()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,cfg['max_epochs']+1):
            check();model.train();synchronize(device);epoch_start=time.monotonic();total=0.;seen=0;norms=[]
            for indices in _batches(len(y),32,generator):
                check()
                xb=torch.as_tensor(np.asarray(x[indices]),dtype=torch.float32,device=device)
                yb=torch.as_tensor(np.asarray(y[indices]),dtype=torch.long,device=device)
                mb=torch.as_tensor(np.asarray(mask[indices]),dtype=torch.bool,device=device)
                optimizer.zero_grad();loss=loss_fn(model(xb,mb),yb)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                loss.backward()
                norm=nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                norms.append(float(norm));optimizer.step();updates+=1
                total+=float(loss.detach())*len(indices)
                seen+=len(indices)
                if max_updates is not None and updates>=max_updates:break
            train=observe(model,x,y,mask,device,check)
            valid=observe(model,xv,yv,mv,device,check) if yv is not None else None
            synchronize(device)
            row=dict(epoch=epoch,updates=updates,train=train,valid=valid,
                optimization_cross_entropy=total/seen,
                gradient_norm=max(norms),lr=float(optimizer.param_groups[0]['lr']),
                seconds=time.monotonic()-epoch_start,
                peak_cuda_bytes=int(torch.cuda.max_memory_allocated()) if device=='cuda' else 0,
                device_allocated_bytes=int(torch.mps.current_allocated_memory()) if device=='mps' else int(torch.cuda.memory_allocated()) if device=='cuda' else 0)
            history.append(row);append_event(folder/'epochs.jsonl',row)
            if valid is not None:
                if valid['macro_f1']>best_score:
                    best_score=valid['macro_f1'];best_epoch=epoch;stale=0
                    best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    save_file(best_state,str(folder/'best.safetensors'))
                else:stale+=1
            else:best_epoch=epoch
            if target_f1 is not None and train['macro_f1']>=target_f1:target_met=True
            print(f'{key} epoch {epoch}: train F1={train["macro_f1"]:.4f}'+
                  (f' valid F1={valid["macro_f1"]:.4f}' if valid else ''),flush=True)
            if target_met or (max_updates is not None and updates>=max_updates) or (valid is not None and stale>=cfg['patience']):break
        if best_state is not None:model.load_state_dict(best_state)
        check();p=probabilities(model,xv,mv,device,check);np.save(folder/'predictions.npy',p)
        save_file({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},str(folder/'weights.safetensors'))
        fields=dict(history=history,best_epoch=best_epoch,best_macro_f1=best_score if yv is not None else None,
            seconds=time.monotonic()-started,parameters=sum(v.numel() for v in model.parameters()),
            prediction_sha256=sha256(folder/'predictions.npy'),weight_sha256=sha256(folder/'weights.safetensors'),
            target_met=target_met,updates=updates)
        store.finish(folder,'complete',fields)
        return dict(folder=folder,record=read_json(folder/'record.json'),probabilities=p,resumed=False)
    except BaseException as exc:
        store.finish(folder,'interrupted' if isinstance(exc,(BudgetExceeded,KeyboardInterrupt)) else 'failed',
                     dict(error=repr(exc),history=history,seconds=time.monotonic()-started))
        raise
    finally:
        del model
        if device=='cuda':torch.cuda.empty_cache()
        elif device=='mps':torch.mps.empty_cache()
