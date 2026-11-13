"""Portable, non-training controls for the Member 5 v2 study."""
from __future__ import annotations

from contextlib import contextmanager
from filelock import FileLock,Timeout
import hashlib
import json
from pathlib import Path
import re
import time

SEEDS = (42, 17, 2026)
BASE = dict(filters=100, lr=.001, dropout=.5, weight_decay=.01,
            class_weight=False, max_epochs=20, patience=4)
CANDIDATES = {
    'C0': dict(BASE), 'C1': {**BASE, 'filters':32}, 'C2': {**BASE, 'filters':64},
    'C3': {**BASE, 'filters':64, 'lr':.0003},
    'C4': {**BASE, 'filters':64, 'lr':.0003, 'dropout':.3},
    'C5': {**BASE, 'filters':64, 'lr':.0003, 'dropout':.6},
}


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Use a simple run/key name, not a path')
    return value


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def signature(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(1024*1024), b''):h.update(part)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    temporary.replace(path)


def append_event(path, value):
    path=Path(path);path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(canonical({'wall_utc_seconds':time.time(), **value})+'\n');f.flush()


@contextmanager
def exclusive_run(root):
    path=Path(root)/'.lock';path.parent.mkdir(parents=True,exist_ok=True)
    lock=FileLock(str(path))
    try:lock.acquire(timeout=0)
    except Timeout as exc:raise RuntimeError('Another process is using this run') from exc
    try:yield
    finally:lock.release()


def rank_candidates(rows, names, seeds):
    """Rank only complete, unique candidate/fold/seed coverage."""
    import statistics
    expected={(k,s) for k in range(3) for s in seeds};ranked=[]
    for name in names:
        part=[r for r in rows if r['candidate']==name]
        if len(part)!=len(expected) or {(r['inner_fold'],r['seed']) for r in part}!=expected:
            raise ValueError(f'Incomplete/duplicate inner results: {name}')
        scores=[float(r['score']) for r in part]
        if any(not (-1<float(x)<2) for x in scores):raise ValueError('Nonfinite/invalid score')
        if len({r['parameters'] for r in part})!=1:raise ValueError('Parameter count changed')
        ranked.append(dict(candidate=name, score=statistics.mean(scores),
                           epochs=max(1,int(statistics.median(r['best_epoch'] for r in part))),
                           parameters=part[0]['parameters']))
    return sorted(ranked,key=lambda r:(-r['score'],r['parameters'],r['candidate']))


def select_entrants(rows, names):
    ranked=rank_candidates(rows,names,[42])
    return sorted({'C0'}|{r['candidate'] for r in ranked[:2]})


class BudgetExceeded(RuntimeError):
    pass


class WallBudget:
    """Cumulative active wall time; an unclean shutdown is conservatively charged."""
    def __init__(self, path, limit_seconds, *, clock=time.monotonic):
        self.path=Path(path);self.limit=float(limit_seconds);self.clock=clock
        self.used=0.;self.started=None;self.last_save=0.
        if self.path.exists():
            old=read_json(self.path)
            if old['limit_seconds']!=self.limit:raise ValueError('Budget changed; new run required')
            self.used=float(old['used_seconds'])
            if old.get('active'):
                self.used+=max(0.,time.time()-old['heartbeat_wall'])
        self._save()

    def elapsed(self):
        return self.used+(self.clock()-self.started if self.started is not None else 0.)

    def remaining(self):return self.limit-self.elapsed()

    def _save(self):
        write_json(self.path,dict(limit_seconds=self.limit,used_seconds=self.elapsed(),
                   active=self.started is not None,heartbeat_wall=time.time()))
        self.last_save=self.clock()

    def start(self):
        if self.started is None:self.started=self.clock();self._save()
        self.check()

    def pause(self):
        self.used=self.elapsed();self.started=None;self._save()

    def check(self):
        if self.remaining()<=0:
            self.pause();raise BudgetExceeded('Active experiment wall-time budget exhausted')
        if self.started is not None and self.clock()-self.last_save>=5:self._save()


class AttemptStore:
    """Every attempt has its own directory; retries never overwrite failed histories."""
    def __init__(self, root, protocol_signature):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.protocol=protocol_signature
        binding=self.root/'binding.json'
        if binding.exists() and read_json(binding)['protocol_signature']!=protocol_signature:
            raise ValueError('Run protocol changed; use a fresh run ID')
        if not binding.exists():write_json(binding,{'protocol_signature':protocol_signature})

    def _signature(self, metadata):return signature({'protocol':self.protocol,'metadata':metadata})

    def completed(self, key, metadata):
        parent=self.root/'fits'/safe_name(key)
        for record in sorted(parent.glob('attempt_*/record.json'),reverse=True):
            old=read_json(record)
            if old['signature']!=self._signature(metadata):raise ValueError('Fit signature changed')
            if old['status']!='complete':continue
            folder=record.parent
            for filename,field in [('predictions.npy','prediction_sha256'),('weights.safetensors','weight_sha256')]:
                if field in old and sha256(folder/filename)!=old[field]:raise ValueError('Artifact checksum changed')
            return folder
        return None

    def begin(self, key, metadata):
        parent=self.root/'fits'/safe_name(key);parent.mkdir(parents=True,exist_ok=True)
        paths=sorted(parent.glob('attempt_*'))
        for path in paths:
            if not (path/'record.json').exists():continue
            old=read_json(path/'record.json')
            if old['signature']!=self._signature(metadata):raise ValueError('Fit signature changed')
            if old['status']=='running':self.finish(path,'interrupted',{'error':'Prior process ended before completion'})
        folder=parent/f'attempt_{len(paths)+1:04d}';folder.mkdir()
        write_json(folder/'record.json',dict(key=key,status='running',signature=self._signature(metadata),metadata=metadata))
        append_event(self.root/'events.jsonl',{'key':key,'attempt':folder.name,'status':'running'})
        return folder

    def finish(self, folder, status, fields):
        record=read_json(Path(folder)/'record.json')
        if record['status'] not in ['running']:raise ValueError('A finalized attempt is immutable')
        record.update(fields);record['status']=status;write_json(Path(folder)/'record.json',record)
        append_event(self.root/'events.jsonl',{'key':record['key'],'attempt':Path(folder).name,'status':status})

    def records(self):
        return [read_json(p)|{'relative_path':str(p.parent.relative_to(self.root))}
                for p in sorted((self.root/'fits').glob('*/attempt_*/record.json'))]
