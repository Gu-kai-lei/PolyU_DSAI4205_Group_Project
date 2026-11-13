"""M5 v2: fixed frozen representations, diagnostics and bounded nested selection.

Explicit local MPS/CPU and CUDA stages fit models. Audit/verify/report are non-training.
All artifacts live in <artifact-root>/<run-id>; no shared writer is called.
"""
from __future__ import annotations

import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from contextlib import contextmanager
import importlib.metadata as md
import json
import io
from pathlib import Path
import platform
import shutil
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Member5_file'))
import numpy as np
import pandas as pd
import torch
from refine import data,inner_partitions,load_token_cache,metrics,prediction_frame
from src.config import DATA_ROOT,EMB_DIR,CLEAN_DIR,VERSION_TAGS
from v2_runtime import (CANDIDATES,SEEDS,AttemptStore,BudgetExceeded,WallBudget,
    append_event,exclusive_run,rank_candidates,read_json,safe_name,select_entrants,
    sha256,signature,write_json)
from v2_training import learning_subsets,run_fit,validate_device,hardware_info

BASELINE=ROOT/'Member5_file/evidence/baseline_oof.csv'
SOURCE_FILES=['Member5_file/diagnose_refine.py','Member5_file/v2_runtime.py',
 'Member5_file/v2_training.py','Member5_file/v2_verification.py',
 'Member5_file/run_local.py',
 'Member5_file/refine.py','Member5_file/mvp.py','src/training.py',
 'src/data.py','src/config.py','src/cache.py','src/evaluate.py','src/fasttext_emb.py']


def search_partition(runner):
    rows=[];names=list(CANDIDATES)
    for name in names:
        for fold in range(3):rows.append(dict(candidate=name,inner_fold=fold,seed=42,**runner(name,fold,42)))
    entrants=select_entrants(rows,names)
    for name in entrants:
        for fold in range(3):
            for seed in [17,2026]:rows.append(dict(candidate=name,inner_fold=fold,seed=seed,**runner(name,fold,seed)))
    ranked=rank_candidates(rows,entrants,SEEDS)
    return dict(entrants=entrants,winner=ranked[0],control=next(r for r in ranked if r['candidate']=='C0'),rows=rows)


def save_team_archive(path,run_id,dev_ids,dev_proba,test_ids,test_proba):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(oof_ids=np.asarray(dev_ids),oof_proba=dev_proba,test_ids=np.asarray(test_ids),
        test_proba=test_proba,classes=np.array([0,1,2]),run_id=run_id,**{**VERSION_TAGS,'seed':42})
    for ids,p in [(dev_ids,dev_proba),(test_ids,test_proba)]:
        if len(set(ids))!=len(ids) or p.shape!=(len(ids),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5) or (p<0).any() or (p>1).any():raise ValueError('Invalid team probabilities/IDs')
    if path.exists():
        with np.load(path) as old:
            if set(old.files)==set(payload) and all(np.array_equal(old[k],v) for k,v in payload.items()):return
        raise FileExistsError('Refusing to overwrite different team archive')
    temporary=path.with_suffix('.tmp.npz');np.savez(temporary,**payload);temporary.replace(path)


def snapshot_bundle(root,destination,budget=None):
    """Explicit run directory only: no Drive credentials, input caches or symlinks."""
    root=Path(root).resolve();destination=Path(destination).resolve()
    if destination.is_relative_to(root):raise ValueError('Snapshot must be outside its source')
    temporary=destination.with_name(destination.name+'.tmp')
    with tarfile.open(temporary,'w:gz',compresslevel=1) as archive:
        for path in sorted(root.rglob('*')):
            relative=path.relative_to(root)
            allowed={'protocol.json','status.json','budget.json','diagnostic_budget.json','memory_budget.json',
                     'events.jsonl','evidence','source','fit_cache','diagnostics','calibration','preds'}
            console=len(relative.parts)==1 and path.name.startswith('console_') and path.suffix=='.log'
            if relative.parts[0] not in allowed and not console:continue
            if path.is_symlink():raise ValueError('Symlink forbidden in result bundle')
            if not path.is_file() or path.name=='.lock' or path.suffix=='.tmp':continue
            if relative==Path('budget.json') and budget is not None:
                # A downloaded checkpoint represents a paused process, not days of offline GPU use.
                payload=json.dumps(dict(limit_seconds=budget.limit,used_seconds=budget.elapsed(),
                    active=False,heartbeat_wall=time.time())).encode()
                info=tarfile.TarInfo(str(Path(root.name)/relative));info.size=len(payload)
                archive.addfile(info,io.BytesIO(payload));continue
            archive.add(path,arcname=str(Path(root.name)/path.relative_to(root)),recursive=False)
    temporary.replace(destination)
    return destination


class CombinedBudget:
    def __init__(self,*budgets):self.budgets=budgets
    def check(self):
        for b in self.budgets:b.check()


def select_embedding_manifest(directory,reference,data_hash,split_hash):
    directory=Path(directory);reference=Path(reference);expected=read_json(reference)
    local=directory/'m5_transformer_manifest.json'
    manifest=read_json(local if local.exists() else reference)
    for key in ['model','revision','max_len','sequence_policy','sentence_pooling','precision']:
        if manifest.get(key)!=expected[key]:raise ValueError('Frozen encoder/input policy changed: '+key)
    if manifest.get('frozen') is not True or manifest.get('data_sha256')!=data_hash or manifest.get('split_sha256')!=split_hash:raise ValueError('Embedding provenance differs from current frozen task')
    if set(manifest['files'])!=set(expected['files']):raise ValueError('Embedding file coverage changed')
    for name,checksum in manifest['files'].items():
        if Path(name).name!=name or sha256(directory/name)!=checksum:raise ValueError('Embedding checksum differs: '+name)
    return manifest,'local_cache_manifest' if local.exists() else 'published_reference_manifest'


class FrozenStudy:
    def __init__(self,run_id,artifact_root,device='cuda',hours=2):
        self.run_id=safe_name(run_id);self.root=Path(artifact_root).resolve()/run_id
        self.root.mkdir(parents=True,exist_ok=True);self.evidence=self.root/'evidence';self.evidence.mkdir(exist_ok=True)
        self.device=validate_device(device);device=self.device
        self.clean,self.dev,self.test=data()
        self.x,self.mask=load_token_cache(EMB_DIR,self.dev.id.tolist());self.y=self.dev.label.to_numpy()
        if self.x.shape[1:]!=(82,768):raise ValueError('This study requires the audited L82/D768 cache')
        manifest,manifest_source=select_embedding_manifest(EMB_DIR,ROOT/'Member5_file/evidence/embedding_manifest.json',
            sha256(CLEAN_DIR/'clean_v2.csv'),sha256(ROOT/'splits/folds_v1.csv'))
        if manifest['revision']!='4328cf26390c98c5e3c738b4460a05b95f4911f5' or manifest.get('frozen') is not True:raise ValueError('Encoder contract changed')
        for name,checksum in manifest['files'].items():
            if sha256(EMB_DIR/name)!=checksum:raise ValueError(f'Embedding checksum changed: {name}')
        protocol=dict(schema_version=2,run_id=run_id,candidates=CANDIDATES,seeds=list(SEEDS),
            selection='three-fold seed42 top2 UNION C0, then two more seeds',outer_folds=5,
            epoch_rule='each candidate own median nine best epochs',batch_size=32,clip_norm=1.,
            budget_seconds=hours*3600,diagnostic_seconds=900,device=device,
            input=dict(encoder=manifest['model'],revision=manifest['revision'],max_len=82,dim=768,
                       files=manifest['files'],manifest_source=manifest_source),
            data_sha256=sha256(CLEAN_DIR/'clean_v2.csv'),split_sha256=sha256(ROOT/'splits/folds_v1.csv'),
            baseline_sha256=sha256(BASELINE),source_hashes={f:sha256(ROOT/f) for f in SOURCE_FILES},
            versions={p:md.version(p) for p in ['torch','numpy','scipy','pandas','scikit-learn','matplotlib','safetensors']},
            python=platform.python_version(),cuda=torch.version.cuda,
            hardware=hardware_info(device),
            evidence_scope='additional exploratory development evidence; previously inspected outer OOF',
            limitations=['historical full-development unlabeled length statistic retained',
                        'no independent human labels','reserved set previously inspected by team'])
        p=self.root/'protocol.json'
        if p.exists() and signature(read_json(p))!=signature(protocol):raise ValueError('Protocol/environment/source changed; new run ID required')
        if not p.exists():
            write_json(p,protocol)
            for f in SOURCE_FILES:
                target=self.root/'source'/f;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/f,target)
        self.protocol=protocol;self.protocol_signature=signature(protocol)
        self.store=AttemptStore(self.root/'fit_cache',self.protocol_signature)
        self.diag_store=AttemptStore(self.root/'diagnostics',self.protocol_signature)
        self.calibration=AttemptStore(self.root/'calibration',self.protocol_signature)
        self.budget=WallBudget(self.root/'budget.json',hours*3600)

    def audit(self):
        base=pd.read_csv(BASELINE).set_index('id').loc[self.dev.id]
        if not np.array_equal(base.label,self.y) or not np.array_equal(base.fold,self.dev.fold) or not np.array_equal(base.group_id,self.dev.group_id):raise ValueError('Baseline not aligned with official development data')
        p=base[[f'prob_{i}' for i in range(3)]].to_numpy();value=metrics(self.y,p)
        if not np.isclose(value['macro_f1'],.5819958867503482,atol=1e-12):raise ValueError('Unexpected baseline')
        base.reset_index().to_csv(self.evidence/'baseline_oof.csv',index=False)
        result=dict(schema_version=2,development_rows=len(self.dev),reserved_rows=len(self.test),
            baseline=value,all_frozen_cache_hashes_verified=True,test_labels_scored=False,
            no_empty_rows=int((~self.mask.any(1)).sum())==0,rows_at_length_boundary=int((self.mask.sum(1)>=80).sum()),
            limitations=self.protocol['limitations'])
        write_json(self.evidence/'audit.json',result);return result

    def _fit(self,key,fit,valid,cfg,seed,*,purpose='inner',store=None,budget=None,deadline=None,**kwargs):
        metadata=dict(purpose=purpose,fit_ids=self.dev.iloc[fit].id.tolist(),valid_ids=self.dev.iloc[valid].id.tolist())
        return run_fit(store or self.store,key,metadata,self.x[fit],self.y[fit],self.mask[fit],
            self.x[valid],self.y[valid] if purpose in ['inner','diagnostic_curve'] else None,self.mask[valid],
            cfg,seed,self.device,budget or self.budget,deadline=deadline,**kwargs)

    def benchmark(self):
        if (self.evidence/'benchmark.json').exists():return read_json(self.evidence/'benchmark.json')
        rng=np.random.default_rng(42)
        x=rng.normal(size=(1222,82,768)).astype('float16');m=np.ones((1222,82),bool);y=np.arange(1222)%3
        means=[];all_times=[]
        for repeat in range(3):
            cfg={**CANDIDATES['C0'],'max_epochs':4,'patience':10}
            result=run_fit(self.calibration,f'fixed_repeat{repeat}',{'purpose':'synthetic_benchmark'},
                x,y,m,x[:303],y[:303],m[:303],cfg,42,self.device,self.budget)
            measured=[r['seconds'] for r in result['record']['history'][1:]]
            means.append(float(np.mean(measured)));all_times.extend(measured)
        # Worst-case warning only. Candidate set is NEVER selected using measured time.
        result=dict(synthetic_only=True,warmup_epochs=1,repeats=3,measured_epochs_per_repeat=3,
            median_seconds_per_epoch=float(np.median(means)),range_seconds_per_epoch=[min(all_times),max(all_times)],
            conservative_max247_fits20_epochs_seconds=float(np.median(means)*247*20),
            candidates_changed_by_benchmark=False)
        write_json(self.evidence/'benchmark.json',result);return result

    def diagnose(self):
        done=self.evidence/'diagnostics.json'
        if done.exists() and read_json(done).get('gate_passed'):
            result=read_json(done);self._diagnostic_report(result);return result
        db=WallBudget(self.root/'diagnostic_budget.json',900);db.start();combined=CombinedBudget(self.budget,db)
        try:
            outer=np.flatnonzero(self.dev.fold.to_numpy()!=0)
            fi,vi=inner_partitions(self.dev.iloc[outer].reset_index(drop=True))[0]
            fit,valid=outer[fi],outer[vi];frame=self.dev.iloc[fit].reset_index(drop=True)
            tiny=[];used=set();rng=np.random.default_rng(42)
            for label in range(3):
                available=frame.loc[frame.label==label].index.to_numpy();rng.shuffle(available);picked=0
                for index in available:
                    group=frame.iloc[index].group_id
                    if group in used:continue
                    used.add(group);tiny.append(int(fit[index]));picked+=1
                    if picked==20:break
                if picked!=20:raise ValueError('Not enough distinct training groups for memory diagnostic')
            tiny=np.array(tiny);cfg={**CANDIDATES['C1'],'dropout':0.,'weight_decay':0.,'max_epochs':200}
            mb=WallBudget(self.root/'memory_budget.json',120);mb.start()
            try:
                probe=self._fit('memory60',tiny,tiny,cfg,42,purpose='diagnostic_memory',store=self.diag_store,
                    budget=CombinedBudget(self.budget,db,mb),max_updates=200,target_f1=.95)
            finally:mb.pause()
            if not probe['record']['target_met']:
                write_json(done,dict(gate_passed=False,reason='Memory probe did not reach .95; investigate inputs/optimizer before formal study'))
                raise RuntimeError('Memory diagnostic inconclusive/failed; no formal study authorized by gate')
            subsets=learning_subsets(frame);curves=[]
            for candidate in ['C0','C3']:
                for fraction,ix in subsets.items():
                    result=self._fit(f'curve_{candidate}_{int(fraction*100)}',fit[ix],valid,CANDIDATES[candidate],42,
                        purpose='diagnostic_curve',store=self.diag_store,budget=combined)
                    curves.append(dict(candidate=candidate,fraction=fraction,n_train=len(ix),
                        best_epoch=result['record']['best_epoch'],history=result['record']['history']))
            result=dict(gate_passed=True,memory_train_f1=probe['record']['history'][-1]['train']['macro_f1'],
                memory_updates=probe['record']['updates'],curves=curves,
                interpretation='Diagnostic patterns only; no human labels, no causal proof; candidates unchanged')
            write_json(done,result);self._diagnostic_report(result);return result
        finally:db.pause()

    def _diagnostic_report(self,result):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        notes=['# TextCNN 训练诊断','',
            '这些结果只来自外折0训练范围内的一份内层划分，不能代替五折性能结论。',
            f'60条记忆测试 F1={result["memory_train_f1"]:.4f}，更新次数={result["memory_updates"]}。',
            '这只说明训练链路能学习小样本，不证明标签正确或泛化能力。','']
        for axis,candidate in zip(axes,['C0','C3']):
            curves=[r for r in result['curves'] if r['candidate']==candidate]
            best=[next(h for h in c['history'] if h['epoch']==c['best_epoch']) for c in curves]
            axis.plot([c['n_train'] for c in curves],[h['train']['macro_f1'] for h in best],'o-',label='Train')
            axis.plot([c['n_train'] for c in curves],[h['valid']['macro_f1'] for h in best],'o-',label='Inner validation')
            axis.set(title=candidate,xlabel='Training rows',ylabel='Macro-F1',ylim=(0,1));axis.legend()
            last=best[-1];train=last['train']['macro_f1'];valid=last['valid']['macro_f1']
            notes.append(f'- {candidate}：完整训练子集在最佳验证轮的训练 F1={train:.4f}，验证 F1={valid:.4f}，差距={train-valid:.4f}。')
        fig.tight_layout();fig.savefig(self.evidence/'learning_curves.png',dpi=150);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for axis,candidate in zip(axes,['C0','C3']):
            history=next(c['history'] for c in result['curves'] if c['candidate']==candidate and c['fraction']==1.)
            axis.plot([h['epoch'] for h in history],[h['train']['cross_entropy'] for h in history],label='Train')
            axis.plot([h['epoch'] for h in history],[h['valid']['cross_entropy'] for h in history],label='Inner validation')
            axis.set(title=candidate,xlabel='Epoch',ylabel='Evaluation cross-entropy');axis.legend()
        fig.tight_layout();fig.savefig(self.evidence/'epoch_losses.png',dpi=150);plt.close(fig)
        notes+=['','训练分高、验证分低与过拟合相符；两者都低也可能来自表示、标签口径或优化。单次划分不能唯一确定原因。',
                '没有人工标签复核；后续六个候选及选模规则不随本诊断结果改变。']
        (self.evidence/'DIAGNOSIS_ZH.md').write_text('\n'.join(notes)+'\n')

    def _select(self,positions,tag):
        partitions=inner_partitions(self.dev.iloc[positions].reset_index(drop=True))
        def runner(candidate,fold,seed):
            fit,valid=partitions[fold]
            result=self._fit(f'{tag}_{candidate}_i{fold}_s{seed}',positions[fit],positions[valid],CANDIDATES[candidate],seed)
            row=dict(score=metrics(self.y[positions[valid]],result['probabilities'])['macro_f1'],
                best_epoch=result['record']['best_epoch'],parameters=result['record']['parameters'])
            return row
        choice=search_partition(runner)
        selection_file='final_search_selection.json' if tag=='final' else f'{tag}_selection.json'
        write_json(self.evidence/selection_file,choice)
        pd.DataFrame(choice['rows']).to_csv(self.evidence/f'{tag}_inner_scores.csv',index=False)
        return choice

    def checkpoint(self,tag):
        self.budget.check()
        path=snapshot_bundle(self.root,self.root.parent/f'{self.run_id}_{tag}.tar.gz',self.budget)
        append_event(self.root/'events.jsonl',{'checkpoint':tag,'archive_sha256':sha256(path),'storage':'local'})
        self.budget.check();return path

    def run(self):
        if (self.root/'status.json').exists() and read_json(self.root/'status.json').get('complete'):
            if read_json(self.root/'status.json')['status']=='checkpoint_failed':
                self.checkpoint('complete_retry')
                status=read_json(self.root/'status.json');status.update(status='complete');status.pop('error',None)
                write_json(self.root/'status.json',status)
            return
        self.audit();self.benchmark();self.diagnose()
        oof={kind:{s:np.full((len(self.dev),3),np.nan) for s in SEEDS} for kind in ['selected','C0']}
        choices=[]
        previous=read_json(self.root/'status.json') if (self.root/'status.json').exists() else {}
        already_completed=int(previous.get('completed_outer_folds',0))
        checkpointed=int(previous.get('checkpointed_outer_folds',0))
        for fold in range(5):
            train=np.flatnonzero(self.dev.fold.to_numpy()!=fold);valid=np.flatnonzero(self.dev.fold.to_numpy()==fold)
            selection=self._select(train,f'outer{fold}');choices.append(dict(fold=fold,**selection['winner']))
            for kind,choice in [('selected',selection['winner']),('C0',selection['control'])]:
                cfg={**CANDIDATES[choice['candidate']],'max_epochs':choice['epochs']}
                for seed in SEEDS:
                    result=self._fit(f'outer{fold}_{kind}_s{seed}',train,valid,cfg,seed,purpose='outer_refit')
                    oof[kind][seed][valid]=result['probabilities']
            write_json(self.root/'status.json',dict(schema_version=2,complete=False,status='running',completed_outer_folds=max(already_completed,fold+1),checkpointed_outer_folds=checkpointed))
            write_json(self.evidence/'outer_selections.json',choices)
            if fold>=checkpointed:
                self.checkpoint(f'outer{fold}');checkpointed=fold+1
                state=read_json(self.root/'status.json');state['checkpointed_outer_folds']=checkpointed
                write_json(self.root/'status.json',state)
        # No outer metric selection: arrays are only scored after every outer fold and final model complete.
        selection=self._select(np.arange(len(self.dev)),'final');winner=selection['winner']
        write_json(self.evidence/'final_selection.json',winner)
        xt,mt=load_token_cache(EMB_DIR,self.test.id.tolist());cfg={**CANDIDATES[winner['candidate']],'max_epochs':winner['epochs']}
        final=run_fit(self.store,'final_seed42',dict(purpose='final_no_test_scoring',
            fit_ids=self.dev.id.tolist(),prediction_ids=self.test.id.tolist()),self.x,self.y,self.mask,
            xt,None,mt,cfg,42,self.device,self.budget)
        for kind in oof:
            for seed,p in oof[kind].items():prediction_frame(self.dev,p).to_csv(self.evidence/f'{kind}_seed{seed}_oof.csv',index=False)
        save_team_archive(self.root/'preds'/f'{self.run_id}.npz',self.run_id,self.dev.id.tolist(),
            oof['selected'][42],self.test.id.tolist(),final['probabilities'])
        write_json(self.evidence/'training_audit.json',self.store.records())
        self._report()
        write_json(self.root/'status.json',dict(schema_version=2,complete=True,status='complete',test_labels_scored=False,
            prediction_archive=f'preds/{self.run_id}.npz',weight_file=str((final['folder']/'weights.safetensors').relative_to(self.root)),
            active_seconds=self.budget.elapsed(),training_seconds=sum(r.get('seconds',0) for r in self.store.records())))
        self.checkpoint('complete')

    def _report(self):
        self.budget.check()
        from refine import report
        report(self.evidence)
        c=read_json(self.evidence/'conclusion.json')
        c.update(evidence_scope=self.protocol['evidence_scope'],evaluated_object='adaptive two-stage selection pipeline, not independent fixed final config',
                 human_ground_truth_verified=False)
        write_json(self.evidence/'conclusion.json',c)
        records=self.store.records()
        write_json(self.evidence/'compute_summary.json',dict(
            observed_active_wall_seconds=self.budget.elapsed(),budget_seconds=self.budget.limit,
            formal_attempts=len(records),completed_formal_fits=sum(r['status']=='complete' for r in records),
            failed_or_interrupted_attempts=sum(r['status']!='complete' for r in records),
            formal_attempt_seconds=sum(r.get('seconds',0) for r in records),
            peak_cuda_bytes=max((h.get('peak_cuda_bytes',0) for r in records for h in r.get('history',[])),default=0),
            maximum_observed_device_allocated_bytes=max((h.get('device_allocated_bytes',0) for r in records for h in r.get('history',[])),default=0),
            device_memory_note='MPS allocator sampled at epoch end; not an exact peak and excludes other applications',
            parameters_by_candidate={k:sum(p.numel() for p in __import__('mvp').TextCNN(768,filters=v['filters'],dropout=v['dropout']).parameters()) for k,v in CANDIDATES.items()},
            hardware=self.protocol['hardware'],currency_cost='Not measured: local electricity/energy usage' if self.device in ['mps','cpu'] else 'Not measured: depends on GPU service',
            restore_limitation='Work lost after the latest durable checkpoint is not observable from that archive'))
        self.budget.check()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','benchmark','diagnose','run','verify','report'])
    parser.add_argument('--run-id',required=True);parser.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    parser.add_argument('--artifact-root',type=Path,default=DATA_ROOT/'local_artifacts/m5_v2')
    parser.add_argument('--budget-hours',type=float,default=2)
    args=parser.parse_args()
    if not 0<args.budget_hours<=2:parser.error('Budget must be positive and no greater than two hours')
    root=args.artifact_root.resolve()/safe_name(args.run_id)
    if args.stage=='verify':
        from v2_verification import verify
        verify(root)
        return
    if args.stage=='report':
        if not read_json(root/'status.json').get('complete'):raise ValueError('Incomplete study cannot publish full scores')
        from refine import report
        report(root/'evidence')
        c=read_json(root/'evidence/conclusion.json');p=read_json(root/'protocol.json')
        c.update(evidence_scope=p['evidence_scope'],evaluated_object='adaptive two-stage selection pipeline, not independent fixed final config',human_ground_truth_verified=False)
        write_json(root/'evidence/conclusion.json',c);return
    with exclusive_run(root):
        study=FrozenStudy(args.run_id,args.artifact_root,args.device,args.budget_hours)
        if args.stage=='audit':print(json.dumps(study.audit(),indent=2));return
        study.budget.start()
        try:
            value=getattr(study,{'benchmark':'benchmark','diagnose':'diagnose','run':'run'}[args.stage])()
            if value is not None:
                if args.stage=='diagnose':value={k:v for k,v in value.items() if k!='curves'}
                print(json.dumps(value,indent=2))
        except BaseException as exc:
            prior=read_json(root/'status.json') if (root/'status.json').exists() else {}
            # Completed model + failed upload is reported separately, never erased as a partial fit.
            write_json(root/'status.json',{**prior,'schema_version':2,'complete':prior.get('complete',False),
                'status':'checkpoint_failed' if prior.get('complete') else 'interrupted','error':repr(exc)})
            raise
        finally:study.budget.pause()


if __name__=='__main__':main()
