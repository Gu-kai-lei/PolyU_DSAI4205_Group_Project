"""Independent reconstruction of v2 attempts, selections, refits and portable outputs."""
from __future__ import annotations
import json
import statistics
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'Member5_file'))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from src.config import CLEAN_DIR,EMB_DIR
from src.data import load_clean,load_splits
from v2_runtime import CANDIDATES,read_json,sha256,signature,write_json


def independently_rank(rows,names,seeds):
    table=pd.DataFrame(rows);values=[]
    for name in names:
        part=table.loc[table.candidate==name]
        expected={(k,s) for k in range(3) for s in seeds}
        if len(part)!=len(expected) or set(zip(part.inner_fold,part.seed))!=expected:raise ValueError('Missing/duplicate inner results')
        if not np.isfinite(part.score.to_numpy()).all():raise ValueError('Nonfinite inner score')
        values.append(dict(candidate=name,score=float(statistics.mean(part.score.tolist())),parameters=int(part.parameters.iloc[0]),
                           epochs=max(1,int(np.median(part.best_epoch)))))
    return sorted(values,key=lambda r:(-r['score'],r['parameters'],r['candidate']))


def check_choice(published,names):
    rows=published['rows']
    stage_a=[r for r in rows if r['seed']==42]
    top=independently_rank(stage_a,names,[42])[:2]
    entrants=sorted({'C0'}|{r['candidate'] for r in top})
    if entrants!=published['entrants']:raise ValueError('Stage B entrants differ from top2 UNION C0')
    if len(rows)!=len(names)*3+len(entrants)*6:raise ValueError('Unexpected search rows')
    if any(r['candidate'] not in entrants for r in rows if r['seed']!=42):raise ValueError('Non-entrant was advanced')
    rank=independently_rank(rows,entrants,[42,17,2026])
    for field,expected in [('winner',rank[0]),('control',next(r for r in rank if r['candidate']=='C0'))]:
        got=published[field]
        if any(got[k]!=v for k,v in expected.items() if k!='score') or not np.isclose(got['score'],expected['score'],atol=1e-12,rtol=0):
            raise ValueError('Winner/control/own median epochs inconsistent')
    return rows


def check_best_epoch(record):
    history=record['history']
    if [h['epoch'] for h in history]!=list(range(1,len(history)+1)):raise ValueError('Noncontiguous epochs')
    scores=np.array([h['valid']['macro_f1'] for h in history],dtype=float)
    if not np.isfinite(scores).all() or np.any(scores<0) or np.any(scores>1):raise ValueError('Invalid validation score')
    best=int(scores.argmax())
    if record['best_epoch']!=best+1 or not np.isclose(record['best_macro_f1'],scores[best],atol=1e-12,rtol=0):raise ValueError('Recorded checkpoint is not earliest best validation epoch')


def bootstrap_recheck(y,groups,a,b,n=2000):
    """Independent grouped confusion-count bootstrap, same registered paired draws."""
    y=np.asarray(y);unique,inverse=np.unique(groups,return_inverse=True)
    def counts(p):
        out=np.zeros((len(unique),9),dtype=np.int64)
        np.add.at(out,(inverse,y*3+np.asarray(p).argmax(1)),1)
        return out
    def f1(cm):
        cm=cm.reshape(-1,3,3);tp=np.diagonal(cm,axis1=1,axis2=2)
        denominator=cm.sum(1)+cm.sum(2)
        return np.divide(2*tp,denominator,out=np.zeros_like(tp,dtype=float),where=denominator!=0).mean(1)
    ac,bc=counts(a),counts(b);rng=np.random.default_rng(42);deltas=[]
    for start in range(0,n,64):
        draws=rng.choice(len(unique),size=(min(64,n-start),len(unique)),replace=True)
        weights=np.zeros((len(draws),len(unique)),dtype=np.int64)
        np.add.at(weights,(np.arange(len(draws))[:,None],draws),1)
        deltas.extend(f1(weights@ac)-f1(weights@bc))
    lo,hi=np.quantile(deltas,[.025,.975])
    return dict(delta=float((f1(ac.sum(0))-f1(bc.sum(0)))[0]),lower=float(lo),upper=float(hi))


def verify_auxiliary(root,protocol,dev):
    root=Path(root);found={}
    for directory,keys in [('diagnostics',{'memory60'}|{f'curve_{c}_{f}' for c in ['C0','C3'] for f in [25,50,100]}),
                           ('calibration',{f'fixed_repeat{i}' for i in range(3)})]:
        records={}
        for path in (root/directory/'fits').glob('*/attempt_*/record.json'):
            r=read_json(path)
            if r['status']!='complete':continue
            if r['key'] in records:raise ValueError('Duplicate auxiliary attempt')
            records[r['key']]=(r,path.parent)
        if set(records)!=keys:raise ValueError(f'Missing diagnostic/calibration attempts: {directory}')
        for key,(r,folder) in records.items():
            if r['signature']!=signature({'protocol':signature(protocol),'metadata':r['metadata']}):raise ValueError('Auxiliary attempt signature differs')
            for name,field in [('predictions.npy','prediction_sha256'),('weights.safetensors','weight_sha256')]:
                if sha256(folder/name)!=r[field]:raise ValueError('Auxiliary attempt artifact differs')
        found.update(records)
    outer=dev.loc[dev.fold!=0].reset_index(drop=True)
    train,valid=next(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(outer,outer.label,outer.group_id))
    frame=outer.iloc[train];valid_ids=outer.iloc[valid].id.tolist();indexed=dev.set_index('id')
    probe,folder=found['memory60'];meta=probe['metadata'];tiny=indexed.loc[meta['fit_ids']]
    if len(tiny)!=60 or tiny.group_id.nunique()!=60 or tiny.label.value_counts().to_dict()!={0:20,1:20,2:20} or not set(tiny.index)<=set(frame.id):raise ValueError('Invalid memory training groups/classes')
    if meta['valid_ids']!=meta['fit_ids'] or meta['seed']!=42 or meta['max_updates']!=200 or meta['target_f1']!=.95:raise ValueError('Memory protocol differs')
    expected={**CANDIDATES['C1'],'dropout':0.,'weight_decay':0.,'max_epochs':200}
    if meta['cfg']!=expected or not 1<=probe['updates']<=200:raise ValueError('Memory configuration differs')
    p=np.load(folder/'predictions.npy');score=f1_score(tiny.label,p.argmax(1),labels=[0,1,2],average='macro',zero_division=0)
    if score<.95 or not probe['target_met'] or not np.isclose(score,probe['history'][-1]['train']['macro_f1']):raise ValueError('Memory gate is unsupported')
    summary=read_json(root/'evidence/diagnostics.json')
    if not np.isclose(summary['memory_train_f1'],score) or summary['memory_updates']!=probe['updates']:raise ValueError('Memory summary differs')
    for c in ['C0','C3']:
        prior=set()
        for fraction in [25,50,100]:
            r,folder=found[f'curve_{c}_{fraction}'];meta=r['metadata'];fit=set(meta['fit_ids'])
            if not prior<=fit or not fit<=set(frame.id) or meta['valid_ids']!=valid_ids or meta['cfg']!=CANDIDATES[c] or meta['seed']!=42:raise ValueError('Learning-curve partition/config differs')
            if fraction==100 and fit!=set(frame.id):raise ValueError('Full curve is incomplete')
            selected=indexed.loc[list(fit)]
            if set(selected.label)!={0,1,2} or set(frame.loc[frame.group_id.isin(selected.group_id)].id)!=fit:raise ValueError('Learning curve splits a group')
            check_best_epoch(r)
            matching=[v for v in summary['curves'] if v['candidate']==c and v['fraction']==fraction/100]
            if len(matching)!=1 or matching[0]['history']!=r['history'] or matching[0]['n_train']!=len(fit):raise ValueError('Learning-curve summary differs')
            prior=fit
    means=[];times=[]
    for i in range(3):
        r,_=found[f'fixed_repeat{i}'];meta=r['metadata']
        if meta['cfg']!={**CANDIDATES['C0'],'max_epochs':4,'patience':10} or meta['seed']!=42 or len(r['history'])!=4 or any(h['train']['n']!=1222 or h['valid']['n']!=303 for h in r['history']):raise ValueError('Synthetic calibration differs')
        values=[h['seconds'] for h in r['history'][1:]]
        if not np.isfinite(values).all() or min(values)<=0:raise ValueError('Calibration timing invalid')
        means.append(float(np.mean(values)));times.extend(values)
    benchmark=read_json(root/'evidence/benchmark.json')
    if not np.isclose(benchmark['median_seconds_per_epoch'],np.median(means)) or not np.allclose(benchmark['range_seconds_per_epoch'],[min(times),max(times)]):raise ValueError('Calibration timing summary differs')


def verify(root):
    root=Path(root).resolve();out=root/'evidence';status=read_json(root/'status.json');protocol=read_json(root/'protocol.json')
    if protocol.get('schema_version')!=2 or status.get('complete') is not True:raise ValueError('Cannot verify incomplete v2 study')
    if protocol['candidates']!=CANDIDATES or protocol['seeds']!=[42,17,2026]:raise ValueError('Registered candidate/seed contract differs')
    if not read_json(out/'diagnostics.json').get('gate_passed'):raise ValueError('Missing diagnostic gate')
    if not read_json(out/'benchmark.json').get('synthetic_only'):raise ValueError('Invalid calibration provenance')
    for relative,checksum in protocol['source_hashes'].items():
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts or rel.parts[0] not in ['Member5_file','src']:raise ValueError('Unsafe source path')
        for path in [ROOT/relative,root/'source'/relative]:
            if sha256(path)!=checksum:raise ValueError(f'Study source differs: {relative}')
    if sha256(CLEAN_DIR/'clean_v2.csv')!=protocol['data_sha256'] or sha256(ROOT/'splits/folds_v1.csv')!=protocol['split_sha256']:raise ValueError('Data/splits changed')
    for name,checksum in protocol['input']['files'].items():
        if Path(name).name!=name:raise ValueError('Unsafe cache path')
        if sha256(EMB_DIR/name)!=checksum:raise ValueError('Frozen cache changed')
    clean=load_clean().sort_values('id');split=load_splits(clean=clean)
    merged=split[['id','split','fold']].merge(clean,on='id',validate='one_to_one').sort_values('id')
    dev=merged.loc[merged.split=='train'].reset_index(drop=True);test=merged.loc[merged.split=='test'].reset_index(drop=True)
    indexed=dev.set_index('id');dev_ids=set(dev.id);test_ids=set(test.id)
    verify_auxiliary(root,protocol,dev)
    if set(dev.group_id)&set(test.group_id) or dev.groupby('group_id').fold.nunique().max()>1:raise ValueError('Official group boundary broken')
    records={};all_attempts=[]
    for file in sorted((root/'fit_cache/fits').glob('*/attempt_*/record.json')):
        r=read_json(file);all_attempts.append(r)
        if r['status']!='complete':continue
        if r['key'] in records:raise ValueError('Duplicate completed attempt')
        expected_signature=signature({'protocol':signature(protocol),'metadata':r['metadata']})
        if r['signature']!=expected_signature:raise ValueError('Attempt signature mismatch')
        for name,field in [('predictions.npy','prediction_sha256'),('weights.safetensors','weight_sha256')]:
            if sha256(file.parent/name)!=r[field]:raise ValueError('Saved artifact checksum mismatch')
        p=np.load(file.parent/'predictions.npy');meta=r['metadata'];ids=meta.get('valid_ids',meta.get('prediction_ids'))
        if p.shape!=(len(ids),3) or not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
        fit=meta['fit_ids']
        if len(set(fit))!=len(fit) or len(set(ids))!=len(ids) or set(fit)&set(ids) or not set(fit)<=dev_ids:raise ValueError('ID leakage/duplicates')
        if meta['purpose']=='final_no_test_scoring':
            if fit!=dev.id.tolist() or ids!=test.id.tolist():raise ValueError('Final fit/prediction ordering incomplete')
        else:
            if not set(ids)<=dev_ids or set(indexed.loc[fit].group_id)&set(indexed.loc[ids].group_id):raise ValueError('Group leakage')
        if not r['history'] or any(not np.isfinite(h['train']['cross_entropy']) or not np.isfinite(h['gradient_norm']) for h in r['history']):raise ValueError('Invalid loss/gradient history')
        if meta['purpose']!='inner':
            if any(h['valid'] is not None for h in r['history']) or len(r['history'])!=meta['cfg']['max_epochs']:raise ValueError('Refit used validation or wrong epoch count')
        else:
            if any(h['valid'] is None for h in r['history']):raise ValueError('Inner validation history missing')
            check_best_epoch(r)
        records[r['key']]=(r,p,file.parent)
    expected_keys={'final_seed42'};selections={}
    for tag in [f'outer{i}' for i in range(5)]+['final']:
        selected=read_json(out/('final_search_selection.json' if tag=='final' else f'{tag}_selection.json'));rows=check_choice(selected,list(protocol['candidates']));selections[tag]=selected
        table=pd.read_csv(out/f'{tag}_inner_scores.csv')
        if table.to_dict('records')!=rows:
            # CSV float roundtrip can differ by 1 ulp; compare keys exactly and scores tightly.
            if len(table)!=len(rows):raise ValueError('Inner CSV differs')
            for a,b in zip(table.to_dict('records'),rows):
                if any(a[k]!=b[k] for k in b if k!='score') or not np.isclose(a['score'],b['score'],atol=1e-12,rtol=0):raise ValueError('Inner CSV changed')
        positions=np.arange(len(dev)) if tag=='final' else np.flatnonzero(dev.fold.to_numpy()!=int(tag[-1]))
        frame=dev.iloc[positions].reset_index(drop=True)
        partitions=list(StratifiedGroupKFold(3,shuffle=True,random_state=42).split(frame,frame.label,frame.group_id))
        for row in rows:
            key=f'{tag}_{row["candidate"]}_i{row["inner_fold"]}_s{row["seed"]}';expected_keys.add(key)
            r,p,_=records[key];meta=r['metadata'];a,b=partitions[row['inner_fold']]
            expected_fit=dev.iloc[positions[a]].id.tolist();expected_valid=dev.iloc[positions[b]].id.tolist()
            if meta['fit_ids']!=expected_fit or meta['valid_ids']!=expected_valid or meta['purpose']!='inner':raise ValueError('Inner partition/order changed')
            if meta['cfg']!=protocol['candidates'][row['candidate']] or meta['seed']!=row['seed']:raise ValueError('Inner config/seed changed')
            score=f1_score(indexed.loc[expected_valid].label,p.argmax(1),labels=[0,1,2],average='macro',zero_division=0)
            if not np.isclose(score,row['score'],atol=1e-12,rtol=0) or not np.isclose(score,r['best_macro_f1'],atol=1e-12,rtol=0) or r['best_epoch']!=row['best_epoch'] or r['parameters']!=row['parameters']:raise ValueError('Inner score/epoch/size not reproducible')
        if tag!='final':
            fold=int(tag[-1]);valid=dev.loc[dev.fold==fold].id.tolist();fit=dev.loc[dev.fold!=fold].id.tolist()
            for kind,choice in [('selected',selected['winner']),('C0',selected['control'])]:
                for seed in [42,17,2026]:
                    key=f'{tag}_{kind}_s{seed}';expected_keys.add(key);r,p,_=records[key];meta=r['metadata']
                    if meta['fit_ids']!=fit or meta['valid_ids']!=valid or meta['seed']!=seed or meta['cfg']!={**protocol['candidates'][choice['candidate']],'max_epochs':choice['epochs']}:raise ValueError('Outer refit/control own epochs mismatch')
    if set(records)!=expected_keys:raise ValueError('Missing or extra completed formal fits')
    final=read_json(out/'final_selection.json');choice=selections['final']['winner']
    if final!=choice:raise ValueError('Final selection changed')
    r,fp,folder=records['final_seed42']
    if r['metadata']['cfg']!={**protocol['candidates'][choice['candidate']],'max_epochs':choice['epochs']} or r['metadata']['seed']!=42:raise ValueError('Final config differs')
    recomputed=[]
    for kind in ['selected','C0']:
        for seed in [42,17,2026]:
            csv=pd.read_csv(out/f'{kind}_seed{seed}_oof.csv')
            if csv.id.tolist()!=dev.id.tolist() or not np.array_equal(csv.label,dev.label) or not np.array_equal(csv.fold,dev.fold) or not np.array_equal(csv.group_id,dev.group_id):raise ValueError('OOF ID/label/group/fold ordering differs')
            p=csv[[f'prob_{i}' for i in range(3)]].to_numpy()
            for fold in range(5):
                ix=dev.fold.to_numpy()==fold
                np.testing.assert_allclose(p[ix],records[f'outer{fold}_{kind}_s{seed}'][1],atol=1e-12,rtol=0)
            if not np.array_equal(csv.prediction,p.argmax(1)):raise ValueError('OOF predicted class differs')
            y=dev.label.to_numpy();pred=p.argmax(1);reported=read_json(out/f'{kind}_seed{seed}_metrics.json')
            prec,rec,f1,_=precision_recall_fscore_support(y,pred,labels=[0,1,2],zero_division=0)
            for c in range(3):
                for field,value in [('precision',prec[c]),('recall',rec[c]),('f1-score',f1[c])]:
                    if not np.isclose(value,reported['classification_report'][str(c)][field],atol=1e-12,rtol=0):raise ValueError('Per-class metric differs')
            if not np.isclose(np.mean(f1),reported['macro_f1'],atol=1e-12,rtol=0) or not np.isclose(accuracy_score(y,pred),reported['accuracy'],atol=1e-12,rtol=0) or confusion_matrix(y,pred,labels=[0,1,2]).tolist()!=reported['confusion_matrix']:raise ValueError('Macro-F1/accuracy/confusion differs')
            recomputed.append(dict(model=kind,seed=seed,macro_f1=float(np.mean(f1)),accuracy=float(accuracy_score(y,pred)),**{f'f1_{c}':float(f1[c]) for c in range(3)}))
    baseline=pd.read_csv(out/'baseline_oof.csv')
    baseline_path=ROOT/'Member5_file/evidence/baseline_oof.csv'
    if sha256(baseline_path)!=protocol['baseline_sha256']:raise ValueError('Original baseline hash differs')
    original=pd.read_csv(baseline_path)
    if baseline.id.tolist()!=dev.id.tolist() or original.id.tolist()!=dev.id.tolist():raise ValueError('Baseline order differs')
    for column in ['label','fold','group_id','prediction']:
        if not np.array_equal(baseline[column],original[column]):raise ValueError('Baseline metadata differs')
    np.testing.assert_allclose(baseline[[f'prob_{c}' for c in range(3)]],original[[f'prob_{c}' for c in range(3)]],atol=1e-15,rtol=0)
    intervals=read_json(out/'bootstrap.json')
    if len(intervals)!=6:raise ValueError('Bootstrap comparison coverage differs')
    for seed in [42,17,2026]:
        a=pd.read_csv(out/f'selected_seed{seed}_oof.csv')[[f'prob_{c}' for c in range(3)]].to_numpy()
        for control in ['C0','baseline']:
            frame=baseline if control=='baseline' else pd.read_csv(out/f'C0_seed{seed}_oof.csv')
            b=frame[[f'prob_{c}' for c in range(3)]].to_numpy()
            values=bootstrap_recheck(dev.label.to_numpy(),dev.group_id.to_numpy(),a,b)
            matched=[v for v in intervals if v['seed']==seed and v['comparison']==f'selected minus {control}']
            if len(matched)!=1 or matched[0]['resamples']!=2000 or matched[0]['unit']!='group_id' or any(not np.isclose(matched[0][k],v,atol=1e-12,rtol=0) for k,v in values.items()):raise ValueError('Paired group bootstrap differs')
    comparison=pd.read_csv(out/'comparison.csv');conclusion=read_json(out/'conclusion.json')
    for row in recomputed:
        match=comparison.loc[(comparison.model==row['model'])&(comparison.seed==row['seed'])]
        if len(match)!=1 or any(not np.isclose(match[k].iloc[0],v,atol=1e-12,rtol=0) for k,v in row.items() if k not in ['model','seed']):raise ValueError('Comparison table differs')
    sel=comparison.loc[comparison.model=='selected'].sort_values('seed');ctl=comparison.loc[comparison.model=='C0'].sort_values('seed')
    delta=sel.macro_f1.to_numpy()-ctl.macro_f1.to_numpy()
    class_deltas=[float(sel[f'f1_{c}'].mean()-ctl[f'f1_{c}'].mean()) for c in range(3)]
    expected=dict(selected_mean=float(sel.macro_f1.mean()),control_mean=float(ctl.macro_f1.mean()),mean_delta=float(delta.mean()),positive_seeds=int((delta>0).sum()),
        practical_target_met=bool(delta.mean()>=.01 and (delta>0).sum()>=2 and min(class_deltas)>=-.02),
        statistical_advantage_confirmed=all(v['lower']>0 for v in intervals if v['comparison']=='selected minus C0'))
    if any(not np.isclose(conclusion[k],v,atol=1e-12,rtol=0) for k,v in expected.items()) or not np.allclose(conclusion['class_mean_deltas'],class_deltas,atol=1e-12,rtol=0):raise ValueError('Conclusion differs from recomputed evidence')
    # Relocation uses root-relative artifacts, ignoring original Colab absolute paths.
    from mvp import TextCNN
    from refine import load_token_cache
    from v2_training import probabilities
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(2)
    with np.load(root/'preds'/f'{protocol["run_id"]}.npz') as a:
        if list(a['classes'])!=[0,1,2] or int(a['seed'])!=42 or a['oof_ids'].tolist()!=dev.id.tolist() or a['test_ids'].tolist()!=test.id.tolist():raise ValueError('Team NPZ contract differs')
        if str(a['run_id'])!=protocol['run_id'] or str(a['data_version'])!='clean_v2' or str(a['split_version'])!='folds_v1':raise ValueError('Team version tags differ')
        published=pd.read_csv(out/'selected_seed42_oof.csv')
        np.testing.assert_allclose(a['oof_proba'],published[[f'prob_{c}' for c in range(3)]],atol=1e-12,rtol=0)
        np.testing.assert_array_equal(a['test_proba'],fp)
        cfg=r['metadata']['cfg'];model=TextCNN(768,filters=cfg['filters'],dropout=cfg['dropout']);model.load_state_dict(load_file(str(folder/'weights.safetensors')))
        if sum(p.numel() for p in model.parameters())!=r['parameters']:raise ValueError('Saved parameter count differs')
        x,m=load_token_cache(EMB_DIR,test.id.tolist());p=probabilities(model,x,m,'cpu')
        np.testing.assert_allclose(p,fp,atol=5e-5,rtol=5e-4)
        if not np.array_equal(p.argmax(1),fp.argmax(1)):raise ValueError('Reloaded classes differ')
        maximum=float(np.max(np.abs(p-fp)))
    result=dict(verified=True,schema_version=2,completed_fits=len(records),all_attempts=len(all_attempts),
        inner_scores_selections_own_epochs_verified=True,all_fit_to_oof_probabilities_verified=True,
        group_boundaries_verified=True,source_cache_data_hashes_verified=True,
        independently_recomputed=recomputed,final_cpu_reload_max_abs_difference=maximum,
        reserved_prediction_rows=len(test),test_labels_scored=False,
        evidence_scope=protocol['evidence_scope'])
    write_json(out/'independent_verification.json',result);print(json.dumps(result,indent=2));return result
