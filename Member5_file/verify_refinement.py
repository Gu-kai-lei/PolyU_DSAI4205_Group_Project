"""Independent audit of saved refinement results, group boundaries and weight reload."""
import argparse
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score
from src.config import DATA_ROOT, CLEAN_DIR, EMB_DIR
from src.data import load_clean, load_splits
from src.training import predict_proba


def resolve_prediction_archive(cache, run_id):
    """Use the relocated cache, never the obsolete machine-specific path in old status files."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', run_id):
        raise ValueError('Invalid run ID')
    path = Path(cache) / 'preds' / f'{run_id}.npz'
    if not path.is_file():
        raise FileNotFoundError(f'Prediction archive missing from supplied cache: {path}')
    return path


def validate_fit_record(row,dev,test_ids):
    p=row['provenance']
    fit=set(p['fit_ids'])
    valid=set(p.get('valid_ids',p.get('prediction_ids',[])))
    indexed=dev.set_index('id')
    dev_ids=set(indexed.index)
    if not fit<=dev_ids or fit&test_ids:
        raise ValueError('Fit contains reserved or unknown IDs')
    if p['purpose']=='final_no_test_scoring':
        if fit!=dev_ids or valid!=test_ids:
            raise ValueError('Final fit/prediction IDs incomplete')
        return
    if not valid<=dev_ids or fit&valid:
        raise ValueError('Training/validation ID overlap or unknown validation IDs')
    if set(indexed.loc[list(fit)].group_id)&set(indexed.loc[list(valid)].group_id):
        raise ValueError('Training/validation group overlap')
    m=re.match(r'outer(\d)_',row['key'])
    if m:
        fold=int(m.group(1))
        outer_valid=set(dev.loc[dev.fold==fold,'id'])
        outer_train=dev_ids-outer_valid
        if p['purpose']=='inner' and not fit|valid<=outer_train:
            raise ValueError('Outer validation used during inner selection')
        if p['purpose']=='outer_refit' and (fit!=outer_train or valid!=outer_valid):
            raise ValueError('Outer refit must use complete training and validation partitions')


def verify(out):
    from mvp import TextCNN
    from refine import file_hash, load_token_cache, write_json
    torch.set_num_threads(2)
    clean=load_clean().sort_values('id')
    split=load_splits(clean=clean)
    dev=split.loc[split.split=='train'].merge(clean,on='id',suffixes=('_split',''))
    dev=dev.sort_values('id').reset_index(drop=True)
    test_ids=set(split.loc[split.split=='test','id'])
    status=json.loads((out/'status.json').read_text())
    if status.get('complete') is not True:
        raise ValueError('Cannot verify incomplete study as complete')
    protocol=json.loads((out/'protocol.json').read_text())
    if file_hash(CLEAN_DIR/'clean_v2.csv')!=protocol['data_sha256'] or file_hash(ROOT/'splits'/'folds_v1.csv')!=protocol['split_sha256']:
        raise ValueError('Data/split changed')
    if file_hash(ROOT/'Member5_file'/'refine.py')!=protocol['code_sha256']:
        raise ValueError('Study source differs from recorded protocol')
    records=json.loads((out/'training_audit.json').read_text())
    cache=DATA_ROOT/'local_artifacts'/'m5_refinement'/out.name
    for row in records:
        if row['status']!='complete':
            raise ValueError('Incomplete or failed fit in completed study')
        validate_fit_record(row,dev,test_ids)
        meta_path=cache/row['key']/'fit.json'
        if json.loads(meta_path.read_text())['signature']!=row['signature']:
            raise ValueError('Fit cache signature differs from published audit')
        if file_hash(meta_path.parent/'predictions.npy')!=row['prediction_sha256']:
            raise ValueError('Fit probabilities changed')
        history=row['history']['history']
        if not 1<=row['history']['best_epoch']<=row['cfg']['max_epochs']:
            raise ValueError('Invalid selected epoch')
        if not all(np.isfinite(x['train_loss']) for x in history):
            raise ValueError('Nonfinite loss')
    recomputed=[]
    for seed in [42,17,2026]:
        for kind in ['selected','C0']:
            frame=pd.read_csv(out/f'{kind}_seed{seed}_oof.csv')
            if frame.id.duplicated().any() or set(frame.id)!=set(dev.id):
                raise ValueError('Incomplete or duplicate OOF IDs')
            frame=frame.set_index('id').loc[dev.id]
            if not np.array_equal(frame.label,dev.label) or not np.array_equal(frame.fold,dev.fold):
                raise ValueError('OOF labels/folds misaligned')
            p=frame[[f'prob_{c}' for c in range(3)]].to_numpy()
            if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-5):
                raise ValueError('Malformed probabilities')
            pred=p.argmax(1)
            if not np.array_equal(frame.prediction,pred):
                raise ValueError('Prediction column differs from probabilities')
            value=float(f1_score(dev.label,pred,labels=[0,1,2],average='macro',zero_division=0))
            reported=json.loads((out/f'{kind}_seed{seed}_metrics.json').read_text())
            if not np.isclose(value,reported['macro_f1'],atol=1e-12) or not np.isclose(accuracy_score(dev.label,pred),reported['accuracy'],atol=1e-12):
                raise ValueError('Metrics do not reconcile')
            recomputed.append(dict(model=kind,seed=seed,macro_f1=value))
    for fold in range(5):
        table=pd.read_csv(out/f'outer{fold}_inner_scores.csv')
        if len(table)!=len(json.loads((out/'benchmark.json').read_text())['candidate_ids'])*9:
            raise ValueError('Incomplete inner search')
        # Independent pandas aggregation rather than the production selection helper.
        ranked=table.groupby('candidate').agg(score=('score','mean'),parameters=('parameters','first')).reset_index()
        ranked=ranked.sort_values(['score','parameters','candidate'],ascending=[False,True,True])
        published=json.loads((out/'outer_selections.json').read_text())[fold]
        if ranked.iloc[0].candidate!=published['candidate']:
            raise ValueError('Selected candidate does not follow preregistered rule')
    final=json.loads((cache/'final_seed42'/'fit.json').read_text())
    model=TextCNN(768,filters=final['cfg']['filters'],dropout=final['cfg']['dropout'])
    weight_path=cache/'final_seed42'/'weights.safetensors'
    if file_hash(weight_path)!=final['weight_sha256']:
        raise ValueError('Final weights changed')
    model.load_state_dict(load_file(str(weight_path)))
    archive=resolve_prediction_archive(cache, out.name)
    with np.load(archive) as a:
        if list(a['classes'])!=[0,1,2] or int(a['seed'])!=42 or set(a['test_ids'])!=test_ids or set(a['oof_ids'])!=set(dev.id):
            raise ValueError('Team archive interface differs')
        original=pd.read_csv(out/'selected_seed42_oof.csv').set_index('id').loc[a['oof_ids']]
        np.testing.assert_allclose(original[[f'prob_{c}' for c in range(3)]],a['oof_proba'],atol=1e-12)
        x,m=load_token_cache(EMB_DIR,a['test_ids'].tolist())
        p=predict_proba(model,x,m,batch_size=64,device='cpu')
        np.testing.assert_allclose(p,a['test_proba'],atol=5e-5,rtol=5e-4)
        maximum=float(np.max(np.abs(p-a['test_proba'])))
        if not np.array_equal(p.argmax(1),a['test_proba'].argmax(1)):
            raise ValueError('Reloaded predicted classes differ')
    result=dict(verified=True,completed_fits=len(records),oof_rows=len(dev),reserved_prediction_rows=len(test_ids),
                independently_recomputed=recomputed,final_cpu_reload_max_abs_difference=maximum,
                labels_scored_on_reserved_set=False,group_boundaries_verified=True,
                selection_rules_verified=True,data_and_split_unchanged=True,
                limitation='Verification establishes implementation/artifact consistency, not independent future-market generalization')
    write_json(out/'independent_verification.json',result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',default='textcnn_refinement_20261112')
    args=p.parse_args()
    if Path(args.run_id).name!=args.run_id:
        p.error('Run ID must be a directory name')
    verify(ROOT/'Member5_file'/'evidence'/'refinement'/args.run_id)
