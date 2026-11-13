"""Independent review of published predictions; no embedding cache or model download."""
from pathlib import Path
import json
import hashlib
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_recall_fscore_support,confusion_matrix
from src.data import load_clean,load_splits
from src.config import CLEAN_DIR
from v2_verification import bootstrap_recheck

def _json(path):return json.loads(Path(path).read_text())


def _same(actual,expected,message):
    if not np.allclose(actual,expected,atol=1e-12,rtol=0):raise ValueError(message)


def _frame(path,dev):
    frame=pd.read_csv(path)
    if frame.id.tolist()!=dev.id.tolist():raise ValueError('Prediction IDs/order differ from official development rows')
    for column in ['label','fold','group_id']:
        if column in frame and not np.array_equal(frame[column],dev[column]):raise ValueError(f'Prediction {column} differs')
    p=frame[[f'prob_{c}' for c in range(3)]].to_numpy()
    if not np.isfinite(p).all() or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(1),1,atol=1e-5):raise ValueError('Invalid probabilities')
    pred=p.argmax(1)
    if not np.array_equal(pred,frame.prediction):raise ValueError('Prediction class does not match probabilities')
    prec,rec,f1,support=precision_recall_fscore_support(dev.label,pred,labels=[0,1,2],zero_division=0)
    return p,dict(accuracy=float(accuracy_score(dev.label,pred)),macro_precision=float(prec.mean()),macro_recall=float(rec.mean()),
        macro_f1=float(f1.mean()),confusion_matrix=confusion_matrix(dev.label,pred,labels=[0,1,2]).tolist(),
        precision=prec.tolist(),recall=rec.tolist(),f1=f1.tolist(),support=support.tolist())


def verify_evidence(directory=None):
    from adopted_model import default_evidence_directory
    directory=Path(directory) if directory is not None else default_evidence_directory()
    clean=load_clean().sort_values('id');splits=load_splits(clean=clean)
    joined=splits[['id','split','fold']].merge(clean,on='id',validate='one_to_one').sort_values('id')
    dev=joined.loc[joined.split=='train'].reset_index(drop=True);reserved=joined.loc[joined.split=='test']
    if set(dev.group_id)&set(reserved.group_id) or dev.groupby('group_id').fold.nunique().max()>1:raise ValueError('Official group isolation broken')
    protocol=_json(directory/'protocol.json')
    for path,field in [(CLEAN_DIR/'clean_v2.csv','data_sha256'),(ROOT/'splits/folds_v1.csv','split_sha256')]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol[field]:raise ValueError('Published evidence uses different data/splits')
    baseline,bm=_frame(directory/'baseline_oof.csv',dev)
    original=pd.read_csv(ROOT/'Member5_file/evidence/baseline_oof.csv')
    _same(baseline,original[[f'prob_{c}' for c in range(3)]].to_numpy(),'Baseline probabilities changed')
    comparison=pd.read_csv(directory/'comparison.csv');scores=[];probabilities={}
    for kind in ['selected','C0']:
        for seed in [42,17,2026]:
            p,m=_frame(directory/f'{kind}_seed{seed}_oof.csv',dev);probabilities[kind,seed]=p
            reported=_json(directory/f'{kind}_seed{seed}_metrics.json')
            for field in ['accuracy','macro_precision','macro_recall','macro_f1','confusion_matrix']:_same(m[field],reported[field],f'{kind}/{seed} {field} differs')
            for c in range(3):
                for field,key in [('precision','precision'),('recall','recall'),('f1','f1-score'),('support','support')]:
                    _same(m[field][c],reported['classification_report'][str(c)][key],'Per-class metric differs')
            row=comparison.loc[(comparison.model==kind)&(comparison.seed==seed)]
            if len(row)!=1:raise ValueError('Missing/duplicate comparison row')
            for field in ['accuracy','macro_precision','macro_recall','macro_f1']:_same(m[field],row[field].iloc[0],'Comparison table differs')
            for c in range(3):_same(m['f1'][c],row[f'f1_{c}'].iloc[0],'Comparison class F1 differs')
            scores.append(dict(model=kind,seed=seed,accuracy=m['accuracy'],macro_f1=m['macro_f1']))
    if len(comparison)!=6:raise ValueError('Unexpected comparison rows')
    conclusion=_json(directory/'conclusion.json')
    selected=np.array([s['macro_f1'] for s in scores if s['model']=='selected']);control=np.array([s['macro_f1'] for s in scores if s['model']=='C0'])
    class_deltas=[comparison.loc[comparison.model=='selected',f'f1_{c}'].mean()-comparison.loc[comparison.model=='C0',f'f1_{c}'].mean() for c in range(3)]
    tradeoffs=[];saved_tradeoffs=pd.read_csv(directory/'class_tradeoffs.csv')
    if saved_tradeoffs.label.tolist()!=[0,1,2]:raise ValueError('Class tradeoff coverage differs')
    for c,name in enumerate(['Neutral','Positive','Negative']):
        row=dict(label=c,name=name,selected_mean_f1=float(comparison.loc[comparison.model=='selected',f'f1_{c}'].mean()),
                 C0_mean_f1=float(comparison.loc[comparison.model=='C0',f'f1_{c}'].mean()),delta=float(class_deltas[c]))
        if saved_tradeoffs.iloc[c]['name']!=name:raise ValueError('Class name differs')
        for field in ['selected_mean_f1','C0_mean_f1','delta']:_same(saved_tradeoffs.iloc[c][field],row[field],'Class tradeoff differs')
        tradeoffs.append(row)
    for key,value in dict(selected_mean=selected.mean(),control_mean=control.mean(),mean_delta=(selected-control).mean(),positive_seeds=int((selected>control).sum()),baseline_macro_f1=bm['macro_f1'],class_mean_deltas=class_deltas).items():_same(conclusion[key],value,'Conclusion differs')
    expected=bool((selected-control).mean()>=.01 and (selected>control).sum()>=2 and min(class_deltas)>=-.02)
    if conclusion['practical_target_met']!=expected:raise ValueError('Practical target decision differs')
    intervals=_json(directory/'bootstrap.json')
    if len(intervals)!=6:raise ValueError('Bootstrap coverage differs')
    for seed in [42,17,2026]:
        for kind in ['C0','baseline']:
            controlp=baseline if kind=='baseline' else probabilities['C0',seed]
            value=bootstrap_recheck(dev.label.to_numpy(),dev.group_id.to_numpy(),probabilities['selected',seed],controlp)
            match=[r for r in intervals if r['seed']==seed and r['comparison']==f'selected minus {kind}']
            if len(match)!=1 or match[0]['resamples']!=2000 or match[0]['unit']!='group_id':raise ValueError('Bootstrap protocol differs')
            for key,v in value.items():_same(match[0][key],v,'Bootstrap differs')
    advantage=all(r['lower']>0 for r in intervals if r['comparison']=='selected minus C0')
    if conclusion['statistical_advantage_confirmed']!=advantage:raise ValueError('Statistical conclusion differs')
    mvp=[]
    for summary in sorted((ROOT/'Member5_file/evidence').glob('textcnn__*/summary.json')):
        saved=_json(summary);_,m=_frame(summary.parent/'oof_predictions.csv',dev)
        for field in ['accuracy','macro_precision','macro_recall','macro_f1']:_same(m[field],saved['metrics'][field],'Original MVP metric differs')
        mvp.append(dict(run_id=saved['run_id'],macro_f1=m['macro_f1']))
    return dict(verified=True,development_rows=len(dev),reserved_rows=len(reserved),scores=scores,original_mvp=mvp,class_tradeoffs=tradeoffs,
        full_training_archive_checked=False,test_labels_scored=False,
        scope='Recomputed published OOF metrics and paired group bootstrap; historical fit/weight verification is not repeated')


if __name__=='__main__':print(json.dumps(verify_evidence(),indent=2))
