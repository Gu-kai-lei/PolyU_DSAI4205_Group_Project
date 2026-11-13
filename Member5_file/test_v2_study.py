"""Pure orchestration checks, no optimizer steps or model training."""
from collections import Counter
import numpy as np
import pytest


def test_two_stage_search_retains_control_with_own_epochs_and_reuses_seed42():
    import diagnose_refine as d
    calls=[]
    def runner(candidate,fold,seed):
        calls.append((candidate,fold,seed))
        score={'C0':.4,'C1':.5,'C2':.8,'C3':.7,'C4':.3,'C5':.2}[candidate]
        return dict(score=score,best_epoch=12 if candidate=='C0' else 4,parameters=100)
    result=d.search_partition(runner)
    assert result['entrants']==['C0','C2','C3']
    assert result['winner']['candidate']=='C2' and result['winner']['epochs']==4
    assert result['control']['epochs']==12
    assert len(calls)==36 and len(set(calls))==36
    assert Counter(c for c,_,_ in calls)=={'C0':9,'C1':3,'C2':9,'C3':9,'C4':3,'C5':3}


def test_reserved_labels_are_not_required_for_team_archive(tmp_path):
    import diagnose_refine as d
    path=tmp_path/'run.npz';p=np.eye(3,dtype=float)
    d.save_team_archive(path,'run',[1,2,3],p,[7,8],p[:2])
    with np.load(path) as a:
        assert list(a['classes'])==[0,1,2] and int(a['seed'])==42
        assert set(a.files)=={'oof_ids','oof_proba','test_ids','test_proba','classes','run_id','data_version','split_version','seed'}
    with pytest.raises(FileExistsError):d.save_team_archive(path,'different',[1,2,3],p,[7,8],p[:2])


def test_result_snapshot_excludes_unrelated_files_and_refuses_symlinks(tmp_path):
    import tarfile
    import diagnose_refine as d
    root=tmp_path/'run';(root/'evidence').mkdir(parents=True)
    (root/'evidence/result.json').write_text('{}');(root/'credentials.json').write_text('secret')
    dest=tmp_path/'result.tar.gz';d.snapshot_bundle(root,dest)
    with tarfile.open(dest) as archive:
        assert 'run/evidence/result.json' in archive.getnames()
        assert 'run/credentials.json' not in archive.getnames()
    (root/'evidence/link').symlink_to(tmp_path/'unrelated')
    with pytest.raises(ValueError):d.snapshot_bundle(root,dest)


def test_checkpoint_budget_is_paused_for_restore(tmp_path):
    import tarfile,json
    import diagnose_refine as d
    from v2_runtime import WallBudget
    root=tmp_path/'run';root.mkdir();clock=[0.]
    budget=WallBudget(root/'budget.json',10,clock=lambda:clock[0]);budget.start();clock[0]=4.
    target=d.snapshot_bundle(root,tmp_path/'snapshot.tar.gz',budget)
    with tarfile.open(target) as archive:
        state=json.load(archive.extractfile('run/budget.json'))
    assert state['used_seconds']==4 and state['active'] is False


def test_independent_selection_rejects_using_winner_epoch_for_control():
    from v2_verification import check_choice
    rows=[dict(candidate=c,inner_fold=k,seed=s,score=value,best_epoch=e,parameters=p)
          for c,value,e,p in [('C0',.5,12,100),('C1',.7,3,32)] for k in range(3) for s in [42,17,2026]]
    choice=dict(rows=rows,entrants=['C0','C1'],winner=dict(candidate='C1',score=.7,epochs=3,parameters=32),
                control=dict(candidate='C0',score=.5,epochs=3,parameters=100))
    with pytest.raises(ValueError):check_choice(choice,['C0','C1'])
    choice['control']['epochs']=12;assert len(check_choice(choice,['C0','C1']))==18


def test_final_search_does_not_overwrite_final_choice(tmp_path):
    import diagnose_refine as d
    from v2_runtime import read_json,write_json
    study=d.FrozenStudy.__new__(d.FrozenStudy);study.evidence=tmp_path
    import pandas as pd
    study.dev=pd.DataFrame({'label':np.tile([0,1,2],9),'group_id':np.arange(27)})
    study.y=study.dev.label.to_numpy()
    def fake_fit(key,fit,valid,cfg,seed):
        return {'probabilities':np.eye(3)[study.y[valid]],'record':{'best_epoch':2,'parameters':100}}
    study._fit=fake_fit
    selection=study._select(np.arange(27),'final')
    write_json(tmp_path/'final_selection.json',selection['winner'])
    assert read_json(tmp_path/'final_search_selection.json')==selection
    assert read_json(tmp_path/'final_selection.json')==selection['winner']


def test_console_logs_are_preserved_in_result_bundle(tmp_path):
    import tarfile
    from diagnose_refine import snapshot_bundle
    root=tmp_path/'run';root.mkdir();(root/'console_run_123.log').write_text('failure details')
    with tarfile.open(snapshot_bundle(root,tmp_path/'out.tar.gz')) as archive:
        assert 'run/console_run_123.log' in archive.getnames()


def test_failed_snapshot_keeps_previous_checkpoint(tmp_path):
    from diagnose_refine import snapshot_bundle
    root=tmp_path/'run';(root/'evidence').mkdir(parents=True)
    destination=tmp_path/'checkpoint.tar.gz';destination.write_bytes(b'previous checkpoint')
    (root/'evidence/link').symlink_to(tmp_path/'outside')
    with pytest.raises(ValueError):snapshot_bundle(root,destination)
    assert destination.read_bytes()==b'previous checkpoint'


def test_independent_epoch_check_rejects_nonbest_checkpoint():
    from v2_verification import check_best_epoch
    record={'best_epoch':1,'best_macro_f1':.4,'history':[
        {'epoch':1,'valid':{'macro_f1':.4}}, {'epoch':2,'valid':{'macro_f1':.7}},
        {'epoch':3,'valid':{'macro_f1':.7}}]}
    with pytest.raises(ValueError):check_best_epoch(record)
    record.update(best_epoch=2,best_macro_f1=.7);assert check_best_epoch(record) is None
    record['history'][2]['epoch']=5
    with pytest.raises(ValueError):check_best_epoch(record)


def test_independent_group_bootstrap_matches_reference():
    from v2_verification import bootstrap_recheck
    from refine import paired_group_bootstrap
    y=np.tile([0,1,2],8);groups=np.repeat(np.arange(12),2)
    a=np.eye(3)[y];b=np.eye(3)[np.roll(y,2)]
    a[::5]=b[::5]
    expected=paired_group_bootstrap(y,groups,a,b,n=200)
    actual=bootstrap_recheck(y,groups,a,b,n=200)
    for field in ['delta','lower','upper']:
        assert actual[field]==pytest.approx(expected[field],abs=1e-12)


def test_independent_tie_ranking_is_invariant_to_fold_permutation():
    from v2_runtime import rank_candidates
    from v2_verification import independently_rank
    rows=[dict(candidate=c,inner_fold=i,seed=42,score=s,best_epoch=3,parameters=p)
          for c,scores,p in [('C0',[.2,.21,.24],100),('C1',[.21,.24,.2],32)]
          for i,s in enumerate(scores)]
    expected=rank_candidates(rows,['C0','C1'],[42])
    assert expected[0]['candidate']=='C1'
    assert independently_rank(rows,['C0','C1'],[42])==expected


def test_auxiliary_gate_requires_raw_attempts_not_summary_flags(tmp_path):
    from v2_verification import verify_auxiliary
    from v2_runtime import write_json
    write_json(tmp_path/'evidence/diagnostics.json',{'gate_passed':True})
    write_json(tmp_path/'evidence/benchmark.json',{'synthetic_only':True})
    with pytest.raises(ValueError,match='attempt'):
        verify_auxiliary(tmp_path,{},None)


def test_auxiliary_verifier_accepts_complete_fixture_then_detects_corruption(tmp_path):
    from refine import data,inner_partitions
    from v2_training import learning_subsets
    from v2_runtime import AttemptStore,CANDIDATES,signature,sha256,write_json
    from v2_verification import verify_auxiliary
    _,dev,_=data();protocol={'fixture':True};outer=dev.loc[dev.fold!=0].reset_index(drop=True)
    train,valid=inner_partitions(outer)[0];frame=outer.iloc[train].reset_index(drop=True);validation=outer.iloc[valid]
    tiny=[];used=set()
    for c in range(3):
        for _,row in frame.loc[frame.label==c].iterrows():
            if row.group_id in used:continue
            tiny.append(int(row.id));used.add(row.group_id)
            if sum(int(dev.set_index('id').loc[i,'label'])==c for i in tiny)==20:break
    stores={key:AttemptStore(tmp_path/key,signature(protocol)) for key in ['diagnostics','calibration']}
    def save(directory,key,meta,history,p,**extra):
        folder=stores[directory].begin(key,meta)
        np.save(folder/'predictions.npy',p);(folder/'weights.safetensors').write_bytes(b'fixture checksum only')
        stores[directory].finish(folder,'complete',dict(history=history,prediction_sha256=sha256(folder/'predictions.npy'),weight_sha256=sha256(folder/'weights.safetensors'),**extra))
        return folder
    memory_cfg={**CANDIDATES['C1'],'dropout':0.,'weight_decay':0.,'max_epochs':200}
    labels=dev.set_index('id').loc[tiny,'label'].to_numpy()
    save('diagnostics','memory60',dict(fit_ids=tiny,valid_ids=tiny,seed=42,max_updates=200,target_f1=.95,cfg=memory_cfg),
         [{'epoch':1,'train':{'macro_f1':1.},'valid':None}],np.eye(3)[labels],updates=2,target_met=True)
    curves=[]
    for c in ['C0','C3']:
        for fraction,ix in learning_subsets(frame).items():
            history=[{'epoch':1,'train':{'macro_f1':1.},'valid':{'macro_f1':1.}}]
            save('diagnostics',f'curve_{c}_{int(fraction*100)}',dict(fit_ids=frame.iloc[ix].id.tolist(),valid_ids=validation.id.tolist(),cfg=CANDIDATES[c],seed=42),
                 history,np.eye(3)[validation.label],best_epoch=1,best_macro_f1=1.)
            curves.append(dict(candidate=c,fraction=fraction,n_train=len(ix),history=history))
    for i in range(3):
        history=[{'epoch':j,'seconds':2.,'train':{'n':1222},'valid':{'n':303}} for j in range(1,5)]
        save('calibration',f'fixed_repeat{i}',dict(cfg={**CANDIDATES['C0'],'max_epochs':4,'patience':10},seed=42),history,np.ones((303,3))/3)
    write_json(tmp_path/'evidence/diagnostics.json',dict(gate_passed=True,memory_train_f1=1.,memory_updates=2,curves=curves))
    write_json(tmp_path/'evidence/benchmark.json',dict(synthetic_only=True,median_seconds_per_epoch=2.,range_seconds_per_epoch=[2.,2.]))
    verify_auxiliary(tmp_path,protocol,dev)
    next((tmp_path/'diagnostics').glob('fits/memory60/attempt_*/predictions.npy')).write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='artifact'):verify_auxiliary(tmp_path,protocol,dev)
