"""Nontraining regressions: portable artifacts, unbiased controls, durable attempts."""
import json
from pathlib import Path
import pytest
import verify_refinement as legacy


def test_downloaded_archive_resolves_without_original_colab_location(tmp_path):
    cache=tmp_path/'moved'/'run_a';(cache/'preds').mkdir(parents=True)
    target=cache/'preds/run_a.npz';target.write_bytes(b'fixture')
    assert legacy.resolve_prediction_archive(cache,'run_a')==target
    with pytest.raises(FileNotFoundError):legacy.resolve_prediction_archive(cache,'other')
    with pytest.raises(ValueError):legacy.resolve_prediction_archive(cache,'../escape')


def test_screening_always_keeps_control_and_rejects_incomplete_rows():
    import v2_runtime as r
    rows=[{'candidate':c,'inner_fold':k,'seed':42,'score':s,'best_epoch':ep,'parameters':p}
          for c,s,ep,p in [('C0',.5,12,100),('C1',.7,3,32),('C2',.6,7,64)] for k in range(3)]
    assert r.select_entrants(rows,['C0','C1','C2'])==['C0','C1','C2']
    ranked=r.rank_candidates(rows,['C0','C1','C2'],[42])
    assert [x['candidate'] for x in ranked]==['C1','C2','C0']
    assert ranked[2]['epochs']==12  # C0 keeps its OWN epoch count, not winner's 3.
    with pytest.raises(ValueError):r.rank_candidates(rows[:-1],['C0','C1','C2'],[42])
    with pytest.raises(ValueError):r.rank_candidates(rows+[rows[0]],['C0','C1','C2'],[42])


def test_budget_persists_active_time_across_resume(tmp_path):
    import v2_runtime as r
    now=[10.]
    b=r.WallBudget(tmp_path/'budget.json',10,clock=lambda:now[0]);b.start()
    now[0]=14.;b.pause()
    resumed=r.WallBudget(tmp_path/'budget.json',10,clock=lambda:now[0]);resumed.start()
    now[0]=19.;assert resumed.remaining()==1
    now[0]=21.
    with pytest.raises(r.BudgetExceeded):resumed.check()
    assert json.loads((tmp_path/'budget.json').read_text())['used_seconds']>=10
    with pytest.raises(ValueError):r.WallBudget(tmp_path/'budget.json',20)


def test_attempt_failure_is_not_overwritten_and_completed_digest_checked(tmp_path):
    import v2_runtime as r
    store=r.AttemptStore(tmp_path,'protocol-one')
    a=store.begin('fit1',{'seed':42});store.finish(a,'failed',{'error':'test failure'})
    b=store.begin('fit1',{'seed':42});assert a!=b
    (b/'predictions.npy').write_bytes(b'probabilities')
    store.finish(b,'complete',{'prediction_sha256':r.sha256(b/'predictions.npy')})
    assert store.completed('fit1',{'seed':42})==b
    assert json.loads((a/'record.json').read_text())['status']=='failed'
    (b/'predictions.npy').write_bytes(b'changed')
    with pytest.raises(ValueError):store.completed('fit1',{'seed':42})
    with pytest.raises(ValueError):r.AttemptStore(tmp_path,'different-protocol')
