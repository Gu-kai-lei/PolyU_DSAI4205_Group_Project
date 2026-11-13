import json
from pathlib import Path
import pytest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def test_default_version_is_users_c3_experiment():
    from adopted_model import selected_version
    from release import verify_evidence
    selected=selected_version()
    assert selected['model']['candidate']=='C3' and selected['model']['epochs']==9
    assert selected['model']['parameters']==689155
    result=verify_evidence()
    scores=[s['macro_f1'] for s in result['scores'] if s['model']=='selected']
    assert np.mean(scores)==pytest.approx(.6166861211377567,abs=1e-12)
    assert selected['statistical_advantage_confirmed'] is False


def test_adopted_loader_rejects_wrong_weight_file(tmp_path):
    from adopted_model import load_adopted_model
    file=tmp_path/'wrong.safetensors';file.write_bytes(b'not the adopted C3')
    with pytest.raises(ValueError,match='checksum'):load_adopted_model(file)


def test_adopted_profile_matches_actual_four_candidate_record():
    from adopted_model import selected_version,default_evidence_directory
    profile=selected_version();benchmark=json.loads((default_evidence_directory()/'benchmark.json').read_text())
    assert profile['historical_candidates_executed']==benchmark['candidate_ids']==['C0','C1','C3','C7']


def test_selected_evidence_cannot_silently_point_to_later_c1(tmp_path):
    from adopted_model import selected_version
    value=selected_version();value['evidence_directory']='local_refinement_20261113'
    p=tmp_path/'wrong.json';p.write_text(json.dumps(value))
    with pytest.raises(ValueError):selected_version(p)


@pytest.mark.skipif(not (ROOT/'data/models/textcnn_c3_v1.safetensors').exists(),reason='Audited model artifacts are local, not in Git')
def test_original_c3_weights_match_all_saved_reserved_predictions():
    from adopted_model import selected_version,predict_cached
    from src.config import DATA_ROOT
    value=selected_version()
    with np.load(DATA_ROOT/'preds'/f'{value["original_run_id"]}.npz') as saved:
        probabilities=predict_cached(saved['test_ids'].tolist())
        np.testing.assert_allclose(probabilities,saved['test_proba'],atol=5e-5,rtol=5e-4)
        np.testing.assert_array_equal(probabilities.argmax(1),saved['test_proba'].argmax(1))


def test_team_probability_archive_rejects_wrong_file(tmp_path):
    from adopted_model import verify_team_archive
    p=tmp_path/'wrong.npz';p.write_bytes(b'not the adopted archive')
    with pytest.raises(ValueError,match='checksum'):verify_team_archive(p)
