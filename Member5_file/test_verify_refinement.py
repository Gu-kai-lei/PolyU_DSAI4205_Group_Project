"""A verifier must reject leaking or mislabelled fit provenance."""
import pandas as pd
import pytest
from verify_refinement import validate_fit_record


def test_inner_fit_cannot_contain_outer_validation_or_reserved_ids():
    dev=pd.DataFrame({'id':[1,2,3,4], 'group_id':[1,2,3,4], 'fold':[0,1,1,2]})
    good={'key':'outer0_C0_i0_s42','provenance':{'fit_ids':[2,3],'valid_ids':[4],'purpose':'inner'}}
    validate_fit_record(good,dev,{9})
    for forbidden in [1,9]:
        bad={**good,'provenance':{**good['provenance'],'fit_ids':[forbidden,3]}}
        with pytest.raises(ValueError):
            validate_fit_record(bad,dev,{9})


def test_inner_fit_cannot_split_related_group():
    dev=pd.DataFrame({'id':[1,2,3], 'group_id':[8,8,9], 'fold':[1,1,0]})
    row={'key':'outer0_C0_i0_s42','provenance':{'fit_ids':[1],'valid_ids':[2],'purpose':'inner'}}
    with pytest.raises(ValueError,match='group'):
        validate_fit_record(row,dev,set())


def test_outer_refit_must_train_on_entire_outer_training_partition():
    dev=pd.DataFrame({'id':[1,2,3], 'group_id':[1,2,3], 'fold':[0,1,2]})
    row={'key':'outer0_selected_s42','provenance':{'fit_ids':[2,3],'valid_ids':[1],'purpose':'outer_refit'}}
    validate_fit_record(row,dev,set())
    row['provenance']['fit_ids']=[2]
    with pytest.raises(ValueError,match='complete'):
        validate_fit_record(row,dev,set())
