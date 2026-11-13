"""Local tests never fit a model; real backward/fit tests are opt-in Colab gates."""
import os
import random
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn


def test_metric_observation_preserves_rng_and_every_module_mode():
    import v2_training as t
    class Noisy(nn.Module):
        def __init__(self):
            super().__init__();self.linear=nn.Linear(4,3);self.drop=nn.Dropout(.5)
        def forward(self,x,mask=None):return self.linear(self.drop(x.mean(1)))+torch.rand(len(x),3)*.001
    torch.manual_seed(42);model=Noisy().train();model.drop.eval()
    x=np.ones((6,5,4),dtype='float32');y=np.array([0,1,2,0,1,2]);m=np.ones((6,5),bool)
    modes=[p.training for p in model.modules()];state=torch.random.get_rng_state().clone()
    ns=np.random.get_state();ps=random.getstate()
    result=t.observe(model,x,y,m,'cpu')
    assert [p.training for p in model.modules()]==modes
    assert torch.equal(state,torch.random.get_rng_state())
    assert np.array_equal(ns[1],np.random.get_state()[1]) and ps==random.getstate()
    assert result['n']==6 and np.isfinite(result['cross_entropy']) and len(result['recall'])==3


def test_learning_subsets_are_nested_complete_groups_and_deterministic():
    import v2_training as t
    frame=pd.DataFrame({'id':range(108),'group_id':np.repeat(range(54),2),'label':np.repeat(np.arange(54)%3,2)})
    parts=t.learning_subsets(frame)
    assert set(parts[.25])<set(parts[.5])<set(parts[1.])
    assert set(parts[1.])==set(range(108))
    for ix in parts.values():
        assert set(frame.iloc[ix].label)=={0,1,2}
        assert all(len(g)==2 for _,g in frame.iloc[ix].groupby('group_id'))
    assert parts==t.learning_subsets(frame)


@pytest.mark.skipif(os.getenv('M5_CLOUD_FITS')!='1',reason='User runs all model-fitting checks in Colab')
def test_cloud_real_fit_logs_reload_and_resume(tmp_path):
    import v2_training as t
    from v2_runtime import AttemptStore,WallBudget,CANDIDATES
    assert torch.cuda.is_available(),'Cloud fitting gate requires GPU'
    x=np.random.default_rng(42).normal(size=(12,5,8)).astype('float32');m=np.ones((12,5),bool);y=np.arange(12)%3
    store=AttemptStore(tmp_path/'run','unitprotocol');budget=WallBudget(tmp_path/'budget.json',60);budget.start()
    cfg={**CANDIDATES['C1'],'filters':4,'max_epochs':2,'patience':2}
    args=(store,'tiny',{'purpose':'test'},x[:9],y[:9],m[:9],x[9:],y[9:],m[9:],cfg,42,'cuda',budget)
    a=t.run_fit(*args);b=t.run_fit(*args)
    assert a['folder']==b['folder'] and b['resumed']
    assert {'train','valid','gradient_norm','lr','seconds','peak_cuda_bytes'}<=set(a['record']['history'][0])
    assert (a['folder']/'weights.safetensors').exists()
    from mvp import TextCNN
    from safetensors.torch import load_file
    model=TextCNN(8,filters=4,dropout=cfg['dropout']).to('cuda')
    model.load_state_dict(load_file(str(a['folder']/'weights.safetensors')))
    np.testing.assert_allclose(t.probabilities(model,x[9:],m[9:],'cuda'),a['probabilities'],atol=1e-7,rtol=0)
    np.testing.assert_array_equal(a['probabilities'],b['probabilities'])
    budget.pause()
