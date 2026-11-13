import numpy as np
import pytest
import torch

from mvp import TextCNN, validate_probabilities


def test_padding_and_short_sequences():
    torch.manual_seed(42)
    model = TextCNN(8).eval()
    for length in [0, 1, 2, 4, 8]:
        x = torch.randn(2, max(length, 1), 8)
        mask = torch.arange(x.shape[1])[None, :].expand(2, -1) < length
        extended = torch.cat([x, torch.randn(2, 7, 8) * 100], dim=1)
        extended_mask = torch.cat([mask, torch.zeros(2, 7, dtype=torch.bool)], dim=1)
        with torch.no_grad():
            a, b = model(x, mask), model(extended, extended_mask)
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize('bad', [np.array([[np.nan, 0., 1.]]),
                                np.array([[-1., 1., 1.]]), np.ones((1, 3))])
def test_bad_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        validate_probabilities([1], bad)


def test_duplicate_ids_rejected():
    with pytest.raises(ValueError):
        validate_probabilities([1, 1], np.ones((2, 3)) / 3)
