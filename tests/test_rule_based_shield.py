"""Penalty magnitudes suppress restricted actions without reversing their sign."""
from dataclasses import replace

import pytest
import torch

from shield.rule_based_shield import RuleBasedShield, RuleBasedShieldConfig


@pytest.mark.parametrize("field", ["logit_penalty", "rescue_logit_penalty"])
@pytest.mark.parametrize("value", [-10., float("nan"), float("inf"), True])
def test_invalid_penalty_magnitudes_are_rejected(field, value):
    with pytest.raises(ValueError, match="finite nonnegative"):
        RuleBasedShield(config=replace(RuleBasedShieldConfig(), **{field: value}))


def test_direct_penalty_argument_cannot_reverse_suppression():
    with pytest.raises(ValueError, match="finite nonnegative"):
        RuleBasedShield(logit_penalty=-10.)


@pytest.mark.parametrize("penalty", [0., 1., 10., 20.])
def test_low_glucose_bolus_offsets_are_nonpositive(penalty):
    obs = torch.zeros(14)
    obs[0] = 80.
    adjusted = RuleBasedShield(logit_penalty=penalty).apply(obs, torch.zeros(10), [5, 5])
    torch.testing.assert_close(adjusted[:5], torch.tensor([0., *([-penalty] * 4)]))
