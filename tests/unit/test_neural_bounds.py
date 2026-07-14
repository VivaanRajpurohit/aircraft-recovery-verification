import numpy as np
import pytest
import torch

from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig
from aircraft_recovery.verification.neural_bounds import Interval, affine_bounds, bound_policy_model


def test_interval_rejects_reversed_or_nonfinite_endpoints() -> None:
    with pytest.raises(ValueError):
        Interval(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        Interval(np.array([0.0]), np.array([np.inf]))


def test_affine_interval_contains_sampled_points() -> None:
    torch.manual_seed(12)
    layer = torch.nn.Linear(4, 3)
    bounds = Interval(np.array([-1.0, 0.2, -0.4, 1.0]), np.array([0.5, 0.7, 1.2, 1.5]))
    output = affine_bounds(bounds, layer)
    generator = np.random.default_rng(22)
    samples = generator.uniform(bounds.lower, bounds.upper, size=(1000, 4))
    actual = layer(torch.tensor(samples, dtype=torch.float32)).detach().numpy()
    assert np.all(actual >= output.lower)
    assert np.all(actual <= output.upper)


def test_small_relu_policy_ibp_is_sound_against_sampling() -> None:
    torch.manual_seed(9)
    model = ImitationPolicyNetwork(PolicyArchitectureConfig(hidden_dimensions=[8, 6], activation="relu"))
    lower = np.full(42, -0.15)
    upper = np.full(42, 0.2)
    bounds = bound_policy_model(model, Interval(lower, upper))
    assert np.all(bounds.controls.lower[:3] >= -1.0)
    assert np.all(bounds.controls.upper[:3] <= 1.0)
    assert np.all(bounds.controls.lower[3:] >= 0.0)
    assert np.all(bounds.controls.upper[3:] <= 1.0)

    generator = np.random.default_rng(4)
    samples = generator.uniform(lower, upper, size=(500, 42))
    with torch.no_grad():
        predictions = model(torch.tensor(samples, dtype=torch.float32))
    for values, interval in (
        (predictions["controls"].numpy(), bounds.controls),
        (predictions["mode_logits"].numpy(), bounds.mode_logits),
        (predictions["targets_normalized"].numpy(), bounds.targets_normalized),
    ):
        assert interval is not None
        assert np.all(values >= interval.lower)
        assert np.all(values <= interval.upper)


def test_non_relu_policy_is_explicitly_unsupported() -> None:
    model = ImitationPolicyNetwork(PolicyArchitectureConfig(hidden_dimensions=[4], activation="silu"))
    with pytest.raises(ValueError, match="ReLU"):
        bound_policy_model(model, Interval(np.zeros(42), np.ones(42)))
