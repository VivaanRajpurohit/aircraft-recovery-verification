import pytest
import torch

from aircraft_recovery.training.models import ImitationPolicyNetwork, PolicyArchitectureConfig


@pytest.mark.parametrize(
    ("dimensions", "minimum", "maximum"),
    [([128, 64], 10_000, 100_000), ([256, 128], 40_000, 1_000_000), ([1024, 1024, 512], 1_000_000, 5_000_000)],
)
def test_configurable_model_parameter_counts(dimensions, minimum, maximum) -> None:
    model = ImitationPolicyNetwork(PolicyArchitectureConfig(hidden_dimensions=dimensions))
    assert minimum < model.parameter_count < maximum


def test_network_requires_exactly_42_features() -> None:
    model = ImitationPolicyNetwork(PolicyArchitectureConfig())
    output = model(torch.zeros(3, 42))
    assert output["controls"].shape == (3, 5)
    with pytest.raises(RuntimeError):
        model(torch.zeros(3, 41))
    with pytest.raises(ValueError, match="exactly 42"):
        PolicyArchitectureConfig(input_dim=43)


def test_action_heads_are_bounded() -> None:
    model = ImitationPolicyNetwork(PolicyArchitectureConfig())
    controls = model(torch.randn(100, 42) * 100.0)["controls"]
    assert torch.all((-1.0 <= controls[:, :3]) & (controls[:, :3] <= 1.0))
    assert torch.all((0.0 <= controls[:, 3:]) & (controls[:, 3:] <= 1.0))


def test_default_architecture_is_two_relu_hidden_layers() -> None:
    config = PolicyArchitectureConfig()
    model = ImitationPolicyNetwork(config)
    assert config.hidden_dimensions == [256, 128]
    assert config.activation == "relu"
    assert sum(isinstance(layer, torch.nn.ReLU) for layer in model.trunk) == 2
