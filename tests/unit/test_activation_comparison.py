from aircraft_recovery.config import load_config
from aircraft_recovery.evaluation import ActivationComparisonConfig


def test_activation_comparison_declares_relu_and_silu() -> None:
    config = load_config(
        "configs/evaluation/activation_comparison.yaml", ActivationComparisonConfig
    )
    assert config.activations == ["relu", "silu"]
