from aircraft_recovery.config import load_config
from aircraft_recovery.safety import SafetyConfig
from aircraft_recovery.verification import run_formal_verification


def test_formal_report_proves_invariants_and_emits_counterexample(tmp_path) -> None:
    config = load_config("configs/safety/default.yaml", SafetyConfig)
    report = run_formal_verification(config, tmp_path)
    assert report["property_coverage"] == {
        "total_obligations": 8,
        "verified_within_bounds": 8,
        "failed": 0,
        "unknown": 0,
    }
    unsafe = report["examples"]["unsafe_action_counterexample"]
    assert unsafe["status"] == "unsafe"
    assert unsafe["counterexample"]
    assert (tmp_path / "verification_report.json").is_file()

