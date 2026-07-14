"""Generate expert demonstrations, episode splits, and validation report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from aircraft_recovery.config import load_config
from aircraft_recovery.data.demonstrations import DemonstrationConfig, generate_demonstrations
from aircraft_recovery.data.splitting import create_split_manifest
from aircraft_recovery.data.validation import validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/demonstrations.yaml")
    parser.add_argument("--force", action="store_true", help="Replace an existing configured dataset directory.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config, DemonstrationConfig)
    output = Path(config.output_directory)
    if args.force and output.exists():
        resolved = output.resolve()
        allowed_root = (Path.cwd() / "datasets").resolve()
        if not resolved.is_relative_to(allowed_root) or resolved == allowed_root:
            raise ValueError(f"--force may replace only a child of {allowed_root}")
        shutil.rmtree(output)
    metadata = generate_demonstrations(config)
    manifest = create_split_manifest(
        output, config.seed, config.train_fraction, config.validation_fraction,
        config.test_fraction, set(config.ood_categories),
    )
    report = validate_dataset(output)
    print(json.dumps({
        "dataset_identifier": metadata["dataset_identifier"],
        "episodes": metadata["episode_count"],
        "transitions": metadata["transition_count"],
        "split_episode_counts": report["split_episode_counts"],
        "split_manifest_identifier": manifest["identifier"],
        "category_episode_counts": metadata["scenario_category_episode_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
