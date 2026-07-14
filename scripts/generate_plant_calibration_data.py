"""Generate deterministic formal-plant calibration or validation transitions."""
import argparse
import json
from aircraft_recovery.verification.plant_calibration import generate_dataset,load_dataset_config

parser=argparse.ArgumentParser(); parser.add_argument("config")
args=parser.parse_args(); print(json.dumps(generate_dataset(load_dataset_config(args.config)),indent=2,sort_keys=True))
