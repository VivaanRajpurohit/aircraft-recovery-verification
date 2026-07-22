import argparse,json
from aircraft_recovery.verification.plant_validation import validate_formal_plant
parser=argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("calibration_directory"); parser.add_argument("validation_directory"); parser.add_argument("output")
args=parser.parse_args(); print(json.dumps(validate_formal_plant(args.model,args.calibration_directory,args.validation_directory,args.output),indent=2,sort_keys=True))
