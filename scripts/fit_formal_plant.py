import argparse,json
from aircraft_recovery.verification.plant_calibration_fit import fit_formal_plant
parser=argparse.ArgumentParser(); parser.add_argument("calibration_directory"); parser.add_argument("configuration"); parser.add_argument("output")
args=parser.parse_args(); model=fit_formal_plant(args.calibration_directory,args.configuration,args.output); print(json.dumps(model.fit_metadata,indent=2,sort_keys=True))
