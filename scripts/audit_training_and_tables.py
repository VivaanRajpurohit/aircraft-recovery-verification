"""Check development predictions, supplementary tables and formal sensitivity."""
from pathlib import Path
import csv,json,math,copy
import numpy as np
import torch
from aircraft_recovery.verification.plant_validation import validate_formal_plant
from aircraft_recovery.training.v2_1a_final_training import ArchitectureScreenDataSource,_read_split,_normalize,_development_metrics_with_saturation,V21AControlPolicyNetwork,WINNING_ARCHITECTURE

root=Path('.');out=root/'results/publication_audit';out.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(p.read_text())
def close(a,b,tol=1e-10):assert abs(a-b)<tol,(a,b)
def csvrows(name):return list(csv.DictReader((root/'results/supplementary_architecture_comparison_20261011'/name).open()))
report={};rr=csvrows('architecture_challenge_864_condition_records.csv');overall=csvrows('architecture_challenge_overall.csv');tiers=csvrows('architecture_challenge_by_tier.csv');assert len(rr)==864
checked=[]
for row in overall+tiers:
    sub=[r for r in rr if r['architecture_id']==row['architecture_id'] and r['condition']==row['condition'] and ('tier' not in row or r['tier']==row['tier'])]
    vals=[float(r['recovery_onset_seconds']) for r in sub];assert len(sub)==int(row['flight_count']);close(float(np.mean(vals)),float(row['mean_recovery_onset_seconds']));
    if 'onset_sample_sd_seconds' in row:close(float(np.std(vals,ddof=1)),float(row['onset_sample_sd_seconds']))
    assert sum(r['violation_flight']=='True' for r in sub)==int(row['violation_flights']);assert sum(r['strict_recovery_success']=='True' for r in sub)==int(row['strict_recovery_count'])
    checked.append(row)
report['supplementary_table_rows_verified']=len(checked);report['supplementary_overall']=overall
formal=root/'results/formal_plant_smoke_v1';sensitivity={}
base=read(formal/'formal_plant_model.json')
for name,state,index,delta in [('bank_response','bank_deg',2,.5),('throttle_response','airspeed_kts',3,.5),('vertical_speed_scale','vertical_speed_fpm',0,1),('clearance_integration','terrain_clearance_ft',2,.001),('remove_residuals',None,None,None)]:
    model=copy.deepcopy(base)
    if state:model['coefficients'][state][index]+=delta
    else:model['residual_bounds']={k:[0.,0.] for k in model['residual_bounds']}
    mp=out/(name+'.json');mp.write_text(json.dumps(model));r=validate_formal_plant(mp,formal/'calibration',formal/'validation',out/(name+'_validation.json'))
    sensitivity[name]={'one_step_failures':len(r['one_step']['containment_failures']),'rollout_failures':len(r['rollout']['first_containment_failures'])}
report['formal_perturbations']=sensitivity
(out/'training_tables.json').write_text(json.dumps(report,indent=2));print('Supplementary tables and formal perturbations checked',sensitivity,flush=True)
torch.set_num_threads(1)
approved=root/'datasets/approved_policy_data/final_policy_approved_v2_1';normalization=read(approved/'preprocessing/train_only_normalization.json');source=ArchitectureScreenDataSource(approved)
development=_read_split(source.development_records(),expected_split='development',expected_count=37440,expected_flights=144)
obs=torch.from_numpy(_normalize(development.observations,normalization));targets=torch.from_numpy(development.controls)
summary=read(root/'results/final_five_seed_training_v2_1a/five_seed_training_summary.json');verified=[]
for run in summary['individual_seed_runs']:
    checkpoint=torch.load(root/run['checkpoint_path'],map_location='cpu',weights_only=False);model=V21AControlPolicyNetwork(WINNING_ARCHITECTURE);model.load_state_dict(checkpoint['model_state_dict']);model.eval();m=_development_metrics_with_saturation(model,obs,targets,512)
    errors={k:abs(m[k]-run['development_metrics'][k]) for k in ['mse','mae','rmse']}
    assert max(errors.values())<1e-7,errors
    verified.append({'seed':run['seed'],'cpu_development_metrics':{k:m[k] for k in errors},'absolute_difference_from_frozen_gpu':errors})
    print('Development checkpoint verified',run['seed'],flush=True)
report['five_seed_reload']=verified;report['development_across_seed_recalculated']={k:{'mean':float(np.mean([r['development_metrics'][k] for r in summary['individual_seed_runs']])),'sample_sd':float(np.std([r['development_metrics'][k] for r in summary['individual_seed_runs']],ddof=1))} for k in ['mse','mae','rmse']}
report['status']='PASS';(out/'training_tables.json').write_text(json.dumps(report,indent=2))
