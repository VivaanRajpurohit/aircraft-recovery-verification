"""Recalculate frozen publication evidence; never overwrite historical results.

Run from repository root after extracting the versioned research-data release.
This replays recorded actions through the original simulator, not a second model.
"""
from pathlib import Path
import argparse, collections, hashlib, itertools, json, math
import numpy as np
import torch
from aircraft_recovery.config import AircraftConfig, load_config, load_yaml
from aircraft_recovery.models import ControlCommands, ControllerOutput
from aircraft_recovery.simulator import SimpleAircraftSimulator
from aircraft_recovery.data_quality.final_generation import _metric_sample, _scenario
from aircraft_recovery.evaluation.v2_1b_nonsealed import _flight_spec, V21AControlPolicyController
from aircraft_recovery.verification.plant_validation import validate_formal_plant

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(encoding='utf-8') as f:
        for line in f: yield json.loads(line)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def close(a,b):
    if a is None or b is None: assert a is b,(a,b)
    else: assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)
def recovery(samples, reference):
    valid=[s['stall_margin_kts']>=10 and abs(s['bank_deg'])<=10 and -7.5<=s['pitch_deg']<=7.5 and abs(s['vertical_speed_fpm'])<=750 and s['terrain_clearance_ft']>=0 for s in samples]
    for i in range(len(samples)-49):
        if samples[i]['timestamp_seconds']>=reference and all(valid[i:i+50]):
            return True,max(0,samples[i]['timestamp_seconds']-reference),not all(valid[i+50:])
    return False,None,False
def safety(samples):
    counts=collections.Counter()
    for s in samples:
        flags=dict(stall=s['stall_margin_kts']<0,overspeed=s['overspeed_margin_kts']<0,bank=abs(s['bank_deg'])>60,pitch=not -20<=s['pitch_deg']<=25,angle_of_attack=s['angle_of_attack_deg']>14,load_factor=not -1<=s['load_factor_g']<=3.5,terrain_clearance=s['terrain_clearance_ft']<0)
        counts.update({k:int(v) for k,v in flags.items()});counts['any']+=int(any(flags.values()))
    return counts
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('.'));ap.add_argument('--out',type=Path,default=Path('results/publication_audit'));args=ap.parse_args()
    root=args.root.resolve();out=args.out;out.mkdir(parents=True,exist_ok=True)
    candidate=root/'datasets/candidate_policy_data/final_policy_candidate_v2_1';approved=root/'datasets/approved_policy_data/final_policy_approved_v2_1'
    flights=list(rows(candidate/'raw/flights.jsonl'));byid={f['flight_id']:f for f in flights};assert len(byid)==1152
    assert len({f['scenario_instance_id'] for f in flights})==1152
    assert len({f['random_seed'] for f in flights})==1152
    splitcounts=dict(collections.Counter(f['split'] for f in flights));assert sorted(splitcounts.values())==[144,144,144,144,576]
    report={'scope':'Frozen-record recalculation and action replay; no refitting; original latency is reaggregated, not replaced by CPU timings.','dataset_flights':splitcounts,'splits':{},'checkpoints':[]}
    aircraft=load_config(root/'configs/aircraft/simple_twin.yaml',AircraftConfig)
    checkpoint=root/'checkpoints/final_policy_v1/moderate/final_training_v2_1a/seed_17001/best.pt'
    neural=V21AControlPolicyController(checkpoint,torch.device('cpu'));torch.set_num_threads(1)
    fresh=[]
    for split,prefix in [('challenge','final_nonsealed_evaluation_v2_1b'),('ood_stress','final_nonsealed_evaluation_v2_1b'),('final_sealed_test','final_sealed_evaluation_v2_1b')]:
        folder=root/'results'/prefix/split;records=list(rows(folder/'flight_records.jsonl'));summary=read(folder/'summary.json')
        assert len(records)==288 and len({r['flight_id'] for r in records})==144
        lookup={(r['flight_id'],r['controller_condition']):r for r in records};assert len(lookup)==288
        lat=collections.defaultdict(list);mlat=collections.defaultdict(list);rules=collections.Counter();n=0;seen=set()
        groups=itertools.groupby(rows(folder/'action_traces.jsonl'),key=lambda r:(r['flight_id'],r['controller_condition']))
        for key,group in groups:
            trace=list(group);expected=lookup[key];assert len(trace)==expected['step_count']==260
            flight=byid[key[0]];scenario=_scenario(root,_flight_spec(flight));sim=SimpleAircraftSimulator(aircraft,scenario);obs=sim.reset(flight['random_seed']);samples=[];mods=0;freshkey=(split,flight['difficulty_tier'])
            check_neural=key[1]=='unmonitored' and freshkey not in seen
            maxerr=0.0
            for i,tr in enumerate(trace):
                assert tr['timestep_index']==i;close(obs.timestamp_seconds,tr['timestamp_seconds'])
                s=_metric_sample(sim.audit_snapshot()['true_state'],obs.environment.terrain_clearance_ft,aircraft);s['timestamp_seconds']=obs.timestamp_seconds;samples.append(s)
                if check_neural:
                    predicted=neural.act(obs).control_commands.model_dump()
                    maxerr=max(maxerr,max(abs(predicted[k]-v) for k,v in tr['proposed_action'].items()))
                action=ControllerOutput.model_construct(control_commands=ControlCommands(**tr['monitor_selected_action']))
                obs=sim.step(action)
                for k,v in tr['executed_action'].items():close(sim.audit_snapshot()['applied_control_commands'][k],v)
                mods+=int(any(abs(v-tr['proposed_action'][k])>1e-12 for k,v in tr['monitor_selected_action'].items()))
                lat[key[1]].append(tr['full_controller_latency_ms'])
                if tr['monitor_decision']:
                    mlat[key[1]].append(tr['monitor_decision']['monitor_duration_ms']);rules.update(tr['monitor_decision']['violated_properties'])
            result=recovery(samples,expected['recovery_timer_reference_seconds']);assert result[0]==expected['strict_recovery_success'];close(result[1],expected['strict_recovery_time_seconds']);assert result[2]==expected['lost_after_recovery']
            sc=safety(samples);assert sc['any']==expected['safety']['any_violation_step_count']
            for k,v in expected['safety']['violations_by_property'].items():assert sc[k]==v
            assert mods==expected['monitor_modification_count'];n+=len(trace)
            if check_neural:
                assert maxerr<2e-6,maxerr;fresh.append({'split':split,'tier':flight['difficulty_tier'],'steps':260,'maximum_absolute_action_error_cpu_vs_frozen':maxerr});seen.add(freshkey)
        result={'conditions':{},'action_steps_replayed':n,'rule_events':dict(rules)}
        for c in ['unmonitored','monitored']:
            rr=[r for r in records if r['controller_condition']==c];success=sum(r['strict_recovery_success'] for r in rr);viol=sum(r['safety']['flight_has_violation'] for r in rr)
            onset=[r['strict_recovery_time_seconds'] for r in rr];p95=float(np.quantile(lat[c],.95));close(p95,summary['overall'][c]['latency']['full_controller']['p95_ms'])
            assert success==summary['overall'][c]['strict_recovery']['successes'];assert viol==summary['overall'][c]['safety_violation_flights']['successes']
            tiers={str(t):{'median':float(np.median(v:=[r['strict_recovery_time_seconds'] for r in rr if r['difficulty_tier']==t])),'min':min(v),'max':max(v),'n':len(v)} for t in range(6)}
            result['conditions'][c]={'n':len(rr),'recovered':success,'safety_violation_flights':viol,'median_onset_s':float(np.median(onset)),'mean_onset_s':float(np.mean(onset)),'modifications':sum(r['monitor_modification_count'] for r in rr),'steps':len(lat[c]),'full_controller_p95_ms':p95,'monitor_p95_ms':float(np.quantile(mlat[c],.95)) if mlat[c] else None,'tiers':tiers}
        ids=sorted({r['flight_id'] for r in records});d=np.array([int(lookup[(i,'monitored')]['strict_recovery_success'])-int(lookup[(i,'unmonitored')]['strict_recovery_success']) for i in ids]);rng=np.random.default_rng(19001);boot=d[rng.integers(0,144,(10000,144))].mean(1)
        result['paired_recovery_difference']=float(d.mean());result['paired_bootstrap_95_ci']=np.quantile(boot,[.025,.975]).tolist();report['splits'][split]=result
        print('Replayed',split,n,'steps',flush=True)
        (out/'audit.json').write_text(json.dumps(report,indent=2))
    report['fresh_cpu_inference']=fresh
    for p in sorted((root/'checkpoints/final_policy_v1').rglob('best.pt')):
        c=torch.load(p,map_location='cpu',weights_only=False);state=c['model_state_dict'];report['checkpoints'].append({'path':p.relative_to(root).as_posix(),'sha256':sha(p),'parameters':sum(t.numel() for t in state.values())})
    for p in sorted((root/'results/supplementary_architecture_comparison_20261011/training').glob('*/best.pt')):
        c=torch.load(p,map_location='cpu',weights_only=False);report['checkpoints'].append({'path':p.relative_to(root).as_posix(),'sha256':sha(p),'parameters':sum(t.numel() for t in c['model_state_dict'].values())})
    formal=root/'results/formal_plant_smoke_v1';r=validate_formal_plant(formal/'formal_plant_model.json',formal/'calibration',formal/'validation',out/'formal_recalculated.json')
    old=read(formal/'validation_report.json');assert r['one_step']==old['one_step'];assert r['rollout']==old['rollout']
    report['formal_validation']={'one_step_records':r['one_step']['evaluated'],'one_step_failures':r['one_step']['containment_failures'],'report':'formal_recalculated.json'}
    report['dataset_integrity']={}
    for name,folder in [('candidate',candidate),('approved',approved)]:
        counts=collections.Counter();h=hashlib.sha256()
        with (folder/'raw/transitions.jsonl').open('rb') as f:
            for line in f:h.update(line);row=json.loads(line);counts[row['split']]+=1
        expected=sum(1 for _ in rows(folder/'raw/flights.jsonl'))*260;assert sum(counts.values())==expected
        report['dataset_integrity'][name]={'transition_counts':dict(counts),'transitions_sha256':h.hexdigest(),'flights_sha256':sha(folder/'raw/flights.jsonl')}
    report['status']='PASS';(out/'audit.json').write_text(json.dumps(report,indent=2));print('PASS',flush=True)
if __name__=='__main__': main()
