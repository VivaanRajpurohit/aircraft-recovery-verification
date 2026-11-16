from pathlib import Path
import importlib.util
import gzip
import json
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('supplement',ROOT/'scripts/run_supplementary_architecture_comparison_20261011.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def test_atomic_publish_preserves_completed_bytes_and_ignores_orphans(tmp_path):
    path=tmp_path/'complete.json'
    module.publish(path,{'status':'COMPLETE'})
    original=path.read_bytes()
    module.publish(path,{'status':'COMPLETE'})
    with pytest.raises(FileExistsError):module.publish(path,{'status':'PARTIAL'})
    assert path.read_bytes()==original
    assert module.load_epoch(tmp_path) is None

def test_condition_validation_rejects_missing_trace_and_wrong_checkpoint(tmp_path):
    value={'checkpoint_sha256':'a','record':{'step_count':260,'invalid_experiment':False},'traces':[{}]*260}
    path=tmp_path/'condition.json.gz';path.write_bytes(gzip.compress(json.dumps(value).encode()))
    assert module.read_condition(path,{'checkpoint_sha256':'a'})==value
    with pytest.raises(ValueError):module.read_condition(path,{'checkpoint_sha256':'b'})
    value['traces'].pop();path.write_bytes(gzip.compress(json.dumps(value).encode()))
    with pytest.raises(ValueError):module.read_condition(path,{'checkpoint_sha256':'a'})

def test_access_boundary_remains_closed():
    module.train.assert_access_policy()
    for role in ('training','model_selection','challenge_evaluation'):
        with pytest.raises(PermissionError):module.ev.assert_role_split(role,'final_sealed_test')
    with pytest.raises(PermissionError):module.ev.assert_role_split('training','challenge')
    with pytest.raises(PermissionError):module.ev.assert_role_split('model_selection','ood_stress')

def test_frozen_scenario_ids_have_portable_collision_resistant_paths(tmp_path):
    ident='final_policy_candidate_v2_1:challenge:tier0:nominal:22000'
    path=module.pair_directory(tmp_path,ident)
    assert path.parent==tmp_path and len(path.name)==64
    assert ':' not in path.name
    assert path==module.pair_directory(tmp_path,ident)
    assert path!=module.pair_directory(tmp_path,ident+'1')

def test_epoch_resume_reproduces_uninterrupted_model(tmp_path,monkeypatch):
    candidate=module.train.ARCHITECTURE_CANDIDATES[0]
    torch.manual_seed(123)
    x=torch.randn(12,42);y=torch.cat((torch.zeros(12,3),torch.ones(12,2)*0.5),dim=1)
    arrays=(x,y,x,y)
    # A fixed diagnostic drives early stopping; compare actual trained weights.
    monkeypatch.setattr(module.train,'_development_metrics',lambda *args:{'mse':1.0})
    monkeypatch.setattr(module.train,'_environment',lambda *args:{})
    monkeypatch.setattr(module.train,'_git_revision',lambda *args:'test-revision')
    monkeypatch.setattr(module,'ROOT',tmp_path)
    plan={'plan_logical_sha256':'test'}
    norm={}
    monkeypatch.setattr(module,'OUT',tmp_path/'straight')
    module.fit(candidate,arrays,norm,plan)
    straight=torch.load(module.OUT/'training/small_reference/last.pt',weights_only=False)
    monkeypatch.setattr(module,'OUT',tmp_path/'resumed')
    actual_publish=module.publish
    interrupted=[False]
    def fail_before_marker(path,value):
        if path.name=='epoch_002.json' and not interrupted[0]:
            interrupted[0]=True;raise RuntimeError('simulated interruption after temporary epoch checkpoint')
        return actual_publish(path,value)
    monkeypatch.setattr(module,'publish',fail_before_marker)
    with pytest.raises(RuntimeError):module.fit(candidate,arrays,norm,plan)
    assert module.load_epoch(module.OUT/'training/small_reference')['epoch']==1
    monkeypatch.setattr(module,'publish',actual_publish)
    module.fit(candidate,arrays,norm,plan)
    resumed=torch.load(module.OUT/'training/small_reference/last.pt',weights_only=False)
    assert straight['epoch']==resumed['epoch']==9
    for key,value in straight['model_state_dict'].items():assert torch.equal(value,resumed['model_state_dict'][key])
    assert straight['history']==resumed['history']
