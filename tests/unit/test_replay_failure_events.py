import json
from pathlib import Path
import pytest
from aircraft_recovery.visualization.replay_events import ReplayEventState,normalize_replay_data

ACTUAL=Path("results/phase5_quick/runs/phase2_engine_aileron_5103_monitored")

def test_actual_run_failures_activation_knowledge_health_and_monitor_events():
    if not ACTUAL.exists(): pytest.skip("quick replay artifact unavailable")
    data=normalize_replay_data(ACTUAL)
    by_id={failure.failure_id:failure for failure in data.failures}
    assert set(by_id)=={"left_engine_1","aileron_loss_1"}
    assert by_id["left_engine_1"].activation_time_s==pytest.approx(2.0)
    assert by_id["left_engine_1"].severity==1.0
    assert by_id["aileron_loss_1"].severity==.55
    assert by_id["aileron_loss_1"].details["remaining_effectiveness_fraction"]==pytest.approx(.45)
    assert len([event for event in data.failure_events if event.kind=="activated"])==1
    assert set(data.state_at(2.0).ground_truth_ids)=={"left_engine_1","aileron_loss_1"}
    assert set(data.state_at(2.0).controller_known_ids)=={"left_engine_1","aileron_loss_1"}
    assert data.state_at(5.0).left_engine==0 and data.state_at(5.0).aileron==pytest.approx(.45)
    statuses=[event.status for event in data.monitor_events]
    assert statuses.count("modify")==54 and statuses.count("reject_fallback")==14

def test_backward_seek_rearms_simultaneous_activation_and_restart_resets():
    data=normalize_replay_data(ACTUAL); state=ReplayEventState(data.failure_events)
    first=state.advance(2.0); assert len(first)==1 and len(first[0].failure_ids)==2
    assert state.advance(2.1)==[]
    state.advance(1.9)
    replayed=state.advance(2.0); assert len(replayed)==1 and len(replayed[0].failure_ids)==2
    state.reset(); assert len(state.advance(2.0))==1

def test_temporary_hidden_failure_clears_and_optional_fields_are_safe(tmp_path):
    run=tmp_path/"synthetic"; run.mkdir()
    metadata={"scenario_configuration":{"failures":[{"failure_id":"hidden_bias","failure_type":"airspeed_bias","affected_subsystem":"airspeed_sensor","activation_time_seconds":1.0,"duration_seconds":1.0,"observable":False,"severity":.2}]}}
    (run/"metadata.json").write_text(json.dumps(metadata))
    rows=[]
    for time,active in ((.5,False),(1.0,True),(1.5,True),(2.0,False)):
        truth=[{"failure_id":"hidden_bias","failure_type":"airspeed_bias","affected_subsystem":"airspeed_sensor","activation_time_seconds":1.0}] if active else []
        rows.append({"next_observation":{"timestamp_seconds":time,"active_failures":[],"control_health":{},"engine_health":{}},"ground_truth_after":{"ground_truth_failures":truth,"controller_known_failures":[]}})
    (run/"history.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    data=normalize_replay_data(run)
    assert data.failures[0].severity==.2
    assert data.failures[0].controller_known is False
    assert data.active_failures(1.5)[0].active
    assert data.active_failures(1.5)[0].controller_known is False
    assert data.active_failures(2.0)==[]
    cleared=[event for event in data.failure_events if event.kind=="cleared"]
    assert len(cleared)==1 and cleared[0].time_s==2.0
