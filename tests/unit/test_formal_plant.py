import numpy as np
import pytest

from aircraft_recovery.verification.formal_plant import (
    FormalPlantModel,StateInterval,resolve_delayed_action,transition,within_validity,
)
from aircraft_recovery.verification.plant_calibration import ACTION_NAMES,STATE_NAMES


def _model(residual=0.0):
    return FormalPlantModel(
        time_step_seconds=0.1,
        coefficients={
            "airspeed_kts":[-0.12,0.998,0.4,0.4],
            "pitch_deg":[0,1,1.2,1],
            "bank_deg":[0,1,3,0.36,1],
            "vertical_speed_fpm":[101.269],
            "terrain_clearance_ft":[0,1,1/600],
        },
        residual_bounds={name:(-residual,residual) for name in ("airspeed_kts","pitch_deg","bank_deg","vertical_speed_fpm","terrain_clearance_ft")},
        numerical_tolerance=1e-8,
        validity={
            "state":{"airspeed_kts":(100,180),"pitch_deg":(-20,20),"bank_deg":(-50,50),"vertical_speed_fpm":(-20000,20000),"terrain_clearance_ft":(100,12000)},
            "failures":{"left_thrust_availability":(0,0),"right_thrust_availability":(.9,1),"aileron_effectiveness":(.3,1),"actuator_delay_seconds":(0,.3)},
            "disturbances":{"pitch_delta_deg":(-1,1),"roll_delta_deg":(-2,2),"crosswind_kts":(-35,35)},
            "sensor_errors":{},
        },fit_metadata={},
    )


def test_transition_has_stable_dimensions_finite_ordered_bounds_and_residual_inflation():
    state={name:0.0 for name in STATE_NAMES}; state.update(airspeed_kts=140,pitch_deg=0,bank_deg=0,terrain_clearance_ft=2000)
    interval=StateInterval.point(state)
    actions={name:(value,value) for name,value in zip(ACTION_NAMES,(.1,.2,0,0,1))}
    failures={"left_thrust_availability":(0,0),"right_thrust_availability":(1,1),"aileron_effectiveness":(.5,.5),"actuator_delay_seconds":(.1,.1)}
    disturbances={"pitch_delta_deg":(0,0),"roll_delta_deg":(0,0),"crosswind_kts":(10,10)}
    nominal=transition(_model(),interval,actions,failures,disturbances)
    inflated=transition(_model(.25),interval,actions,failures,disturbances)
    assert nominal.lower.shape==nominal.upper.shape==(10,)
    assert np.all(np.isfinite(nominal.lower)) and np.all(nominal.lower<=nominal.upper)
    assert np.all(inflated.lower[:5]<nominal.lower[:5])
    assert np.all(inflated.upper[:5]>nominal.upper[:5])
    assert tuple(nominal.as_dict())==STATE_NAMES


def test_validity_rejects_extrapolation():
    state={name:0.0 for name in STATE_NAMES}; state.update(airspeed_kts=181,pitch_deg=0,bank_deg=0,terrain_clearance_ft=2000)
    interval=StateInterval.point(state); failures={"left_thrust_availability":(0,0),"right_thrust_availability":(1,1),"aileron_effectiveness":(.5,.5),"actuator_delay_seconds":(0,0)}
    disturbances={"pitch_delta_deg":(0,0),"roll_delta_deg":(0,0),"crosswind_kts":(0,0)}
    assert not within_validity(_model(),interval,failures,disturbances)
    with pytest.raises(ValueError,match="outside"):
        transition(_model(),interval,{name:(0,0) for name in ACTION_NAMES},failures,disturbances)


def test_actuator_delay_uses_recorded_queue_without_changing_action_schema():
    old={name:float(index)/10 for index,name in enumerate(ACTION_NAMES)}
    recent={name:float(index+1)/10 for index,name in enumerate(ACTION_NAMES)}
    proposed={name:1.0 for name in ACTION_NAMES}
    assert resolve_delayed_action([old,recent],proposed,0)==proposed
    assert resolve_delayed_action([old,recent],proposed,1)==recent
    assert resolve_delayed_action([old,recent],proposed,2)==old
    assert resolve_delayed_action([],proposed,1)=={name:0.0 for name in ACTION_NAMES}
