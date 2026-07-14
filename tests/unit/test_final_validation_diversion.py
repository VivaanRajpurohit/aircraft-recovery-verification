import json
from aircraft_recovery.analysis.diversion_validation import heading_error_deg
from aircraft_recovery.config import AircraftConfig,ScenarioConfig,load_config
from aircraft_recovery.controllers import DeterministicFallbackController,FallbackConfig
from aircraft_recovery.controllers.emergency import EmergencyMode
from aircraft_recovery.simulator import SimpleAircraftSimulator

def test_heading_error_wraps_across_zero():
    assert heading_error_deg(359.0,1.0)==2.0
    assert heading_error_deg(1.0,359.0)==2.0

def test_nominal_requested_diversion_enters_emergency_state_machine():
    scenario=load_config("configs/scenarios/validation_diversion_nominal.yaml",ScenarioConfig)
    aircraft=load_config("configs/aircraft/simple_twin.yaml",AircraftConfig)
    fallback=load_config("configs/controllers/fallback.yaml",FallbackConfig)
    controller=DeterministicFallbackController(fallback,scenario.airports)
    observation=SimpleAircraftSimulator(aircraft,scenario).reset(scenario.seed)
    controller.act(observation)
    assert observation.mission.emergency_declared
    assert controller.state_machine.mode==EmergencyMode.FAILURE_DETECTED

def test_diversion_heading_error_produces_bank_guidance():
    scenario=load_config("configs/scenarios/validation_diversion_nominal.yaml",ScenarioConfig)
    scenario.initial_state.heading_deg=45.0
    aircraft=load_config("configs/aircraft/simple_twin.yaml",AircraftConfig)
    fallback=load_config("configs/controllers/fallback.yaml",FallbackConfig)
    controller=DeterministicFallbackController(fallback,scenario.airports)
    simulator=SimpleAircraftSimulator(aircraft,scenario); observation=simulator.reset(scenario.seed)
    action=None
    for _ in range(50):
        action=controller.act(observation); observation=simulator.step(action)
    assert action is not None
    assert controller.state_machine.mode in {EmergencyMode.DIVERT,EmergencyMode.APPROACH}
    assert action.stabilization_targets.target_roll_deg>0
    assert action.control_commands.aileron>0
