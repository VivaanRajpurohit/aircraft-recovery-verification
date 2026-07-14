from aircraft_recovery.controllers.pid import PIDController


def test_pid_anti_windup_stops_integral_growth_during_saturation() -> None:
    pid = PIDController(kp=2.0, ki=1.0, kd=0.0, integral_limit=5.0)
    for _ in range(100):
        assert pid.update(10.0, 0.1) == 1.0
    assert pid.integral == 0.0


def test_pid_integral_is_explicitly_bounded() -> None:
    pid = PIDController(kp=0.0, ki=0.01, kd=0.0, integral_limit=0.5)
    for _ in range(100):
        pid.update(0.1, 0.1)
    assert pid.integral <= 0.5

