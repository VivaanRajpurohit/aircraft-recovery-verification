"""Deterministic PID primitive with conditional-integration anti-windup."""

from __future__ import annotations


class PIDController:
    """Bounded PID with integral clamping and saturation-aware integration."""

    def __init__(
        self,
        kp: float,
        ki: float,
        kd: float,
        output_min: float = -1.0,
        output_max: float = 1.0,
        integral_limit: float = 10.0,
    ) -> None:
        self.kp, self.ki, self.kd = kp, ki, kd
        self.output_min, self.output_max = output_min, output_max
        self.integral_limit = abs(integral_limit)
        self.integral = 0.0
        self._previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self._previous_error = None

    def update(self, error: float, dt: float) -> float:
        derivative = 0.0 if self._previous_error is None else (error - self._previous_error) / dt
        candidate_integral = max(
            -self.integral_limit,
            min(self.integral_limit, self.integral + error * dt),
        )
        candidate = self.kp * error + self.ki * candidate_integral + self.kd * derivative
        saturated_high = candidate > self.output_max and error > 0.0
        saturated_low = candidate < self.output_min and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = candidate_integral
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self._previous_error = error
        return max(self.output_min, min(self.output_max, output))

