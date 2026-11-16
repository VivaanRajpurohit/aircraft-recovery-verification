"""Pure visualization mapping from executed controls to named aircraft nodes."""

from __future__ import annotations

from typing import Any


CONTROL_NAMES = ("elevator", "aileron", "rudder", "throttle_left", "throttle_right")


def animation_angles(executed: dict[str, float], settings: Any) -> dict[str, float]:
    """Return bounded visual deflections; this never feeds back into physics."""
    clamp = lambda value: max(-1.0, min(1.0, float(value)))
    aileron = clamp(executed.get("aileron", 0.0)) * settings.aileron_limit_deg
    elevator = clamp(executed.get("elevator", 0.0)) * settings.elevator_limit_deg
    rudder = clamp(executed.get("rudder", 0.0)) * settings.rudder_limit_deg
    flap = max(0.0, min(1.0, float(executed.get("flap", 0.0)))) * settings.flap_limit_deg
    return {
        "left_aileron": aileron * settings.left_aileron_sign,
        "right_aileron": aileron * settings.right_aileron_sign,
        "elevator": elevator * settings.elevator_sign,
        "rudder": rudder * settings.rudder_sign,
        "left_flap": flap * settings.flap_sign,
        "right_flap": flap * settings.flap_sign,
    }


class AircraftNodeAnimator:
    """Own runtime node transforms for an imported, animation-ready aircraft."""

    def __init__(self, model: Any, settings: Any) -> None:
        self.settings = settings
        self.nodes: dict[str, Any] = {}
        self.base_hpr: dict[str, tuple[float, float, float]] = {}
        self.propeller_degrees = {"left_propeller": 0.0, "right_propeller": 0.0}
        if not settings.enabled:
            self.missing_nodes: list[str] = []
            return
        for role, name in settings.node_names().items():
            node = model.find(f"**/{name}")
            if not node.isEmpty():
                self.nodes[role] = node
                hpr = node.getHpr()
                self.base_hpr[role] = (float(hpr.x), float(hpr.y), float(hpr.z))
        self.missing_nodes = sorted(set(settings.node_names()) - set(self.nodes))

    @property
    def available(self) -> bool:
        return bool(self.settings.enabled and not self.missing_nodes)

    def reset(self) -> None:
        for role, node in self.nodes.items():
            node.setHpr(*self.base_hpr[role])
        self.propeller_degrees = {"left_propeller": 0.0, "right_propeller": 0.0}

    def apply(self, executed: dict[str, float], delta_seconds: float) -> dict[str, float]:
        values = {name: float(executed.get(name, 0.0)) for name in (*CONTROL_NAMES, "flap")}
        angles = animation_angles(values, self.settings)
        for role in ("left_aileron", "right_aileron", "elevator", "left_flap", "right_flap"):
            node = self.nodes.get(role)
            if node is not None:
                h, p, r = self.base_hpr[role]
                node.setHpr(h, p + angles[role], r)
        rudder = self.nodes.get("rudder")
        if rudder is not None:
            h, p, r = self.base_hpr["rudder"]
            rudder.setHpr(h, p, r + angles["rudder"])
        for role, throttle_name, direction in (
            ("left_propeller", "throttle_left", self.settings.left_propeller_sign),
            ("right_propeller", "throttle_right", self.settings.right_propeller_sign),
        ):
            throttle = max(0.0, min(1.0, values[throttle_name]))
            rate = self.settings.propeller_idle_deg_per_second + throttle * (
                self.settings.propeller_max_deg_per_second - self.settings.propeller_idle_deg_per_second
            )
            self.propeller_degrees[role] = (
                self.propeller_degrees[role] + direction * rate * max(0.0, delta_seconds)
            ) % 360.0
            node = self.nodes.get(role)
            if node is not None:
                h, p, r = self.base_hpr[role]
                node.setHpr(h + self.propeller_degrees[role], p, r)
        return {**angles, **self.propeller_degrees}
