"""Open a timed Panda3D visual cycle for the generated aircraft node contract."""

from __future__ import annotations

import argparse
from pathlib import Path


STATES = (
    ("NEUTRAL", {}),
    ("FULL LEFT ROLL", {"aileron": -1.0}),
    ("FULL RIGHT ROLL", {"aileron": 1.0}),
    ("FULL PITCH UP", {"elevator": 1.0}),
    ("FULL PITCH DOWN", {"elevator": -1.0}),
    ("FULL YAW LEFT", {"rudder": -1.0}),
    ("FULL YAW RIGHT", {"rudder": 1.0}),
    ("LEFT ENGINE LOW / RIGHT ENGINE HIGH", {"throttle_left": .05, "throttle_right": 1.0}),
    ("RIGHT ENGINE LOW / LEFT ENGINE HIGH", {"throttle_left": 1.0, "throttle_right": .05}),
    ("RETURN TO NEUTRAL", {}),
)


def controls(overrides: dict[str, float]) -> dict[str, float]:
    return {
        "elevator": 0.0, "aileron": 0.0, "rudder": 0.0,
        "throttle_left": .45, "throttle_right": .45, **overrides,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/visualization/generated_plane.yaml")
    parser.add_argument("--state-seconds", type=float, default=1.4)
    parser.add_argument("--loop", action="store_true")
    args = parser.parse_args()
    if args.state_seconds <= 0:
        parser.error("--state-seconds must be positive")

    from panda3d.core import AmbientLight, DirectionalLight, Filename, loadPrcFileData
    loadPrcFileData("", "window-title Generated Aircraft Animation Contract Test\nwin-size 1200 800\nsync-video 1")
    from direct.gui.OnscreenText import OnscreenText
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from panda3d.core import TextNode
    from aircraft_recovery.visualization.aircraft_animation import AircraftNodeAnimator
    from aircraft_recovery.visualization.replay_3d import load_replay_3d_settings

    settings = load_replay_3d_settings(args.config)
    base = ShowBase()
    base.disableMouse()
    model_path = Path(settings.aircraft.model_path).resolve()
    model = base.loader.loadModel(Filename.fromOsSpecific(str(model_path)), noCache=True)
    if model.isEmpty():
        raise RuntimeError(f"Could not load {model_path}")
    model.reparentTo(base.render)
    model.setScale(settings.aircraft.scale)
    model.setHpr(settings.aircraft.yaw_offset_deg, settings.aircraft.pitch_offset_deg, settings.aircraft.roll_offset_deg)
    animator = AircraftNodeAnimator(model, settings.aircraft.animation)
    if not animator.available:
        raise RuntimeError(f"Generated model animation nodes missing: {animator.missing_nodes}")
    base.camera.setPos(18, -23, 10)
    base.camera.lookAt(0, 0, 0.5)
    base.setBackgroundColor(.18, .25, .34, 1)
    ambient = AmbientLight("ambient"); ambient.setColor((.5, .5, .55, 1))
    base.render.setLight(base.render.attachNewNode(ambient))
    sun = DirectionalLight("sun"); sun.setColor((1, .96, .88, 1))
    sun_node = base.render.attachNewNode(sun); sun_node.setHpr(-35, -45, 0); base.render.setLight(sun_node)
    label = OnscreenText(
        parent=base.a2dTopLeft, text="", pos=(.05, -.08), scale=.045,
        align=TextNode.ALeft, mayChange=True, fg=(1, .95, .55, 1), shadow=(0, 0, 0, 1),
    )
    elapsed = 0.0
    index = 0

    def update(task):
        nonlocal elapsed, index
        dt = max(0.0, globalClock.getDt())
        elapsed += dt
        if elapsed >= args.state_seconds:
            elapsed = 0.0
            index += 1
            if index >= len(STATES):
                if args.loop:
                    index = 0
                else:
                    base.userExit()
                    return Task.done
        name, values = STATES[index]
        applied = controls(values)
        angles = animator.apply(applied, dt)
        label.setText(
            "AUTOMATED GENERATED AIRCRAFT TEST\n"
            f"State {index + 1}/{len(STATES)}: {name}\n"
            f"Executed: {applied}\n"
            f"Visual angles: {angles}\n"
            "Esc: exit"
        )
        return Task.cont

    base.accept("escape", base.userExit)
    base.taskMgr.add(update, "generated-aircraft-animation-cycle")
    base.run()
    base.destroy()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
