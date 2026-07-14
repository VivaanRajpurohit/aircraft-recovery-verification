# Research software scope

This repository studies a neural controller operating in closed loop with an
internal numerical aircraft simulator. Neural proposals pass through a runtime
assurance layer before the selected action advances the numerical dynamics.

The software is simulation-only. It must not connect to real aircraft, drones,
avionics, actuators, external flight simulators, network-connected vehicles, or
physical hardware. Formal results apply only to named mathematical models,
bounds, assumptions, properties, solver settings, and finite horizons.

The Panda3D viewer is a read-only visualization of completed experiment logs.
It is not part of the controller loop and creates no experimental evidence.
