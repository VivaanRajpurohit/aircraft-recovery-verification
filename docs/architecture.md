# Phase 1 Architecture

The validated `ControllerInput` and `ControllerOutput` models form the stable
JSON boundary. A simulator implements `AircraftSimulator`; a controller
implements `Controller`; and `ScenarioRunner` advances the clock, retains the
paired seed, and logs each validated exchange. This separates dynamics,
control, experiment orchestration, and artifacts.

The Phase 1 `StabilizationController` is only an executable interface fixture.
It is neither the Phase 2 independent fallback nor the Phase 3 learned baseline.
It performs ordinary command clipping but no safety verification.

## JSBSim adapter path

A future `JSBSimAircraftSimulator` will implement the existing simulator
protocol. At reset, it will load an aircraft model, initial conditions, weather,
and a deterministic seed. At each step it will translate normalized commands to
JSBSim flight-control properties, advance by the configured time step, convert
JSBSim SI/imperial properties explicitly, and create the same validated input
packet. Controller, logger, experiment, and schema code will therefore remain
unchanged. Adapter contract tests will run both simulators against identical
interface invariants.

Phase 1 evaluates one controller command per dynamics step and validates that
`controller_frequency_hz == 1 / time_step_seconds`. A later multi-rate adapter
may hold commands across substeps while preserving the same packet boundary.

## GPU memory planning

FP32 weights use four bytes per parameter. Raw inference weights therefore use
about 0.4 MB (100k), 4 MB (1M), 20 MB (5M), and 40 MB (10M). Adam training
commonly retains weights, gradients, and two FP32 moment tensors, with practical
parameter-state planning near 16 bytes per parameter: 1.6 MB, 16 MB, 80 MB, and
160 MB respectively. Activations, batches, CUDA context, and framework
workspaces add overhead. All proposed policy sizes are comfortable on a 12 GB
RTX 3060; batch and sequence dimensions still require measurement.
