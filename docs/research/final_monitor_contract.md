# Final Monitor Contract

Frozen configuration: `configs/research/final_monitor_v1.yaml`. The monitored policy proposes five bounded actions: elevator/aileron/rudder in [−1,1] and each throttle in [0,1]. The existing one-step bounded abstraction checks the unchanged simulator-domain envelope and a maximum command-rate assumption of 2.5 control-units/s. It rejects nonfinite/invalid navigation inputs, uses a 20 ms solver timeout, and tries only the frozen projection blends 0.25, 0.5, 0.75, and 1.0.

At each monitored step, a safe proposal is accepted; otherwise an admissible projected action is selected; if none is established, input is rejected, or solver status requires rejection, fallback is selected. Evaluation repeats next step—there is no added monitor latch or outcome-based threshold adaptation. Proposal, decision/status, selected action, reason, solver result, modification flag, fallback flag, and latency must be serialized. Development data cannot alter these values after this freeze; challenge, OOD-style stress, and sealed results can never tune them.

This is a one-step low-order abstraction with bounded assumptions, not verification of the neural network, nonlinear plant, hardware, or real aircraft.
