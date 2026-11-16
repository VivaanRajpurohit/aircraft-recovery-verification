# Experimental Unit Definition

The independent evaluation unit is one complete simulated flight/trajectory identified by its flight ID, scenario-instance ID, and simulator seed. Recovery, safety, timing, and controller comparisons aggregate once per flight. Repeated timesteps from a flight are serially dependent and must never be treated as independent replicates or independently bootstrapped.

The supervised-learning sample is one transition: observation, expert proposed action, executed action/audit fields, next observation, and provenance. Transitions may be optimized in minibatches, but splitting is performed at flight level and all transitions from a flight stay together. Controller comparisons reuse the same flight specification and are analyzed as paired flights.
