# Safety properties

For every reachable set at every step, the abstraction checks:

- elevator, aileron, and rudder remain in `[-1, 1]`;
- both throttle commands remain in `[0, 1]`;
- each command changes by no more than 0.25 per step;
- airspeed remains in `[100, 205]` kt;
- absolute bank remains at or below 60 degrees;
- pitch remains in `[-20, 25]` degrees;
- terrain clearance remains strictly positive.

Angle of attack and load factor retain the existing limits of 14 degrees and `[-1, 3.5]` g, respectively, but are not proof obligations until a sound relation to the formal state has been derived and validated. Reports must mark them `not_represented`, not silently pass them.

Runtime-assurance invariants require invalid or non-finite observations, solver timeout, and solver unknown to select fallback; an action rejected by the monitor cannot be executed unchanged; a projected action must satisfy command and rate bounds; decision branches are mutually exclusive; and terminated states cannot return to an operational mode.

A cell is safe only when the complete reachable set proves every enabled invariant. Sampling cannot establish this classification.
