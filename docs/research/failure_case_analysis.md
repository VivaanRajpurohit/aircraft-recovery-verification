# Failure-Case Analysis Protocol

Unsuccessful trajectories are classified reproducibly using saved metrics before optional human review:

| Rule ID | Objective condition |
|---|---|
| FC-ROLL-01 | Absolute bank fails to enter the recovery bound before termination |
| FC-OSC-01 | Control sign-change rate or total variation exceeds the preregistered development threshold |
| FC-AIRSPEED-01 | Airspeed crosses the configured minimum safety bound |
| FC-PITCH-01 | Absolute pitch remains outside recovery bound after the allowed settling interval |
| FC-THRUST-01 | Asymmetric-thrust compensation is followed by sustained yaw/bank violation |
| FC-DELAY-01 | First executed corrective command occurs after the configured timing bound |
| FC-MONITOR-01 | Monitor modification/rejection precedes the terminal unsafe outcome |
| FC-FALLBACK-01 | Fallback controls more than 50% of post-failure steps |
| FC-DOMAIN-01 | Simulator state exits the declared experimental validity region |
| FC-OTHER-01 | Termination reason is neither recovery nor a classified envelope failure |

Multiple labels may apply. Automatic rules, thresholds, source trajectory hash, controller/checkpoint hash, and reviewer overrides are stored. Difficult cases are never deleted. Human changes append a new review event rather than mutating the original classification.

