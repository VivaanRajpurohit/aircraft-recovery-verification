# Pilot Data-Quality Report

- Dataset: `final_policy_pilot_v2`
- Flights/transitions: 36 / 9350
- Record status: {'PASS': 9350, 'FAIL': 0, 'REVIEW_REQUIRED': 0}
- Integrity errors: none
- Leakage errors: none
- Splits: {'train': 18, 'development': 6, 'challenge': 6, 'ood_stress': 6}
- Tiers: {'0': 6, '1': 6, '2': 6, '3': 6, '4': 6, '5': 6}
- Families: {'nominal': 6, 'control_only': 5, 'engine_only': 3, 'engine_control': 8, 'delay_disturbance_thrust': 4, 'boundary_concurrent': 4, 'sensor_delay': 2, 'boundary_sensor_delay': 2, 'engine_partial': 2}
- Outcomes: {'successful': 36}
- Proposed-action saturation frequency: 0.442460
- Approval gate passed: True

The candidate records remain immutable. No sealed-test record was opened. `REVIEW_REQUIRED` is not equivalent to invalid data, but it is not silently treated as approval.
