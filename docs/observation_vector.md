# Neural Observation Vector Contract

`OBSERVATION_FEATURES` is the authoritative fixed order. The current vector has
42 FP32 values:

1. time;
2. seven aircraft-state values;
3. four flight-envelope values;
4. seven control-health values;
5. five engine-health values;
6. latitude, longitude, and navigation-valid mask;
7. five environment values;
8. mission phase index, emergency flag, and active-failure count;
9. airport-available mask and six nearest-airport summary values.

The exact names and order are defined in
`aircraft_recovery.common.observations.OBSERVATION_FEATURES`. An absent airport
uses an explicit zero availability mask and zero-filled payload. Invalid
navigation uses a separate validity mask. Training will add normalization
statistics without changing semantic ordering; checkpoints must store the
feature-list version.

