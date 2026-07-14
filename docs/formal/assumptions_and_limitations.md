# Assumptions and limitations

The analysis concerns a finite-horizon mathematical abstraction of the repository's internal numerical simulator. It does not verify a real aircraft, avionics, hardware, external simulator, or unrestricted operating envelope.

The initial abstraction is an interval transition system derived from the simulator equations. Sound neural bounds use interval bound propagation (IBP) through the exact 42–256–128 policy. IBP is sound for the represented input box but can be conservative. Monitor, projection, fallback, delay-queue, and emergency-mode branches must all be retained or conservatively joined.

Plant coefficients come from source configuration; residual bounds must come from held-out numerical-simulator validation before Gate C. Calibration and validation seeds and scenarios remain disjoint. Until then, multi-step outputs are implementation diagnostics and must use `unknown_overapproximation` where provisional residuals or branch joins prevent proof.

The current point-mass simulator omits aerodynamic geometry, structural flexibility, detailed propulsion, spatial terrain, and many coupled wind effects. Its turbulence sampling is stochastic and is replaced in the formal model by explicit deterministic disturbance bounds. Model validity is limited to the state and failure ranges in the selected protocol.

The replay viewer consumes immutable completed logs and is visualization only. It does not run the controller, monitor, formal analyzer, or simulator and provides no validation evidence.
