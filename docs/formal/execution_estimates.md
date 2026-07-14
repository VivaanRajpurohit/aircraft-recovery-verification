# Recovery-envelope execution estimates

These are planning estimates, not measured envelope runtimes. They assume pairwise maps rather than the Cartesian product of every axis.

| Level | Cells | Horizon | Timeout/cell | Expected elapsed | Peak memory | Disk |
|---|---:|---:|---:|---:|---:|---:|
| Smoke | 6 | 5 steps | 2 s | under 1 minute | under 1 GB | under 5 MB |
| Pilot | 141 | 30 steps | 15 s | 10–35 minutes with 4 workers | 4–8 GB | under 100 MB |
| Full target | 1,134 | 100 steps | 60 s | 4–20 hours with 8 workers | 16–24 GB | 1–3 GB |

The full estimate has high uncertainty because branch splitting and interval-width stopping have not been benchmarked. Before approval, the smoke run must measure time, memory, branch count, output size, and timeout rate; pilot and full estimates must then be revised from observed data.
