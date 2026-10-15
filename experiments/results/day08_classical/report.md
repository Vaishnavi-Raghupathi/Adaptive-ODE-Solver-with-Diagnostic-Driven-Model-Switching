# Day 8: classical benchmark controls

Deterministic repeats measure timing variation only. Candidate failures remain visible; numerical reference disagreements are in reference_checks.json.

| System | Repeat | Method | Status | Validation MSE | Future test MSE | Selected |
|---|---:|---|---|---:|---:|---|
| lorenz | 0 | rk4 | ok | 1.659e-02 | 7.538e-02 | False |
| lorenz | 0 | rk45 | ok | 1.704e-09 | 1.021e-08 | False |
| lorenz | 0 | dop853 | ok | 1.590e-11 | 1.428e-10 | False |
| lorenz | 0 | bdf | ok | 9.533e-09 | 2.199e-08 | False |
| lorenz | 0 | radau | ok | 1.758e-14 | 7.866e-14 | True |
| lorenz | 0 | lsoda | ok | 9.613e-10 | 8.668e-09 | False |
| lorenz | 1 | rk4 | ok | 1.659e-02 | 7.538e-02 | False |
| lorenz | 1 | rk45 | ok | 1.704e-09 | 1.021e-08 | False |
| lorenz | 1 | dop853 | ok | 1.590e-11 | 1.428e-10 | False |
| lorenz | 1 | bdf | ok | 9.533e-09 | 2.199e-08 | False |
| lorenz | 1 | radau | ok | 1.758e-14 | 7.866e-14 | True |
| lorenz | 1 | lsoda | ok | 9.613e-10 | 8.668e-09 | False |
| lorenz | 2 | rk4 | ok | 1.659e-02 | 7.538e-02 | False |
| lorenz | 2 | rk45 | ok | 1.704e-09 | 1.021e-08 | False |
| lorenz | 2 | dop853 | ok | 1.590e-11 | 1.428e-10 | False |
| lorenz | 2 | bdf | ok | 9.533e-09 | 2.199e-08 | False |
| lorenz | 2 | radau | ok | 1.758e-14 | 7.866e-14 | True |
| lorenz | 2 | lsoda | ok | 9.613e-10 | 8.668e-09 | False |
| vanderpol | 0 | rk4 | ok | 7.465e-07 | 9.204e-04 | False |
| vanderpol | 0 | rk45 | ok | 4.173e-15 | 4.272e-12 | False |
| vanderpol | 0 | dop853 | ok | 4.963e-16 | 5.846e-13 | False |
| vanderpol | 0 | bdf | ok | 4.102e-13 | 6.077e-10 | False |
| vanderpol | 0 | radau | ok | 3.944e-18 | 3.784e-17 | True |
| vanderpol | 0 | lsoda | ok | 2.747e-12 | 3.559e-09 | False |
| vanderpol | 1 | rk4 | ok | 7.465e-07 | 9.204e-04 | False |
| vanderpol | 1 | rk45 | ok | 4.173e-15 | 4.272e-12 | False |
| vanderpol | 1 | dop853 | ok | 4.963e-16 | 5.846e-13 | False |
| vanderpol | 1 | bdf | ok | 4.102e-13 | 6.077e-10 | False |
| vanderpol | 1 | radau | ok | 3.944e-18 | 3.784e-17 | True |
| vanderpol | 1 | lsoda | ok | 2.747e-12 | 3.559e-09 | False |
| vanderpol | 2 | rk4 | ok | 7.465e-07 | 9.204e-04 | False |
| vanderpol | 2 | rk45 | ok | 4.173e-15 | 4.272e-12 | False |
| vanderpol | 2 | dop853 | ok | 4.963e-16 | 5.846e-13 | False |
| vanderpol | 2 | bdf | ok | 4.102e-13 | 6.077e-10 | False |
| vanderpol | 2 | radau | ok | 3.944e-18 | 3.784e-17 | True |
| vanderpol | 2 | lsoda | ok | 2.747e-12 | 3.559e-09 | False |
| robertson | 0 | rk4 | failed | — | — | False |
| robertson | 0 | rk45 | failed | — | — | False |
| robertson | 0 | dop853 | failed | — | — | False |
| robertson | 0 | bdf | ok | 4.346e-15 | 6.476e-15 | False |
| robertson | 0 | radau | ok | 3.111e-18 | 2.469e-18 | True |
| robertson | 0 | lsoda | ok | 1.155e-15 | 8.340e-16 | False |
| robertson | 1 | rk4 | failed | — | — | False |
| robertson | 1 | rk45 | failed | — | — | False |
| robertson | 1 | dop853 | failed | — | — | False |
| robertson | 1 | bdf | ok | 4.346e-15 | 6.476e-15 | False |
| robertson | 1 | radau | ok | 3.111e-18 | 2.469e-18 | True |
| robertson | 1 | lsoda | ok | 1.155e-15 | 8.340e-16 | False |
| robertson | 2 | rk4 | failed | — | — | False |
| robertson | 2 | rk45 | failed | — | — | False |
| robertson | 2 | dop853 | failed | — | — | False |
| robertson | 2 | bdf | ok | 4.346e-15 | 6.476e-15 | False |
| robertson | 2 | radau | ok | 3.111e-18 | 2.469e-18 | True |
| robertson | 2 | lsoda | ok | 1.155e-15 | 8.340e-16 | False |
| mismatch | 0 | rk4 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 0 | rk45 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 0 | dop853 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 0 | bdf | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 0 | radau | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 0 | lsoda | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 0 | calibrated_oscillator | ok | 7.098e-19 | 4.568e-19 | True |
| mismatch | 1 | rk4 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 1 | rk45 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 1 | dop853 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 1 | bdf | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 1 | radau | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 1 | lsoda | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 1 | calibrated_oscillator | ok | 7.098e-19 | 4.568e-19 | True |
| mismatch | 2 | rk4 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 2 | rk45 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 2 | dop853 | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 2 | bdf | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 2 | radau | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 2 | lsoda | ok | 1.278e+00 | 8.746e-01 | False |
| mismatch | 2 | calibrated_oscillator | ok | 7.098e-19 | 4.568e-19 | True |

Metrics average state dimensions in raw units, so do not compare absolute MSE between systems. Selection is validation-based and ignores runtime; timing is reported separately. Read docs/classical_protocol.md for scope and budgets.
