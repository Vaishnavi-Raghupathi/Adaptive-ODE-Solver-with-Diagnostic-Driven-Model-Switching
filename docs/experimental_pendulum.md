# First measured-data pilot: double pendulum

The [MultiArm-Pendulum repository](https://github.com/dynamicslab/MultiArm-Pendulum)
accompanies Kaheman et al., *The Experimental Multi-Arm Pendulum on a Cart: A
Benchmark System for Chaos, Learning, and Control*
([paper](https://arxiv.org/abs/2205.06231), [archive](https://zenodo.org/records/6633719)).
Its [data description](https://github.com/dynamicslab/MultiArm-Pendulum/blob/e99a7e13fe2adc8fc647321a156b6cb59a48a698/Datas/DoublePendulum/DataExplanation.md)
identifies four measured double-pendulum free-swing files, with the cart fixed,
and distinguishes them from experiments with external cart excitation.

## Reproduce

Install this project and run from the repository root:

```bash
python examples/05_experimental_pendulum.py \
  --download --source runs/experimental-pendulum/source \
  --output runs/experimental-pendulum/pilot
```

Use a new output directory for each evaluation. Later runs can omit `--download`
to use the cached files. The example downloads about 19.5 MB (18.6 MiB), including
the upstream README, data description and MIT license. It verifies each file
against a pinned Git blob from commit
`e99a7e13fe2adc8fc647321a156b6cb59a48a698` and records SHA-256 hashes. No upstream
MATLAB scripts are executed. Data are downloaded separately and are not included
in the Python package. Preserve the upstream license and cite the dataset paper
when reusing the measurements.

## Fixed pilot protocol

| Files | Role | Complete recordings | Excerpt used per file |
|---|---|---|---|
| FreeSwing 1 and 2 | Training | 170.833 and 182.528 seconds | First 5 seconds |
| FreeSwing 3 | Validation | 150.019 seconds | First 5 seconds |
| FreeSwing 4 | Test | 208.920 seconds | First 5 seconds |

Each excerpt has 5,001 observations at the native 1 ms interval. The state order
is `Theta1, Theta2, dTheta1, dTheta2`, preserving upstream angular positions and
velocities. There is no new filtering, derivative construction for observations,
or resampling. The learned ODE candidates still estimate their training targets
using their usual within-record finite differences. All full recordings pass
finite-value and strictly increasing, uniform-time checks.

The first observation initializes each validation/test rollout; subsequent
observations do not reset it. The score normalizes each state by its training
standard deviation. The recommendation requires validation normalized RMSE at
most 0.1, then minimizes median prediction time over three repeated validation
rollouts. This threshold is an illustrative pilot convention, not a requirement
established by an application. The file roles, five-second horizon and threshold
were fixed before running the model comparison.

No physical equations or parameters are supplied. Consequently this experiment
compares learned dynamics and a constant initial-state baseline. It does **not**
compare the accuracy of RK45, Radau or other numerical integrators. Calibration
on training experiments followed by a numerical tolerance sweep would be a
separate experiment. Published parameters should not be assumed independent of
the held-out files without establishing their calibration provenance.

## Initial result

| Candidate | Validation normalized RMSE | Test normalized RMSE | Outcome |
|---|---:|---:|---|
| Linear learned ODE | 1.38224 | 0.359124 | Does not meet validation target |
| Constant initial state | 1.39077 | 1.22585 | Does not meet validation target |
| Quadratic learned ODE | Unavailable | Not evaluated | Validation integration failed |

The quadratic model's integrator reported that its required step size was smaller
than floating-point spacing. The platform retained this failure and recommended
**no method**. The lower linear-model score on the test file does not override
the failed validation requirement. These findings expose inadequate model
coverage for this particular task; they do not establish that the measured data
are unsuitable or that another numerical integrator would fix the learned model.

Separate source files are treated as separate experimental runs. Independence
across devices or acquisition sessions is not established. One validation and one
test recording cannot establish broad generalization; no bootstrap interval is
reported. Upstream velocity estimation or filtering may also affect apparent
forecasting accuracy. Observation error combines model, initial-state and
measurement errors and must not be called numerical truncation error.

The output includes `report.md`, `results.json` with source provenance and caveats,
`metrics.csv`, `trajectories.npz`, `data-audit.json`, four CSV inputs, and a manifest.
The CSV export can be replayed through the installed CLI:

```bash
adaptive-solver trajectories runs/experimental-pendulum/pilot/manifest.json \
  --metric normalized_rmse --policy fastest_within_target --accuracy-target .1 \
  --timing-repeats 3 --output runs/experimental-pendulum/replay
```

The CLI writes the complete comparison and returns exit status 1 when no method
qualifies. This is the expected outcome for this pilot, rather than a failed
file conversion. The Python example returns 0 for a completed comparison even
when the recommendation is empty. CSV/CLI replay reproduced the original
candidate scores to `1e-12` numerical tolerance.

The next modeling step is to define a physics-based or trigonometric candidate
using training data, select settings using validation data, and evaluate on new
held-out experiments. The inspected fourth file is now a development reference;
repeatedly tuning against its outcome would invalidate an untouched-test claim.
