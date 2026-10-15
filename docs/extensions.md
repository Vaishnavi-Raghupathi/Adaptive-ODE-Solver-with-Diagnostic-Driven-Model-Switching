# Meaningful next extensions, based on the measured results

The current deliverable is worth distributing as a small, honest benchmarking library. Neural corrections were never selected in the 60-case robustness study; adding more network architectures or marketing it as a universal neural solver is not supported by that evidence.

## Already implemented: a correction to the actual differential equation

The state-based polynomial and neural acceleration candidates learn `f(x,v)` corrections and integrate the resulting ODE. They can be transferred to a new initial state without fitting that new trajectory. This is a meaningful extension over the original time-conditioned trajectory residual NN, which cannot make that claim. The new polynomial model improved the missing-cubic future test in five of six noise/data-density conditions under the predeclared criterion, but its library explicitly contains the true missing cubic term. Sparse/noisy variants sometimes became unstable; those failures are retained.

## Recommended next: multi-trajectory fitting and a genuinely separate test family

Train a correction on multiple initial conditions, choose settings on separate validation trajectories, and test on a final unseen set of initial states/parameter ranges. The current API's fitting callback takes one trajectory, so this needs an explicit multi-trajectory interface rather than concatenating trajectories and accidentally differentiating across boundaries. Retain parameter fitting, polynomial correction and neural correction as baselines. This addresses the clearest limitation of a model trained on one trajectory and makes the transfer claim stronger.

Acceptance: source-disjoint training/validation/test trajectories; fixed parameter ranges and seeds; failures and test metrics reported for all held-out trajectories; no tuning on Day 5 or v2 final test sets.

## Recommended next: fit without differentiating noisy measurements

Current state corrections estimate acceleration from measured velocity, making smoothing/data density important. An integral/weak-form objective or direct trajectory-loss fitting can avoid explicit noisy derivatives. Compare the new objective against the current finite-difference/filter baseline under a new protocol. Do not assume that it improves stability until measured.

Acceptance: matched training data and compute budgets; noise/data-density sweep; full failure accounting; tests on separate final trajectories. Prefer a simple regularized polynomial model before adding a much larger neural training stack.

## Valuable stability extension: physically constrained corrections

The v2 study retained 21 unsuccessful polynomial variants, two conditions/seeds with no usable polynomial variant, and four failed new-initial-state predictions. Add constraints or regularization that favor bounded/dissipative oscillator behavior, and compare against the unconstrained model on a fresh final evaluation. The observed failures motivate this experiment; they do not by themselves prove which constraint will fix them. Report the tradeoff between stability and accuracy instead of removing divergent cases.

## Useful product extension: cost-aware recommendations

Currently selection minimizes validation MSE only. Add a configurable policy such as minimum runtime among candidates within an acceptable validation-error tolerance, plus an accuracy/cost Pareto table. Separate training, prediction and repeated-use cost so a model trained once for many queries is evaluated differently from a one-shot integration. Define the policy before measuring final tests; never pick a policy to make an already observed test winner look better.

Acceptance: deterministic policy tests, explicit cost accounting, benchmark under a fixed tolerance/budget, and comparisons against the existing accuracy-only policy.

## Useful realism extension: partial observations and measured data

The studies observe both position and velocity and know the initial state exactly. A position-only workflow, uncertain initial conditions, and a real recorded dataset would test more realistic identification problems. This requires defining the observation model and baselines, not treating a derived noisy velocity as known truth. Document units, sensor noise, run boundaries and data permissions; split by physical recording/run where possible.

## Longer-term research direction: recommend before running the full tournament

The current library runs the candidates and ranks them. A learned selector based on equation/data diagnostics could reduce tournament cost, but requires a much larger and varied benchmark dataset with test systems held out by family. Compare total selection overhead and achieved error against simple defaults such as LSODA or a prescribed explicit/implicit policy. This is a separate research contribution and should not be implied by the current package name.

The best near-term portfolio investment is multi-trajectory evaluation plus a measured improvement in stability or selection cost. A public PyPI badge alone adds less evidence than that work.
