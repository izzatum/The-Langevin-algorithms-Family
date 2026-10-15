# Statistical validation and scalable PyTorch execution implementation plan

> Agentic execution: use subagent-driven implementation with independent reviews. The user explicitly authorized improvements 2, 3, 5 and 6 and fixing discovered bugs; execution proceeds under that authorization.

**Goal:** strengthen uncertainty estimates and experiments, reduce optional sampling overhead, batch neural-network targets, and validate a full nonlinear spatial posterior.

**Architecture:** retain existing tensor samplers, optimizer adapters and safe snapshot defaults. Add explicit opt-in execution controls, independent-chain diagnostics, chunked functional model evaluation, and reproducible benchmark drivers. Reference solvers belong to the benchmark layer, with separate numerical tests and sources.

**Tech stack:** Python >=3.11, PyTorch >=2.6, optional existing notebook/plot dependencies. Core numerical runtime remains PyTorch only.

**Spec:** [approved architecture](architecture.md), the user's selection of improvements 2/3/5/6, and the contracts below.

## Global constraints

- Preserve existing call signatures/default behavior; new controls are keyword-only.
- Preserve owned returned tensors, finite-state validation, rejection/rollback and checkpoint/RNG semantics.
- No scientific guarantee follows from a unit test or a short experiment. Show actual convergence, bias and runtime limits.
- Independent-chain diagnostics accept `(chains, draws, dimensions)`; interacting particles must not be relabeled independent MCMC chains.
- Retain exact deterministic full-potential semantics for corrected samplers. No minibatch target or uncorrected adaptive preconditioning under an exactness claim.
- CPU work is available. CUDA tests must explicitly skip locally and remain pending until real hardware execution. No new RunPod allocation in this milestone; preserve the existing $10 cumulative budget ledger.
- Recording infrastructure overhaul (item 4), GPU provisioning (item 1), a general run driver and publication infrastructure are outside this selected milestone.
- Fix reproducible bugs found in the repository, with focused regression coverage; do not claim that every possible bug has been eliminated.

## Review focus

1. Constant, heavy-tailed, antithetic and very short chains: diagnostics return documented undefined values or reject invalid input, never fabricated convergence.
2. Fast stepping: retained results cannot alias later state; invalid MALA proposals reject correctly; optimizer failure restores numerical state and RNG as specified.
3. Batched models: tied parameters, buffers, dtype/device, chunk boundaries and independent-chain semantics match single-state evaluation.
4. Full-grid reference: reference uncertainty and finite mixing are measured; calibration datasets are generated from the stated prior and likelihood.
5. Benchmarks: retain chain/time axes, count actual work, distinguish posterior/learning/filtering objectives, and reject unavailable CUDA without fallback.

## Tasks and interfaces

### Task 1: Independent-chain diagnostics

Files: `src/lafa/diagnostics/chains.py`, `src/lafa/diagnostics/__init__.py`, `tests/test_chain_quality.py`, optional reference fixtures under `tests/data/`.

Produces `bulk_effective_sample_size(chains)`, `tail_effective_sample_size(chains, *, probability=0.05)`, `mcse_mean(chains)`, and `mcse_quantile(chains, probability=0.5)`, each returning one value per dimension. Bulk ESS uses split rank-normalized draws; tail ESS is the minimum lower/upper indicator ESS. Document the conservative ESS cap and valid sample-count requirements. Quantile MCSE must use a cited, defined estimator rather than pretending quantile draws are independent. Retain existing raw ESS and R-hat APIs.

- [ ] Add independent reference tests for iid, AR(1), ties, heavy tails, degenerate/short chains, dtype/device and extremes.
- [ ] Implement the diagnostics and check against an established external diagnostic implementation or frozen independently generated reference values; optional reference dependencies never enter core requirements.
- [ ] Run targeted tests and record formulas, source references, actual agreement tolerances and limitations.

### Task 2: Sampling execution and optimizer fixes

Files: `src/lafa/samplers/__init__.py`, `src/lafa/optim/__init__.py`, `src/lafa/functional/__init__.py`, new `src/lafa/execution.py` if useful, `tests/test_execution.py`, `benchmarks/performance.py`.

Produces native sampler `step(*, fields=None)`: `None` preserves the full `StepResult`; an empty tuple executes a transition and returns `None`; a validated subset of `position`, `potential`, `gradient`, `accepted`, `stats` returns an owned dictionary containing only requested fields. Add `advance(steps)` as bounded-memory no-result stepping, preserving algorithm-specific histories and checks. Do not change optimizer `step` return conventions.

Add an opt-in compiled target/gradient execution path for compatible pure tensor potentials; its exact public constructor/helper must be documented in the report before benchmark integration. Measure full sampling, including target/gradient/randomness/validation and selected output. Keep eager default and clear errors/fallback policy for unsupported targets. Reuse parameter-scale buffers when group learning rates are unchanged without caching stale closure gradients; retain externally changed parameter/group behavior.

- [ ] Test full/selected/no-result state and RNG equivalence for ULA/MALA/Lip/BAOAB/kinetic paths, snapshots, adaptation, rejection, exceptions and checkpoints.
- [ ] Implement selective results and measured low-allocation optimizer changes; reproduce and repair rollback/state bugs encountered.
- [ ] Verify eager/compiled agreement and run end-to-end performance sweeps with compile initialization separate, float32/64 and honest CPU/CUDA metadata.

### Task 3: Batched module targets and function-space VJPs

Files: `src/lafa/targets.py`, `src/lafa/particles/function_space.py`, `tests/test_batched_targets.py`, new `benchmarks/neural_network.py`.

Extend `ModulePotential(module, loss, *, args=(), kwargs=None, batch_mode='vmap', chunk_size=None)` to accept `(D,)` and `(chains,D)` parameter states, preserving one loss per state. Modes are explicit `vmap` or `loop`; unsupported transformed operations must give an actionable error recommending loop mode, with no silent catch-all suppression. Add a target `value_and_grad` path supporting both shapes without retaining graphs. Chunking bounds intermediate activation memory; copied buffers preserve existing mutation isolation and tied-weight safeguards.

Extend `FunctionSpaceSVGD(..., vectorized=False, chunk_size=None, **kwargs)` to opt into chunked batched function maps and VJPs while preserving the existing loop default; inherited function-space repulsive variants must work too.

- [ ] Test values/gradients/VJPs against independent single-state loops, chunk boundaries, tied weights, buffer mutation, invalid shapes/devices and input ownership.
- [ ] Implement batched calls and show a small Bayesian neural-network posterior/predictive example with true full-data likelihood and prior.
- [ ] Run CPU scaling measurements across model/chain/chunk sizes, recording correctness and memory/timing limits; CUDA cases skip honestly.

### Task 4: Cross-family statistical benchmarks

Files: `benchmarks/accuracy.py`, new `benchmarks/families.py`, `tests/test_benchmark_quality.py`, result artifacts under `benchmarks/results/cpu/`.

Consumes Task 1 diagnostics. Preserve chain/time axes until diagnostics are computed; flattened draws may still be used for marginal moments/plots. Produce JSON containing budgets, seeds, measured runtime, target/gradient/HVP/prox counts as applicable, finite reference errors and diagnostic applicability. Undefined diagnostic values serialize as null with an explanation, never invalid JSON.

Cover all six families using applicable named representatives and existing independent references: overdamped/inertial/specialized posterior methods on known distributions; particle empirical-distribution error; KSD against direct reference samples; proximal parameter-learning objectives against integrated likelihood; filtering against the Kalman reference. Compare multiple step sizes or ensemble sizes and multiple seeds. Do not force all outputs into MCMC ESS or identical transition-cost claims.

- [ ] Test report semantics, actual evaluation counters, diagnostic axes, method coverage and no-CUDA fallback.
- [ ] Implement drivers and execute bounded multi-seed CPU experiments with uncertainty across repetitions and explicit limitations.
- [ ] Retain reproducible JSON and compact figures showing accuracy versus measured cost.

### Task 5: Full-grid nonlinear posterior and calibration

Files: `src/lafa/problems/tomography.py` only if fixture extensions are needed; new `benchmarks/nonlinear_calibration.py`, optional `benchmarks/reference_sampling.py`, `tests/test_nonlinear_calibration.py`, result artifacts under `benchmarks/results/cpu/`.

Consumes Task 1 diagnostics. Infer every latent cell of a manageable 2D grid (at least 3x3), preserving the genuine nonlinear transmission forward model. Use an independent benchmark-only elliptical-slice reference for the Gaussian-prior model (Murray, Adams & MacKay, 2010), with convergence diagnostics, independent seeds and measured reference MCSE. Validate the reference against an analytic Gaussian conditioning fixture before using it. Do not claim an exact finite-run reference.

Generate repeated synthetic datasets by sampling full-grid latent truth from the documented prior and fresh observation noise. Report cell/field and posterior-predictive coverage, binomial or dataset-level uncertainty, reference mean/interval uncertainty, candidate step-size sensitivity and a second grid resolution. Clearly separate repeated-dataset calibration from repeated chains on one dataset; do not treat correlated cells as independent binomial trials.

- [ ] Test independent-reference invariants and Gaussian posterior agreement, reproducible prior-predictive generation, full-grid dimensions and report formulas.
- [ ] Run bounded CPU reference/candidate experiments on 3x3 and 4x4 grids and repeated datasets; publish actual mixing and calibration limits.
- [ ] Produce a figure with truth, posterior mean/SD and coverage/reference comparisons, retaining commands and seeds.

### Task 6: Repository bug audit, integration and documentation

Files: owned fixes assigned after read-only audit; root owns README/docs/references, notebook updates and final artifacts.

- [ ] Audit numerical contracts, state/checkpoint recovery, shape/dtype/device handling, recording and specialized drivers; reproduce actionable issues and assign non-overlapping fixes.
- [ ] Independently review each task; fix findings with targeted tests before integration.
- [ ] Update the single bibliography, API/algorithm/validation guides and concise README; update relevant existing tutorials to demonstrate new behavior.
- [ ] Run the complete CPU suite, all six notebooks, package build, actual benchmark commands and documentation examples. Preserve CUDA skips separately.
- [ ] Review the whole branch, push a PR, pass GitHub checks, and merge under the user's existing publication authorization.
