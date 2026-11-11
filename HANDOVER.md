# HANDOVER

**Objective:** fix the nine audit findings and test them (request 2); the audit itself was request 1. Branch `claude/friendly-carson-3wgbvb`, draft PR #6.

**State:** all nine findings are fixed and committed, one commit per group (`git log ce4b213..HEAD`). Each group was implemented in an isolated worktree, reviewed adversarially by a separate agent (one clean, six with minor defects only, compile after one repair round), then integrated here; the reviewers' minor defects were closed inline.

## Findings and where they were fixed

1. **Superseded.** The audit said `compile_potential` returned a silently wrong gradient on new PyTorch (`[2,2]` vs "true" `[3,3]`). Eager autograd gives the same `[2,2]`, because `.item()` detaches there too. What was real: `tests/test_execution.py::test_compilation_failure_is_explicit_without_silent_fallback` assumed `fullgraph` rejects `.item()`, true on torch 2.6 to 2.9 and false from 2.10. Fixed in the test (data-dependent branch) plus a docstring note in `src/lafa/execution.py`; no runtime cross-check.
2. `positive_transform(potential, *, temperature=1.)` scales the Jacobian term — `src/lafa/targets.py`, `tests/test_targets_transform.py`.
3. `{'params': tensor}` groups accepted; bare Tensor as `params` raises `TypeError` — `src/lafa/optim/__init__.py`.
4. LipULA/LipMALA adapters follow `lr` changes (step × new/old group-0 lr, clamped at `max_step_size`; optional checkpoint key `sampler_lr`) — same file.
5. Rollback on `BaseException` — same file.
6. SALEnKF pool transition and process Cholesky once per inner step — `src/lafa/specialized/filtering.py`, `tests/test_filtering_cost.py`.
7. SVGD family and function-space methods retain the accepted candidate's evaluation — `src/lafa/particles/{svgd,function_space,repulsive}.py`, `tests/test_particle_caching.py`.
8. ReplicaExchange carries (U, grad U); optimizer adapters reuse the cached gradient for the same closure object at bit-identical parameters — `src/lafa/specialized/tempering.py`, `src/lafa/optim/__init__.py`.
9. Zero-size params exempt from the alias check; `ScheduleComplete(RuntimeError)` + `finished` replace `StopIteration`; bool rejected for `adapt_steps`, `steps_per_temperature`, `steps_per_sigma`, `inner_steps`, `burnin`; `adapting` semantics documented and pinned (computation unchanged: it means "the next transition still adapts"); `MANIFEST.in` ships `tests/conftest.py`; `pyproject.toml` reads the version from `lafa.__version__`.

## Decisions and why

- No compiled-versus-eager runtime check: it would not have caught the original case, and the docstring now states that accepted constructs may differ from eager.
- Retained evaluations are never serialised (ReplicaExchange, particles, optimizer closure cache): a checkpoint loaded against a changed target must not reuse a stale gradient. Cost: for ReplicaExchange a resume is bit-exact only when a row's value does not depend on its batch position (reviewer measured rounding-level gaps for softplus/log-cosh potentials); documented in the class docstring and `docs/api.md`.
- Lip `lr` change at the cap: lr ratios between groups are always honoured, so a clamped increase of group 0 lowers the other groups' effective steps. Documented and pinned (`test_lip_clamped_group_zero_increase_lowers_the_other_groups_step`) rather than adding a second factor to the mobility.
- Optimizer checkpoints written with a bool `adapt_steps` load as the equal int (`load_state_dict`), since the samplers now reject bool.
- ReplicaExchange drops its carried pair across the position commit so an interrupt cannot leave a stale pair.
- ProjectedSVGD untouched.

## Values (source)

- Baseline before fixes: 545 passed, 1 failed, 77 skipped on Python 3.13 / torch 2.15 (audit run).
- After fixes, final run of `scratchpad/verify.sh` on a copy of the tree (logs in `scratchpad/verify/`, session-local): 890 passed, 78 skipped on Python 3.13 / torch 2.15.0 (`t215.log`), Python 3.13 / torch 2.6.0 (`t26.log`) and Python 3.12.3 / torch 2.6.0 (`t312.log`); `python -m build` succeeds and the extracted sdist gives 888 passed, 80 skipped (`tsdist.log`; the difference is `tests/test_packaging_metadata.py`, whose repository-file checks skip inside an sdist); `mkdocs build --strict` and `scripts/check_docs.py` pass; all six notebooks execute (`notebooks/execution-results.json`).
- New tests were mutation-checked: each group's reviewer ran its own mutants, and the follow-up tests added at integration each fail against the defect they target on both torch versions (`scratchpad/mut/run.py`).
- Repro after fixes (`scratchpad/repro_after.py`, torch 2.15): `positive_transform` T=2 E[z]=1.964 (was 1.00); optimizer MALA 11 closure calls / 10 steps (was 20); SVGD 11 target batches / 10 steps (was 20); FunctionSpaceSVGD 256 `function_map` calls over 10 steps for 8 particles, cold start included (was 480); SALEnKF 610 transition row-evals per stage, m=10, pool=50 (was 5110); ReplicaExchange 11 batches for construction + 10 steps (was 21).

## Behaviour changes callers can notice

- `AnnealedLangevin` / `NoiseAnnealedLangevin` raise `ScheduleComplete` instead of `StopIteration`.
- Reuse assumes a deterministic target that changes only through the sampled state. After changing data, buffers or other parameters in place: `invalidate_cache()` (optimizer adapters, particles, ReplicaExchange). A non-deterministic target is now drawn once per step where it was drawn twice.
- Bool counts raise `ValueError`.

## Open items

- **New, not fixed (outside the nine findings):** a process can compile only eight distinct potentials; the ninth `compile_potential` evaluation raises the explicit `RuntimeError` (cause `RecompileLimitExceeded` on 2.6, `FailOnRecompileLimitHit` on 2.15) until `torch.compiler.reset()`. All compiled potentials share one `evaluate` code object. Repro: `scratchpad/limit8.py`.
- Recorded evidence predates these changes and was deliberately not regenerated: "546 passed, 77 skipped" in `README.md:15`, `docs/index.md:76`, `docs/langevin-algorithms.md:84`, `docs/validation.md:21,25`; `benchmarks/results/cpu/pytest.xml`; work counts and timings in `benchmarks/results/*.json` (SALEnKF, SVGD, optimizer MALA rows).
- `scripts/check_docs.py:277` still hard-codes `"0.1.0"`; `CITATION.cff` and `mkdocs.yml` are hand-maintained copies that `tests/test_packaging_metadata.py` checks for drift.
- `backward_probabilities(block_size=True)` is still accepted (treated as 1).
- Not verified: CUDA paths (77 CUDA-parametrized tests skipped on CPU); paper-specific constants (HOLA taming, SVN Eq. 17, SA-LEnKF, R10 proximal schemes); PyTorch releases other than 2.6.0 and 2.15.0 in the full suite (a reviewer ran `tests/test_execution.py` on 2.7.1 to 2.14.1).
