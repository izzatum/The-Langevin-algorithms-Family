# HANDOVER

**Objective:** audit the codebase for bugs/issues and suggest fixes (read-only; no source changed, nothing committed).

**Method:** read all of `src/lafa/`, skimmed `benchmarks/accuracy.py`, `benchmarks/performance.py`, `scripts/execute_notebooks.py`, CI workflows, README, `docs/api.md`; ran the suite and call-count repros on Python 3.13 / torch 2.15.0+cpu.

**Established values (source):**
- Test suite: 545 passed, 1 failed, 77 skipped (CUDA) — `python -m pytest -q`, also from the built sdist.
- Failing test: `tests/test_execution.py:255` (`test_compilation_failure_is_explicit_without_silent_fallback`).
- Repro numbers (script was in the session scratchpad, not kept): compiled `.item()` potential gives grad `[2,2]` vs true `[3,3]`; `positive_transform` at T=2 gives E[z]=1.00 vs 2; optimizer MALA 20 closure calls / 10 steps; SVGD 20 target batches / 10 steps; FunctionSpaceSVGD 480 `function_map` calls for 8 particles x 10 steps; SALEnKF 5110 transition row-evals per stage vs 500 needed (m=10, pool=50).

**Findings, by location:**
1. `src/lafa/execution.py:17-33` — no check that the compiled gradient matches eager; silent wrong gradient on new torch. CI pins `torch==2.6.0` (`.github/workflows/tests.yml:20`), `pyproject.toml` allows `>=2.6`.
2. `src/lafa/targets.py:60-62` — `positive_transform` Jacobian term is tempered when `temperature != 1`.
3. `src/lafa/optim/__init__.py:29` — `{'params': tensor}` group rejected with a misleading alias error.
4. `src/lafa/optim/__init__.py:184-188` — LipULA/LipMALA adapters ignore later `lr` changes.
5. `src/lafa/optim/__init__.py:120` — rollback catches `Exception` only; Ctrl-C leaves the proposal in the parameters.
6. `src/lafa/specialized/filtering.py:205-207,173-180` — pool transition + Cholesky recomputed per particle row.
7. `src/lafa/particles/function_space.py:72-84`, `svgd.py:130-136` — redundant target/function evaluations per step.
8. `src/lafa/specialized/tempering.py:106`, `optim/__init__.py:181-183` — current-state gradient recomputed every step.
9. Minor: `targets.py:113` zero-size params collide; `tempering.py:35,67` raises `StopIteration`; bools accepted as integer counts (`samplers/__init__.py:185`, `tempering.py:28,60`); `samplers/__init__.py:200` vs `:207` (`<=` vs `<`); `MANIFEST.in` omits `tests/conftest.py`; version string in four files.

**Not verified:** paper-specific constants (HOLA taming, SVN Eq. 17, SA-LEnKF, R10 proximal schemes) against the sources; all CUDA paths; `scripts/runpod_*.py`, `scripts/check_docs.py`, `benchmarks/families.py`, `neural_network.py`, `nonlinear_calibration.py`, notebooks.

**Remaining:** user to decide which findings to fix; none are implemented.
