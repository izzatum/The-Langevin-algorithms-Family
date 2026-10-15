# Reproduce the scientific benchmarks

Run from the repository root after installing `.[test,notebooks]`. Drivers save software/device metadata, seeds, budgets and actual measurements. They use one CPU thread by default; an explicit unavailable CUDA request fails. The full-grid calibration driver currently runs on CPU only.

| Driver | Question and reference | Included evidence |
|---|---|---|
| `accuracy.py` | ULA/MALA moments and independent-chain diagnostics on Gaussian, mixture, Rosenbrock and reduced nonlinear inversion; analytic/quadrature references | [24 CPU runs](results/cpu/accuracy.json) |
| `families.py` | Representative methods from all six families; distinguish posterior, particle transport, shared-parameter learning, KSD and filtering objectives | [114 runs / 38 summaries](results/cpu/families.json), [error/cost figure](results/cpu/families_accuracy_cost.png) |
| `performance.py` | Whole-step eager/compiled and full/selected/no-result execution; optimizer scale caching; particle/recording costs | [250 CPU configurations](results/cpu/performance.json) |
| `neural_network.py` | Full-data Bayesian neural-network target and function VJPs; loop/vmap/chunk scaling and MALA predictive summaries | [108 configurations and posterior example](results/cpu/neural-network.json) |
| `nonlinear_calibration.py` | Full 3×3/4×4 latent fields; independent elliptical-slice references and repeated prior-predictive datasets | [calibration](results/cpu/nonlinear-calibration.json), [fields](results/cpu/nonlinear-fields.png), [error/coverage](results/cpu/nonlinear-error-coverage.png) |

```bash
python benchmarks/accuracy.py --device cpu --output /tmp/lafa-accuracy
python benchmarks/families.py --device cpu --output /tmp/lafa-families
python benchmarks/performance.py --device cpu --compile --steps 120 --repeats 5 --output /tmp/lafa-performance.json
python benchmarks/neural_network.py --device cpu --burnin 5000 --draws 10000 --output /tmp/lafa-neural
python benchmarks/nonlinear_calibration.py --reference-draws 3000 --draws 2000 \
  --output /tmp/lafa-nonlinear --raw-output /tmp/lafa-nonlinear-raw
python benchmarks/nonlinear_calibration.py --refine-reference-draws 16000 --reference-warmup 2000 \
  --output /tmp/lafa-nonlinear --raw-output /tmp/lafa-nonlinear-raw
python benchmarks/nonlinear_calibration.py --grids 4 --refine-reference-draws 32000 \
  --reference-warmup 2000 --reference-seed-offset 60000000 \
  --output /tmp/lafa-nonlinear --raw-output /tmp/lafa-nonlinear-raw
```

The refinement commands reuse candidate chains and refine references; exact budgets for each saved reference are recorded in its JSON row. Raw chain/time tensors can be large and belong outside Git. Full-grid raw output is opt-in; the neural-network driver always writes its raw posterior trace alongside results. See `--help` for smaller smoke budgets. Compilation/cache initialization is saved separately from warmed step timing; caches are not forced cold.

Retained independent MCMC traces have `(chains, draws, dimensions)` axes until diagnostics are computed. Flattening is reserved for marginal summaries/figures. Particle populations, shared learning parameters and filtering stages are not independent MCMC chains. Family counters measure actual callbacks/rows, not interchangeable FLOPs. Diagnostics have separately recorded timing and do not inflate sampling work counts.

Three family replications give descriptive seed variability. Full-grid coordinate coverage uses independent datasets as trials and Wilson intervals; cells within a field are correlated. Eight datasets per grid give coarse coverage estimates. Elliptical-slice references are finite Monte Carlo estimates with their own Rhat/ESS/MCSE, not exact posterior truth. A reference mixing screen is a heuristic.

Short chains show poor mixing, finite ensembles retain approximation error, and compiled targets sometimes run slower. CPU RSS is a cumulative process high-water mark; activation-tensor sizes do not measure total memory. Read [validation and limitations](../docs/validation.md) before interpreting speed, accuracy or uncertainty. No included artifact establishes real CUDA performance.
