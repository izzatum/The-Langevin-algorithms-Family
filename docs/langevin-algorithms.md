---
title: "Langevin Algorithms: ULA, MALA and Related Methods"
description: "Understand Langevin sampling equations, ULA versus MALA, kinetic and adaptive methods, particle approximations and convergence checks, with runnable PyTorch tutorials."
---

# Langevin algorithms: ULA, MALA and related methods

**Langevin algorithms are gradient-based methods for exploring probability distributions.** They combine movement toward high target density with random noise. In Bayesian inference, the target is often a posterior distribution whose normalization constant is unknown. LaFA implements these methods in PyTorch alongside related particle approximations and uncertainty diagnostics.

This guide explains the conventions used by LaFA. The [algorithm register](research/algorithm-register.md) specifies exact implementations; the [24-source bibliography](references.md) supplies the original scientific references.

## From a probability density to Langevin dynamics

Write the target density as `π(x) ∝ exp(-U(x))`, where `U = -log π` up to an additive constant. For a Bayesian posterior, `U` combines the negative log likelihood and negative log prior. Its gradient therefore contains both data and prior information. The normalization constant cancels from gradients and Metropolis–Hastings ratios.

With constant positive diagonal mobility `M` and temperature `T`, the overdamped Langevin diffusion is

```text
dX_t = -M ∇U(X_t) dt + √(2 T M) dW_t.
```

Under suitable regularity and confinement assumptions, its invariant density is proportional to `exp(-U/T)`. LaFA defaults to **T = 1**, which corresponds to the specified target `π`. Changing temperature changes that target. The continuous diffusion and its numerical discretization have different convergence properties; see [R20](references.md#r20).

With step size `h > 0`, one Euler proposal is

```text
y = x - h M ∇U(x) + √(2 h T M) ξ,    ξ ~ Normal(0, I).
```

The square root is coordinatewise for diagonal mobility. Noise scales with the square root of the step size. LaFA's [API guide](api.md) describes batched potential shapes and device conventions.

## What is the difference between ULA and MALA?

The **unadjusted Langevin algorithm (ULA)** accepts every Euler proposal. Its steps are simple, but a fixed step generally produces an invariant distribution different from the desired target, when an invariant distribution exists. Smaller steps can reduce this discretization bias while increasing the work needed to explore the distribution ([R02](references.md#r02), [R20](references.md#r20)).

The **Metropolis-adjusted Langevin algorithm (MALA)** uses the same proposal and accepts it with probability

```text
α(x, y) = min(1, exp((U(x) - U(y))/T) q(x | y) / q(y | x)).
```

Here `q` is the Gaussian proposal density. Its forward and reverse means differ, so their ratio must be included. A rejected proposal leaves the chain at its previous position; those repeats belong in the trace. For a valid fixed kernel, the correction preserves `exp(-U/T)` ([R19](references.md#r19), [R20](references.md#r20)). It does not make a finite run independent or converged. [Tutorial 1](tutorials.md) compares ULA bias, MALA acceptance and adaptive steps on analytic and nonlinear targets, including methods discussed in [R01](references.md#r01).

## Compare the implemented method families

| Method | Main mechanism | Practical boundary in LaFA |
|---|---|---|
| ULA / MALA | Overdamped gradient-plus-noise proposals; optional MH correction | ULA has finite-step bias; fixed MALA still needs mixing checks. |
| Kinetic Langevin / BAOAB | Momentum, mass and friction; Euler or splitting integration | Discretizations differ; tuning and position accuracy depend on the target. [R05](references.md#r05), [R06](references.md#r06) |
| LipULA / LipMALA | Step sizes from local gradient differences | Continuous adaptation is approximate; `adapt_steps` freezes the production kernel. [R01](references.md#r01) |
| Anchored Langevin | Reference-potential drift/noise and an operational clock | Supplied exact reference; Euler bias remains; reflected extension is unavailable. [R03](references.md#r03) |
| HOLA | Higher-order overdamped updates with correlated noise | Needs higher derivatives; dense implementation has a dimension guard. [R07](references.md#r07) |
| Proximal interacting Langevin | Joint proximal maps, shared parameters and latent particles | MYIPLA/PIPULA/PIPGLA perform parameter learning with latent inference. [R10](references.md#r10) |
| Stein particle methods | Kernel-driven interacting transport | SVGD, projected, Newton and function-space variants have finite-particle approximations. [R11](references.md#r11)–[R14](references.md#r14) |

Kinetic methods augment position with momentum. BAOAB uses a particular splitting with an exact Ornstein–Uhlenbeck momentum substep; HOLA is an **overdamped** higher-order method. Equal iteration counts conceal their different gradient and derivative costs.

Stein variational gradient descent (SVGD) transports interacting particles using kernel attraction and repulsion. These particles are not independent MCMC chains. LaFA's projected SVGD uses a frozen-complement approximation; its Stein variational Newton implementation solves a block-diagonal system. The [register](research/algorithm-register.md) documents these boundaries.

## Which method should I try first?

For a smooth, deterministic potential with affordable gradients, use fixed-kernel MALA as an initial corrected baseline. Compare ULA when acceptance calculations are costly and discretization bias is acceptable and measurable. Check step sensitivity against a known reference where possible.

For persistent correlation, compare kinetic methods and parameter scaling before assuming more iterations will suffice. Annealing and replica exchange can help investigate multimodal exploration, but finite temperature schedules and incomplete mode visits remain limitations. Nonsmooth latent-variable learning requires the appropriate joint proximal map. Particle methods offer a different approximation whose quality depends on population size, kernels and the target.

Follow [tutorial 1](tutorials.md) for ULA/MALA, tutorial 2 for momentum and HOLA, tutorial 3 for temperatures and proximal learning, and tutorial 4 for Stein methods. Tutorial 5 studies kernel Stein discrepancy (KSD); tutorial 6 covers repulsive ensembles, birth–death, filtering and spatial inversion. Their [validation evidence](validation.md) separates these inference objectives.

## Frequently asked questions

### How do I assess convergence?

Use multiple independent chains, discard warm-up, preserve chain/time axes and examine traces, mode exploration, rank-normalized split R-hat, bulk/tail effective sample size and Monte Carlo standard error ([R21](references.md#r21), [R22](references.md#r22)). Check relevant predictions as well as coordinates. A high acceptance rate or small KSD does not establish convergence; KSD can miss modes ([R15](references.md#r15)). Diagnostics also cannot correct a misspecified target.

### Is Langevin sampling the same as optimization?

Both use gradients, but optimization seeks a minimizing point, whereas sampling estimates a distribution. Langevin noise is calibrated to a sampling model. LaFA's optimizer-style interface changes how parameters are updated, not the statistical objective. A posterior mean or credible interval needs an adequate sampling run.

### Does LaFA implement stochastic gradient Langevin dynamics?

There is no dedicated SGLD implementation. Changing minibatches introduce a different gradient estimator. The documented MALA closure must evaluate a deterministic full-data potential; substituting minibatch losses does not preserve its stated acceptance calculation.

### Are CPU and GPU results validated?

Recorded evidence includes **546 CPU tests passed**, **77 CUDA cases skipped** and **six executed CPU notebooks**. Real GPU validation remains pending. Short experiments include poor mixing and uncertain references. Consult the [validation report](validation.md) before making accuracy, calibration or performance claims, and cite both the software revision and the relevant [method sources](references.md).
