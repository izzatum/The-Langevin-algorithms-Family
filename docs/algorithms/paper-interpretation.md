# Interpretation of the seismic Lip-Langevin paper

This note records the project's mathematical interpretation of Izzatullah, van Leeuwen, and Peter (2021), [R01 in the consolidated bibliography](../references.md#r01). The user confirmed that no separate erratum or author reference implementation was available and delegated the choice of a mathematically sound interpretation. These are project decisions, not an author-issued correction or a claim to reproduce unpublished author code.

The supplied journal PDF was examined as text. Algorithms 1–2 on printed p. 1529 and Equation (10) on p. 1531 were also inspected in rendered page images; this matters because text extraction loses square roots, signs, and fraction placement.

## Potential, drift, and mobility

Use the potential convention

\[
U(x)=-\log\pi(x)+C,\qquad \pi_T(x)\propto e^{-U(x)/T}.
\]

For constant symmetric positive definite mobility \(M\), the overdamped transition is

\[
x'=x-hM\nabla U(x)+\sqrt{2hT}\,M^{1/2}\xi,
\quad \xi\sim\mathcal N(0,I).
\]

Thus the drift is toward decreasing potential, equivalently increasing log density. Equation (6) of R01 uses \(+hM\nabla\log\pi\), consistent with this convention, whereas Equation (5) and the proposal lines of Algorithms 1–2 print \(-M\nabla\log\pi\). They cannot all describe the same target when the symbol denotes the log density. The implementation uses \(-\nabla U\), consistently with the negative log-posterior definition in Equation (3).

The noise covariance must be \(2hTM\). Multiplying a standard normal by \(M\) gives covariance \(M^2\), generally incorrect. The first-step initialization in Algorithm 2 prints \(M\), whereas its main loop and Algorithm 1 print \(M^{1/2}\). Use a valid matrix square root for every step. For identity mobility these expressions coincide, so the paper's unpreconditioned examples do not distinguish them.

State-dependent mobility requires additional mathematical treatment; the constant-mobility formula does not justify an arbitrary adaptive preconditioner. Continuous differentiability alone does not imply a globally Lipschitz gradient, despite the implication stated near Equation (4). Regularity assumptions must be checked for each target.

## Metropolis–Hastings correction

For a fixed step \(h\) and mobility \(M\), let

\[
q_h(y\mid x)=\mathcal N\!\left(y; x-hM\nabla U(x),\,2hTM\right).
\]

The correct log acceptance ratio is

\[
\ell= -\frac{U(y)-U(x)}T
       +\log q_h(x\mid y)-\log q_h(y\mid x),
\qquad
\log a=\min(0,\ell).
\]

Accept when \(\log u<\log a\), with \(u\sim\mathrm{Uniform}(0,1)\). The reverse proposal density is in the numerator of the density ratio. Algorithm 1 prints the forward density there, reversing the Gaussian factors. The library follows the standard MH rule, with [R19](../references.md#r19) as the MH methodological source and [R20](../references.md#r20) as the foundational Langevin convergence/ergodicity source. Computing in log space avoids avoidable overflow/underflow.

The acceptance decision covers the full state of each chain, including all parameter blocks. Rejection restores parameters and accepted-state potential/gradient caches; it must not retain a proposed gradient as though it belonged to the accepted state. Nonfinite proposals require a defined rejection or failure policy. A stochastic or state-mutating closure does not automatically provide an exact target density evaluation.

## Local Lipschitz recurrence

With constant mobility, define \(g_t=M\nabla U(x_t)\), dimension \(d\), and \(c_d=d^{-1/3}\). The recurrence displayed in Equation (8) and Algorithms 1–2 is

\[
h_t=\min\left\{
\sqrt{1+a_{t-1}}\,h_{t-1},\;
c_d\frac{\|x_t-x_{t-1}\|}{\|g_t-g_{t-1}\|}
\right\},
\qquad a_t=h_t/h_{t-1}.
\]

The square root acts on \(1+a_{t-1}\); it is not a linear growth factor. Using \(\nabla U\) rather than \(\nabla\log\pi\) does not change the norm of the gradient difference. The preconditioned gradient belongs in the denominator when reproducing this constant-mobility rule. The paper uses \(a_0=+\infty\) and an initial supplied \(h_0>0\).

An implementation must specify the following cases, which are not fully resolved by the printed pseudocode:

- **First transition:** use the user-supplied finite positive \(h_0\) before a gradient difference exists; do not invent a previous sample.
- **Zero gradient difference:** omit the local-Lipschitz bound, as discussed following Equation (8), rather than divide by zero. A simultaneous infinite growth bound needs a finite, explicitly documented safeguard or a clear failure; never pass an infinite step to the numerical kernel.
- **Repeated states:** the \(0/0\) ratio contains no curvature information. Treat it under the zero-gradient-difference rule, rather than silently interpreting it as a numerical zero step.
- **Rejection in Lip-MALA:** preserve the previous adaptation state when an update is performed only after acceptance. The paper's rejection branch assigns only \(x_t=x_{t-1}\); carrying forward \(h\) and \(a\) completes that branch explicitly. A different rejection policy is a different variant.
- **Floating-point safeguards:** any step cap/floor, finite initialization of \(a_0\), or change in update timing must be visible configuration and documented as an implementation extension. Norms and differences need dtype-appropriate numerical checks.

The expression \(d^{-1/3}\) is the paper's chosen heuristic scaling; it is not a general finite-dimensional optimality theorem for arbitrary targets.

## Continuous adaptation and frozen production

History-dependent adaptation makes the process different from a fixed-state MALA kernel. Using the ordinary Gaussian MH ratio at each iteration does not by itself prove that a continuously adapting chain has invariant marginal \(\pi\). The project's continuously adaptive Lip-MALA is therefore identified as a research method, with bias and stability to be measured. No unconditional exactness claim follows from the presence of accept/reject.

A separate practical variant adapts during warm-up, then freezes the finite positive step and constant mobility for production. Conditional on the frozen settings, production uses the ordinary valid MALA kernel. Its long-run claim still needs the usual target, irreducibility, and ergodicity conditions; finite warm-up does not create exact independent draws. This frozen variant is an implementation extension, rather than a literal reproduction of the paper's continuously adaptive algorithm.

Lip-ULA remains unadjusted. More iterations at a fixed nonzero step do not generally remove its discretization bias. The paper's qualitative variance-inflation observations motivate Gaussian bias experiments, but do not imply that every non-Gaussian target has the correct mean or a universally inflated covariance.

## Equation (10): a normalizable Rosenbrock interpretation

Rendered Equation (10) on p. 1531 prints

\[
\pi(x)\propto\exp\!\left(\alpha(x_1^2-x_2)^2+(x_1-\beta)^4\right),
\]

with \(\alpha=10\) and \(\beta=0.25\). The displayed exponent has a positive sign and no square root. This expression is not integrable over \(\mathbb R^2\). A normalizable interpretation uses

\[
U(x)=\alpha(x_1^2-x_2)^2+(x_1-\beta)^4,
\qquad \pi(x)=Z^{-1}e^{-U(x)},\quad \alpha>0.
\]

This is an explicit sign correction. It is supported independently by the paper's reported exact moments, rather than by an ambiguous text extraction.

To derive the normalizer and moments, set \(z=x_1-\beta\) and \(\epsilon=x_2-x_1^2\). The transformation has unit Jacobian and factors the density into \(e^{-z^4}\) and \(e^{-\alpha\epsilon^2}\). Consequently,

\[
Z=\frac{\Gamma(1/4)}2\sqrt{\frac\pi\alpha},\qquad
E[z]=E[z^3]=0,\quad
c=E[z^2]=\frac{\Gamma(3/4)}{\Gamma(1/4)},\quad
E[z^4]=\frac14,
\]

and \(\epsilon\sim\mathcal N(0,1/(2\alpha))\), independently of \(z\). Therefore

\[
\begin{aligned}
E[x_1]&=\beta,&\operatorname{Var}(x_1)&=c,\\
E[x_2]&=\beta^2+c,&
\operatorname{Var}(x_2)&=\frac14-c^2+4\beta^2c+\frac1{2\alpha}.
\end{aligned}
\]

At the paper's parameter values these evaluate to

| Quantity | Analytic value | R01 Table 3 |
|---|---:|---:|
| \(E[x_1]\) | 0.25000000 | 0.2500 |
| \(E[x_2]\) | 0.40048912 | 0.4005 |
| \(\operatorname{Var}(x_1)\) | 0.33798912 | 0.3380 |
| \(\operatorname{Var}(x_2)\) | 0.27026063 | 0.2703 |

Here \(Z\approx1.01607558\). This agreement validates the interpretation of the target moments, not any sampler or reproduction of the numerical trajectories. A separate Gaussian-transformed banana benchmark must retain its own definition and must not be mislabeled as Equation (10).

## Reproduction and evidence limits

Reproduction experiments should state target definition, corrected formulas, recurrence initialization/rejection policy, seed, dtype/device, step/gradient budgets, warm-up, and any caps. Compare analytic Gaussian covariance and the moments above; report uncertainty across seeds and residual finite-step bias. The paper's acceptance rate and iteration count are not sufficient validation metrics.

This note records the examined journal PDF. It does not claim an exhaustive erratum search, author-code reproduction, or a theorem proving the continuously adaptive implementation. Source review and remaining literature limitations are tracked in the [consolidated bibliography](../references.md#what-was-actually-reviewed).
