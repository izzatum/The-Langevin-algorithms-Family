# Scientific references and algorithm coverage

This is the project's consolidated scientific bibliography. Reference IDs are stable: append new records rather than renumbering existing ones. A citation identifies a mathematical source; it does not establish that a particular software implementation is complete, correct, or experimentally validated. Implementation maturity and supported variants belong in the [algorithm register](research/algorithm-register.md).

Metadata were checked on 16 October 2026 (client-local date) using Python `urllib` against original arXiv records, PMLR proceedings pages, and DOI metadata deposited with Crossref. All authors are listed. Where a proceedings/journal version was not verified, the venue is explicitly the arXiv preprint, rather than an inferred conference. The arXiv year and publication year can differ.

## Algorithm-to-reference map

| Algorithm or family | Reference IDs | Scientific scope and implementation boundary |
|---|---|---|
| Unadjusted Langevin algorithm (ULA) | [R01](#r01), [R02](#r02), [R20](#r20) | Euler discretization with finite-step invariant bias; constant and decreasing steps have different claims. |
| Metropolis-adjusted Langevin algorithm (MALA) | [R01](#r01), [R19](#r19), [R20](#r20) | Fixed-kernel Gaussian proposal with the reverse/forward MH density ratio. |
| Lip-ULA; continuously adaptive Lip-MALA | [R01](#r01) | Local Lipschitz history and square-root growth cap. Continuous adaptation is a research variant; acceptance/rejection alone does not prove exactness. See the [interpretation note](algorithms/paper-interpretation.md). |
| Warm-up adaptation followed by frozen MALA | [R01](#r01), [R19](#r19) | Documented implementation extension; a fixed valid production kernel, rather than a literal reproduction of the continuously adaptive pseudocode. |
| Anchored Langevin | [R03](#r03) | Specified from arXiv v1, Eqs. (5)–(7), (16), (33)–(34), (41)–(43), Algorithms 1–2; implementation/validation tracked separately. A quadratic attraction term is not a substitute. |
| Reflected anchored Langevin | [R04](#r04) | Not yet implemented: separate constrained extension, not implied by unconstrained anchored coverage; only targeted full-text review completed. |
| BAOAB | [R05](#r05) | Kinetic splitting with exact Ornstein–Uhlenbeck substep; superconvergence depends on assumptions and observable. |
| Kinetic/underdamped Langevin | [R05](#r05), [R06](#r06) | Momentum, mass, friction, and noise convention must be specified. BAOAB and the discretization analyzed in R06 are distinct schemes. |
| Higher Order Langevin Monte Carlo (HOLA) | [R07](#r07) | Specified from arXiv v3, Eqs. (2)–(4), including correlated noise and taming; implementation/validation tracked separately. Not another name for BAOAB. |
| Annealed Langevin | [R08](#r08) | R08 describes noise-conditioned score annealing. Temperature annealing of a known potential must be labeled separately. |
| Replica exchange Langevin | [R09](#r09), [R19](#r19) | Coupled temperatures and exchanges. Correct swaps do not eliminate bias in unadjusted within-replica discretizations. |
| PIPLA; MYIPLA; PIPULA; PIPGLA | [R10](#r10) | Specified from arXiv v3, Eqs. (7)–(14), Remark 1; implementation/validation tracked separately. Shared parameter learning and latent particles in nondifferentiable latent-variable models. PIPULA is a special case of MYIPLA; generic proximal ULA alone does not cover these methods. |
| SVGD; weight-space SVGD (w-SVGD) | [R11](#r11) | Kernelized deterministic particle transport; finite interacting particles are not independent MCMC chains. |
| Projected SVGD (pSVGD) | [R12](#r12) | Likelihood/data-informed projection and complement treatment; a random projection is not equivalent. |
| Stein variational Newton (SVN) | [R13](#r13) | Function-space second variation and coupled Newton direction; a preconditioned SVGD direction is not automatically SVN. |
| Function-space SVGD (f-SVGD) | [R14](#r14) | Function values, evaluation inputs, and pullback through model Jacobians are part of the formulation. Weight-space kernels alone do not implement f-SVGD. |
| Kernel Stein discrepancy (KSD) | [R15](#r15); application in [R01](#r01), Appendix A | IMQ/RBF Stein kernels, estimator conventions and convergence-detection assumptions must be explicit. A squared-KSD U-statistic can be negative. |
| Repulsive deep ensembles | [R16](#r16) | Bayesian interacting-ensemble construction; kernel/density estimator and weight/function coordinates affect the algorithm. |
| SA-LEnKF | [R17](#r17) | Specified from author manuscript Algorithms 1–2 and supplement Algorithms 3–4; implementation/validation tracked separately. Filtering states and learning unknown parameters are separate tasks. |
| KL-driven Wasserstein–Fisher–Rao (WFR)/birth–death sampling | [R18](#r18) | Specified from arXiv v1, Eqs. (5)–(8), (13)–(14), Algorithm 1; implementation/validation tracked separately. Transport plus reaction/birth–death; finite-particle approximations and normalized weights/ancestry remain explicit. |
| Rank-normalized folded split R-hat | [R21](#r21) | Split chains, rank normalization, and folded scale-sensitive comparison; assess both location and scale. Use multiple suitable MCMC chains, rather than treating interacting particles as independent chains. |
| Raw-mean effective sample size (ESS) | [R22](#r22), pooling context in [R21](#r21) | Pooled autocorrelation estimate with Geyer initial positive and monotone paired sums. This implementation estimates ESS for the raw sample mean; it does not implement rank-normalized bulk ESS or quantile/tail ESS. |
| Online recording mean/variance | [R23](#r23) | Welford updates retain a count, mean, and corrected sum of squares without storing the trajectory. This is a streaming moment calculation, not an online substitute for exact full-chain ESS. |

The explicit “not yet implemented” entry is a literature-gated extension, not a placeholder advertised as a working API. “Specified” means that a source formulation was reviewed, not that executable code or numerical validation is complete. Other rows describe the intended formulation, with current implementation, unavailable methods, and test status maintained in the algorithm register. No blanket implementation or validation claim is made by this table.

## Bibliography

<a id="r01"></a>
**[R01]** Muhammad Izzatullah, Tristan van Leeuwen, and Daniel Peter (2021). “Bayesian seismic inversion: a fast sampling Langevin dynamics Markov chain Monte Carlo method.” *Geophysical Journal International*, **227**(3), 1523–1553. DOI: [10.1093/gji/ggab287](https://doi.org/10.1093/gji/ggab287).

<a id="r02"></a>
**[R02]** Alain Durmus and Eric Moulines (2015, preprint). “Non-asymptotic convergence analysis for the Unadjusted Langevin Algorithm.” *arXiv preprint*, [arXiv:1507.05021](https://arxiv.org/abs/1507.05021). The verified record used here is the preprint; a journal citation is not inferred from title similarity.

<a id="r03"></a>
**[R03]** Mert Gürbüzbalaban, Hoang M. Nguyen, Xicheng Zhang, and Lingjiong Zhu (2025). “Anchored Langevin Algorithms.” *arXiv preprint*, [arXiv:2509.19455](https://arxiv.org/abs/2509.19455).

<a id="r04"></a>
**[R04]** Changwei Tu, Xiaoyu Wang, Yingli Wang, Xicheng Zhang, and Lingjiong Zhu (2026). “Reflected Anchored Langevin Algorithms.” *arXiv preprint*, [arXiv:2610.09522](https://arxiv.org/abs/2610.09522).

<a id="r05"></a>
**[R05]** Benedict Leimkuhler and Charles Matthews (2013; preprint 2012). “Rational Construction of Stochastic Numerical Methods for Molecular Sampling.” *Applied Mathematics Research eXpress*, **2013**(1), 34–56. DOI: [10.1093/amrx/abs010](https://doi.org/10.1093/amrx/abs010). Preprint: [arXiv:1203.5428](https://arxiv.org/abs/1203.5428).

<a id="r06"></a>
**[R06]** Xiang Cheng, Niladri S. Chatterji, Peter L. Bartlett, and Michael I. Jordan (2018; preprint 2017). “Underdamped Langevin MCMC: A non-asymptotic analysis.” *Proceedings of the 31st Conference On Learning Theory*, PMLR **75**, 300–323. [Original proceedings](https://proceedings.mlr.press/v75/cheng18a.html); [arXiv:1707.03663](https://arxiv.org/abs/1707.03663).

<a id="r07"></a>
**[R07]** Sotirios Sabanis and Ying Zhang (2019; preprint 2018). “Higher order Langevin Monte Carlo algorithm.” *Electronic Journal of Statistics*, **13**(2), 3805–3850. DOI: [10.1214/19-EJS1615](https://doi.org/10.1214/19-EJS1615). Preprint: [arXiv:1808.00728](https://arxiv.org/abs/1808.00728).

<a id="r08"></a>
**[R08]** Yang Song and Stefano Ermon (2019). “Generative Modeling by Estimating Gradients of the Data Distribution.” *Advances in Neural Information Processing Systems (NeurIPS 2019)*; venue identified in the authors' arXiv record. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600).

<a id="r09"></a>
**[R09]** Yi Chen, Jinglin Chen, Jing Dong, Jian Peng, and Zhaoran Wang (2020). “Accelerating Nonconvex Learning via Replica Exchange Langevin Diffusion.” *arXiv preprint*, [arXiv:2007.01990](https://arxiv.org/abs/2007.01990).

<a id="r10"></a>
**[R10]** Paula Cordero Encinar, Francesca Romana Crucinio, and Omer Deniz Akyildiz (2025; preprint 2024). “Proximal Interacting Particle Langevin Algorithms.” *Proceedings of the 41st Conference on Uncertainty in Artificial Intelligence*, PMLR **286**, 1220–1265. [Original proceedings](https://proceedings.mlr.press/v286/encinar25a.html); [arXiv:2406.14292](https://arxiv.org/abs/2406.14292). The targeted full-text planning review used arXiv v3; equation numbers should be checked against that version when translating a scheme.

<a id="r11"></a>
**[R11]** Qiang Liu and Dilin Wang (2016). “Stein Variational Gradient Descent: A General Purpose Bayesian Inference Algorithm.” *arXiv preprint*, [arXiv:1608.04471](https://arxiv.org/abs/1608.04471).

<a id="r12"></a>
**[R12]** Peng Chen and Omar Ghattas (2020). “Projected Stein Variational Gradient Descent.” *arXiv preprint*, [arXiv:2002.03469](https://arxiv.org/abs/2002.03469).

<a id="r13"></a>
**[R13]** Gianluca Detommaso, Tiangang Cui, Alessio Spantini, Youssef Marzouk, and Robert Scheichl (2018). “A Stein variational Newton method.” *Advances in Neural Information Processing Systems (NIPS 2018)*; venue recorded in the authors' arXiv journal reference. [arXiv:1806.03085](https://arxiv.org/abs/1806.03085).

<a id="r14"></a>
**[R14]** Ziyu Wang, Tongzheng Ren, Jun Zhu, and Bo Zhang (2019). “Function Space Particle Optimization for Bayesian Neural Networks.” *International Conference on Learning Representations (ICLR 2019)*; arXiv describes the linked preprint as an extended version of the ICLR paper. [arXiv:1902.09754](https://arxiv.org/abs/1902.09754).

<a id="r15"></a>
**[R15]** Jackson Gorham and Lester Mackey (2017). “Measuring Sample Quality with Kernels.” *Proceedings of the 34th International Conference on Machine Learning*, PMLR **70**, 1292–1301. [Original proceedings](https://proceedings.mlr.press/v70/gorham17a.html).

<a id="r16"></a>
**[R16]** Francesco D'Angelo and Vincent Fortuin (2021). “Repulsive Deep Ensembles are Bayesian.” *arXiv preprint*, [arXiv:2106.11642](https://arxiv.org/abs/2106.11642).

<a id="r17"></a>
**[R17]** Tianning Dong, Peiyi Zhang, and Faming Liang (2023; published online 2022). “A Stochastic Approximation-Langevinized Ensemble Kalman Filter Algorithm for State Space Models with Unknown Parameters.” *Journal of Computational and Graphical Statistics*, **32**(2), 448–469. DOI: [10.1080/10618600.2022.2107531](https://doi.org/10.1080/10618600.2022.2107531). Crossref distinguishes online publication on 7 October 2022 from print publication on 3 April 2023. Accessible [author manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC10795543/) and [supplement PDF](https://ndownloader.figshare.com/files/36483124).

<a id="r18"></a>
**[R18]** Yulong Lu, Jianfeng Lu, and James Nolen (2019). “Accelerating Langevin Sampling with Birth-death.” *arXiv preprint*, [arXiv:1905.09863](https://arxiv.org/abs/1905.09863).

<a id="r19"></a>
**[R19]** W. K. Hastings (1970). “Monte Carlo sampling methods using Markov chains and their applications.” *Biometrika*, **57**(1), 97–109. DOI: [10.1093/biomet/57.1.97](https://doi.org/10.1093/biomet/57.1.97). The deposited author name uses initials; an expanded name is not invented here.

<a id="r20"></a>
**[R20]** Gareth O. Roberts and Richard L. Tweedie (1996). “Exponential Convergence of Langevin Distributions and Their Discrete Approximations.” *Bernoulli*, **2**(4), 341–363. DOI: [10.2307/3318418](https://doi.org/10.2307/3318418). Author list, title, year, venue, volume, issue, starting page, and DOI were verified against deposited Crossref metadata; the page range also appears in R01's references. This is the foundational ULA/MALA convergence and ergodicity source, distinct from R19's general MH construction.

<a id="r21"></a>
**[R21]** Aki Vehtari, Andrew Gelman, Daniel Simpson, Bob Carpenter, and Paul-Christian Bürkner (2021). “Rank-Normalization, Folding, and Localization: An Improved R-hat for Assessing Convergence of MCMC (with Discussion).” *Bayesian Analysis*, **16**(2), 667–718. DOI: [10.1214/20-BA1221](https://doi.org/10.1214/20-BA1221). “R-hat” renders the mathematical symbol in the deposited title.

<a id="r22"></a>
**[R22]** Charles J. Geyer (1992). “Practical Markov Chain Monte Carlo.” *Statistical Science*, **7**(4), 473–483. DOI: [10.1214/ss/1177011137](https://doi.org/10.1214/ss/1177011137).

<a id="r23"></a>
**[R23]** B. P. Welford (1962). “Note on a Method for Calculating Corrected Sums of Squares and Products.” *Technometrics*, **4**(3), 419–420. DOI: [10.1080/00401706.1962.10490022](https://doi.org/10.1080/00401706.1962.10490022). The verified deposited author name uses initials.

## What was actually reviewed

Review depth is evidence provenance, not a promise of implementation correctness. The planning review and this documentation pass are distinguished below.

| Sources | Evidence actually examined |
|---|---|
| R01 | Planning: supplied full text, Section 4, Algorithms 1–2, Appendix A, Appendix C. Documentation pass: fresh text extraction; visual inspection of rendered Algorithms 1–2 on p. 1529 and Equation (10) on p. 1531; DOI metadata. |
| R03 | Scientific implementation review: arXiv:2509.19455v1 (23 September 2025), Eqs. (5)–(7), (16), (33)–(34), (41)–(43), Algorithms 1–2 including Monte Carlo Gaussian smoothing. Algorithm 2's zeroth-order estimator uses a symbol inconsistent with Eq. (43); the specification follows Eq. (43) to avoid counting the smooth component twice. |
| R04 | Planning: abstract and selected full-text introduction, drift/diffusion definitions and references. Documentation pass: original arXiv metadata. Full reflected-method derivation remains outstanding. |
| R07 | Scientific implementation review: arXiv:1808.00728v3 (25 September 2019), Eqs. (2)–(4) taming/correlated noise, Eq. (8) untamed scheme, Section 2 assumptions H1–H6 and theorem statements. |
| R10 | Planning: selected method passages. Scientific implementation review: arXiv:2406.14292v3, Eqs. (7)–(14), Remark 1 on joint proximal maps, Section 4 assumptions and Theorems 4.1–4.2. Original PMLR metadata checked here. |
| R11 | Scientific implementation review: arXiv:1608.04471v3 (9 September 2019), SVGD equations and Algorithm 1. |
| R12 | Scientific implementation review: arXiv:2002.03469v2 (10 June 2020), Sections 3.1–3.3, Eqs. (10)–(32), Algorithms 1–2; selected subset uses a whitened isotropic Gaussian prior. |
| R13 | Scientific implementation review: arXiv:1806.03085v2 (29 October 2018), Sections 3–4, Eqs. (11)–(20), Appendices A/B. |
| R14 | Scientific implementation review: arXiv:1902.09754v2 (8 May 2019), Section 3.1, Eqs. (2)–(4), Table 1, Algorithm 1, Appendix E Eqs. (14)–(22). |
| R15 | Scientific implementation review: original PMLR PDF, Proposition 2's Stein kernel Eq. (3) and Theorem 8's IMQ assumptions; R01 Appendix A also examined. |
| R16 | Scientific implementation review: arXiv:2106.11642v3 (28 March 2023), Sections 2–3, Eqs. (3)–(20); KDE-WGD/fWGD selected. |
| R17 | Initial publisher access returned HTTP 403. Subsequently obtained full author-manuscript HTML from PMC10795543: Section 2 derivations/assumptions, Algorithms 1–2 visually read from figures. Figshare supplement (17 July 2022): Algorithms 3–4, Eqs. (S2)–(S5). Main PDF remains challenged, but HTML/images and supplement resolve the method-access gate. Prior-sign discrepancy is a documented project interpretation, not an author erratum. |
| R18 | Scientific implementation review: arXiv:1905.09863v1 (23 May 2019), Eqs. (5)–(8), (13)–(14), Algorithm 1 and stated theorem assumptions. |
| R02, R05, R08, R09 | Planning: original abstracts. Documentation pass: original arXiv metadata. Full scheme review is separate from metadata verification; R05's published venue/DOI comes from its arXiv journal reference. |
| R06, R19, R20 | Added here for underdamped analysis, the MH ratio, and foundational Langevin ergodicity, respectively. Original PMLR/arXiv metadata (R06) and deposited DOI metadata (R19/R20) were checked; their full derivations were not reviewed in this documentation pass. |
| R21–R23 | Documentation pass: author lists, titles, publication years, venues, volumes/issues, page ranges, and DOIs verified by `urllib` against deposited Crossref metadata. Full-text derivations were not audited during this bibliography update; the map states the narrower implemented diagnostic scope. |

The scientific implementation reviews above were performed by the project's method-review agents during this implementation round, using downloaded original full texts and appendices. They establish that the indicated definitions were examined; they are not an exhaustive independent proof audit, implementation completion claim, or statistical validation result.

Exa was unavailable; the user authorized primary-source fallback. A search/errata/author-code review through Exa is still outstanding. DOI metadata verification does not imply access to publisher full text. Abstracts and metadata cannot substitute for method definitions, appendices, assumptions, or implementation validation. No author implementation has been copied or license-audited as part of this bibliography.
