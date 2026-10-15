"""Bounded cross-family accuracy versus actual cost, with independent references.

Run: python benchmarks/families.py --output benchmarks/results/cpu
Posterior, empirical-particle, parameter-learning, diagnostic and filtering
objectives are explicitly distinct. Seed SD is replication variability, not a
confidence interval for asymptotic correctness. Requires the optional matplotlib
extra only for figures. No silent CUDA fallback is permitted.
"""
from pathlib import Path
import argparse
import json
import math
import os
import platform
import time
import torch
try:
    from .accuracy import (CountedPotential, chain_diagnostics, inapplicable_diagnostics,
                           strict_report, synchronize)
except ImportError:
    from accuracy import (CountedPotential, chain_diagnostics, inapplicable_diagnostics,
                          strict_report, synchronize)
from lafa.problems import Gaussian, LatentGaussianLaplace, LinearGaussianStateSpace
from lafa.samplers import ULA, MALA, BAOAB
from lafa.specialized import (HOLA, AnchoredLangevin, MYIPLA, PIPGLA, PIPULA,
                              BirthDeathLangevin, SALEnKF)
from lafa.particles import SVGD, RepulsiveEnsemble
from lafa.kernels import RBF, IMQ
from lafa.diagnostics import ksd


FAMILIES = ('overdamped', 'inertial_higher_order', 'specialized', 'stein_particles',
            'diagnostics', 'ensembles_hybrid')


def generator(device, seed): return torch.Generator(device=device).manual_seed(seed)


def moments(samples, reference):
    mean = samples.mean(0)
    centered = samples-mean
    covariance = centered.T@centered/len(samples)
    return dict(mean=mean, covariance=covariance,
                mean_l2_error=torch.linalg.vector_norm(mean-reference.mean),
                covariance_frobenius_error=torch.linalg.matrix_norm(covariance-reference.covariance))


class CountedProx:
    def __init__(self, prox): self.prox, self.batches, self.points = prox, 0, 0
    def __call__(self, theta, particles, step):
        self.batches += 1
        self.points += len(particles)
        return self.prox(theta, particles, step)


class GaussianDerivatives:
    """Exact analytic derivative provider; each invoked operation is counted."""
    def __init__(self, problem, counts):
        self.problem, self.counts = problem, counts
        self.precision = torch.cholesky_inverse(problem.cholesky)
        self.norm = torch.linalg.eigvalsh(self.precision).abs().max()
    def gradient(self, x):
        self.counts['gradient_batches'] += 1
        self.counts['gradient_points'] += 1
        return self.precision@(x-self.problem.mean)
    def hvp(self, x, v):
        self.counts['hvp_calls'] += 1
        return self.precision@v
    def grad_laplacian(self, x):
        self.counts['grad_laplacian_calls'] = self.counts.get('grad_laplacian_calls', 0)+1
        return torch.zeros_like(x)
    def spectral_norm(self, x):
        self.counts['spectral_norm_calls'] = self.counts.get('spectral_norm_calls', 0)+1
        return self.norm


def posterior_run(problem, method, family, h, steps, args, seed):
    rng = generator(args.device, seed)
    # Deliberately overdispersed independent initializations expose slow mixing.
    initial = 3*torch.randn((args.chains,2),device=args.device,dtype=args.dtype,generator=rng)
    counted = CountedPotential(problem.potential)
    synchronize(args.device); started = time.perf_counter()
    if method == 'HOLA':
        obj = HOLA(counted,initial,h,tamed=False,derivatives=GaussianDerivatives(problem,counted.counts),generator=rng)
    elif method == 'AnchoredLangevin':
        obj = AnchoredLangevin(counted,counted,initial,h,generator=rng)
    else:
        obj = dict(ULA=ULA,MALA=MALA,BAOAB=BAOAB)[method](counted,initial,h,generator=rng)
    samples, accepted, attempts = [], 0, 0
    for i in range(args.warmup+steps):
        result = obj.step()
        if i >= args.warmup:
            if getattr(result,'accepted',None) is not None:
                accepted += int(result.accepted.sum()); attempts += args.chains
            if (i-args.warmup)%args.record_every == 0: samples.append(result.position)
    synchronize(args.device); seconds = time.perf_counter()-started
    work = counted.snapshot() # immutable, before any diagnostic target evaluation
    chains = torch.stack(samples,dim=1)
    stats = moments(chains.reshape(-1,2),problem)
    diagnostic_started = time.perf_counter()
    diagnostics = chain_diagnostics(chains)
    synchronize(args.device)
    diagnostic_seconds = time.perf_counter()-diagnostic_started
    return dict(family=family,method=method,objective='posterior_moments',seed=seed,
        budget=dict(steps=steps,warmup=args.warmup,step_size=h,chains=args.chains,record_every=args.record_every),
        seconds=seconds,diagnostic_seconds=diagnostic_seconds,work=work,diagnostics=diagnostics,metrics=stats,
        primary_error=stats['covariance_frobenius_error'],primary_error_name='covariance_frobenius_error',
        acceptance_rate=accepted/attempts if attempts else None,
        acceptance_reason=None if attempts else 'No Metropolis accept/reject step for this method.',
        reference='analytic Gaussian moments',
        limitation='Finite warmup/mixing and integrator bias remain; HOLA uses an exact quadratic derivative provider; anchored reference equals target.')


def empirical_run(problem, method, family, n, steps, args, seed):
    rng = generator(args.device,seed)
    initial = 3*torch.randn((n,2),device=args.device,dtype=args.dtype,generator=rng)
    counted = CountedPotential(problem.potential)
    synchronize(args.device); started = time.perf_counter()
    if method == 'BirthDeathLangevin':
        obj = BirthDeathLangevin(counted,initial,.025,bandwidth=.7,generator=rng)
    else:
        obj = dict(SVGD=SVGD,RepulsiveEnsemble=RepulsiveEnsemble)[method](counted,initial,step_size=.05,kernel=RBF(.7),block_size=32)
    for _ in range(steps): obj.step()
    synchronize(args.device); seconds = time.perf_counter()-started
    work = counted.snapshot()
    # Each executed pair sweep visits all ordered pairs. Birth/death recomputes
    # N KDE rows for each of the N event candidates, including no-event cases.
    work['ordered_pair_evaluations'] = steps*n*n*(n if method=='BirthDeathLangevin' else 1)
    particles = obj.position
    stats = moments(particles,problem)
    stats['imq_ksd_v_squared'] = ksd(particles,problem.score(particles),kernel=IMQ())
    return dict(family=family,method=method,objective='final_empirical_distribution',seed=seed,
        budget=dict(steps=steps,particles=n,step_size=.025 if method=='BirthDeathLangevin' else .05,bandwidth=.7),
        seconds=seconds,work=work,metrics=stats,
        diagnostics=inapplicable_diagnostics('Interacting particles are dependent; no independent-chain ESS/Rhat/MCSE is assigned.'),
        primary_error=stats['imq_ksd_v_squared'],primary_error_name='imq_ksd_v_squared',
        reference='analytic Gaussian score and moments',
        pair_count_semantics='Exact ordered pairs from executed loops: steps*N^2, or steps*N^3 for birth/death KDE sweeps.',
        limitation='Finite population, fixed kernel bandwidth and discretization give approximate empirical distributions; no stationary posterior guarantee.')


def learning_reference(problem):
    # Convex negative integrated likelihood: bracket derivative then bisect.
    def gradient(theta):
        t = problem.observations.new_tensor([theta],requires_grad=True)
        return float(torch.autograd.grad(-problem.log_marginal_likelihood(t),t)[0])
    lo, hi = float(problem.observations.min()-8),float(problem.observations.max()+8)
    if not gradient(lo)<0<gradient(hi): raise RuntimeError('Marginal optimum not bracketed')
    for _ in range(80):
        mid = (lo+hi)/2
        if gradient(mid)>0: hi=mid
        else: lo=mid
    optimum = problem.observations.new_tensor([(lo+hi)/2])
    return dict(theta=optimum,negative_log_marginal=-problem.log_marginal_likelihood(optimum),
                optimum_bracket_width=hi-lo,objective='negative exact integrated Normal/Laplace likelihood; no theta regularizer')


def learning_run(problem, reference, method, n, steps, args, seed):
    rng = generator(args.device,seed)
    initial = torch.zeros((n,problem.dimension),device=args.device,dtype=args.dtype)
    theta = initial.new_tensor([-1.5])
    counted = CountedPotential(problem.smooth_potential)
    prox = CountedProx(problem.full_joint_prox if method=='PIPULA' else problem.prox_joint)
    synchronize(args.device); started = time.perf_counter()
    if method == 'PIPULA': obj = PIPULA(prox,theta,initial,.02,generator=rng)
    else: obj = dict(MYIPLA=MYIPLA,PIPGLA=PIPGLA)[method](counted,prox,theta,initial,.02,smoothing=.02,generator=rng)
    values = []
    for i in range(args.warmup+steps):
        result = obj.step()
        if i>=args.warmup: values.append(result.theta)
    synchronize(args.device); seconds = time.perf_counter()-started
    work = counted.snapshot(); work.update(prox_batches=prox.batches,prox_points=prox.points)
    estimate = torch.stack(values).mean(0)
    gap = -problem.log_marginal_likelihood(estimate)-reference['negative_log_marginal']
    return dict(family='specialized',method=method,objective='shared_parameter_learning',seed=seed,
        budget=dict(steps=steps,warmup=args.warmup,particles=n,step_size=.02,smoothing=.02),
        seconds=seconds,work=work,
        metrics=dict(theta_time_average=estimate,theta_final=result.theta,
                     theta_absolute_error=(estimate-reference['theta']).abs().squeeze(),negative_log_marginal_gap=gap),
        primary_error=gap,primary_error_name='negative_log_marginal_gap',
        diagnostics=inapplicable_diagnostics('One shared learned theta and interacting latent particles are not posterior chains.'),
        reference='exact integrated Normal/Laplace likelihood, convex scalar derivative bisection',
        limitation='Finite powered-marginal ensemble, smoothing, step bias and transient averaging; theta estimates are not Bayesian posterior draws.')


def diagnostic_run(problem, n, shifted, args, seed):
    rng = generator(args.device,seed)
    synchronize(args.device); started = time.perf_counter()
    samples = problem.sample(n,generator=rng)
    if shifted: samples = samples+samples.new_tensor([1.5,0.])
    score = problem.score(samples)
    v = ksd(samples,score,kernel=IMQ(),estimator='v')
    u = ksd(samples,score,kernel=IMQ(),estimator='u')
    synchronize(args.device); seconds = time.perf_counter()-started
    return dict(family='diagnostics',method='IMQ KSD '+('shifted' if shifted else 'direct iid'),
        objective='diagnostic_discrepancy',seed=seed,budget=dict(direct_draws=n,shift=[1.5,0.] if shifted else [0.,0.]),
        seconds=seconds,work=dict(target_batches=0,target_points=0,gradient_batches=0,gradient_points=0,
            analytic_score_batches=1,analytic_score_points=n,hvp_calls=0,prox_batches=0,prox_points=0,
            ordered_pair_evaluations=2*n*n),
        metrics=dict(imq_ksd_v_squared=v,imq_ksd_u_squared=u),
        primary_error=v,primary_error_name='imq_ksd_v_squared',
        diagnostics=inapplicable_diagnostics('Diagnostic of iid empirical reference draws, not a Markov transition or mixing estimate.'),
        reference='exact independent Gaussian draws; shifted alternative has a known incorrect mean',
        limitation='U may be negative and is unbiased only under iid fixed masses; V includes diagonal finite-N error; KSD alone does not establish convergence.')


class CountedTransition:
    def __init__(self, matrix): self.matrix,self.calls,self.points=matrix,0,0
    def __call__(self, x, theta):
        self.calls += 1; self.points += len(x)
        return x@self.matrix.T


def filtering_run(model, observations, reference, n, inner_steps, args, seed):
    rng = generator(args.device,seed)
    initial = model.initial_mean+torch.randn((n,model.dimension),device=args.device,dtype=args.dtype,generator=rng)@torch.linalg.cholesky(model.initial_covariance).T
    counted = CountedPotential(lambda x: .5*torch.linalg.solve_triangular(torch.linalg.cholesky(model.initial_covariance),(x-model.initial_mean).T,upper=False).square().sum(0))
    transition = CountedTransition(model.F)
    synchronize(args.device); started = time.perf_counter()
    obj = SALEnKF(transition,model.Q,model.H,model.R,counted,initial,initial.new_zeros(1),
                 epsilon=.06,gamma=0.,inner_steps=inner_steps,burnin=inner_steps//2,generator=rng)
    means,covariances=[],[]
    for y in observations:
        result = obj.step(y); means.append(result.mean); covariances.append(result.covariance)
    synchronize(args.device); seconds = time.perf_counter()-started
    work = counted.snapshot(); work.update(transition_batches=transition.calls,transition_points=transition.points,
        conditional_prior_analytic_gradient_points=(len(observations)-1)*inner_steps*n)
    mean_rmse = (torch.stack(means)-reference.means).square().mean().sqrt()
    covariance_rmse = (torch.stack(covariances)-reference.covariances).square().mean().sqrt()
    return dict(family='ensembles_hybrid',method='SALEnKF known parameter',objective='sequential_filtering',seed=seed,
        budget=dict(stages=len(observations),particles=n,inner_steps=inner_steps,burnin=inner_steps//2,epsilon=.06,gamma=0.),
        seconds=seconds,work=work,metrics=dict(mean_rmse=mean_rmse,covariance_rmse=covariance_rmse,means=torch.stack(means),covariances=torch.stack(covariances)),
        primary_error=mean_rmse,primary_error_name='filter_mean_rmse',
        diagnostics=inapplicable_diagnostics('Finite filtering pools, ancestor resampling and changing stage targets preclude stationary independent-chain diagnostics.'),
        reference='independent analytic Kalman filter with known fixed parameters',
        limitation='Fixed synthetic observation series; replication uncertainty is algorithmic only. Short inner loops, epsilon and finite ancestor pools bias filtering.')


def summaries(runs):
    groups = {}
    for row in runs:
        key = (row['family'],row['method'],json.dumps(row['budget'],sort_keys=True))
        groups.setdefault(key,[]).append(row)
    result = []
    for (family,method,budget),rows in groups.items():
        errors=torch.tensor([float(r['primary_error']) for r in rows],dtype=torch.float64)
        seconds=torch.tensor([r['seconds'] for r in rows],dtype=torch.float64)
        sd = errors.std().item() if len(rows)>1 else None
        result.append(dict(family=family,method=method,budget=json.loads(budget),seeds=[r['seed'] for r in rows],
            repetitions=len(rows),primary_error_name=rows[0]['primary_error_name'],error_mean=errors.mean().item(),
            error_seed_sd=sd,error_seed_standard_error=sd/math.sqrt(len(rows)) if sd is not None else None,
            uncertainty_reason=None if sd is not None else 'At least two independent seeds required for seed uncertainty.',
            seconds_mean=seconds.mean().item(),seconds_seed_sd=seconds.std().item() if len(rows)>1 else None))
    return result


def figure(data, output):
    os.environ.setdefault('MPLCONFIGDIR','/tmp/lafa-matplotlib-cache')
    os.environ.setdefault('XDG_CACHE_HOME','/tmp/lafa-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    objectives=('posterior_moments','final_empirical_distribution','shared_parameter_learning','diagnostic_discrepancy','sequential_filtering')
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    objectives_by_method={r['method']:r['objective'] for r in data['runs']}
    for ax,objective in zip(axes.flat,objectives):
        selected=[r for r in data['summaries'] if objectives_by_method[r['method']]==objective]
        for method in sorted({r['method'] for r in selected}):
            rows=sorted((r for r in selected if r['method']==method),key=lambda r:r['seconds_mean'])
            # Points are actual budget configurations; connecting lines guide the eye.
            ax.errorbar([r['seconds_mean'] for r in rows],[r['error_mean'] for r in rows],
                xerr=[r['seconds_seed_sd'] or 0 for r in rows],yerr=[r['error_seed_sd'] or 0 for r in rows],
                fmt='o-',markersize=3,capsize=2,label=method)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_title(objective.replace('_',' '))
        ax.set_xlabel('measured seconds (mean ± seed SD)')
        ax.set_ylabel(selected[0]['primary_error_name']+' (± seed SD)'); ax.legend(fontsize=6)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,1,'Six families; five distinct objectives\n\n3 seeds by default; bars show seed SD.\nCosts exclude independent reference solving\nand post-run diagnostics, except KSD itself.\nMethod configuration differs across points.\nFinite budgets do not certify convergence.\nSee JSON for chain diagnostics and work counts.',va='top',fontsize=11)
    fig.savefig(output/'families_accuracy_cost.png',dpi=150); plt.close(fig)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=Path('benchmarks/results/cpu'))
    p.add_argument('--device',default='cpu')
    p.add_argument('--seeds',type=int,nargs='+',default=[7,19,31])
    p.add_argument('--steps',type=int,nargs='+',default=[240,720])
    p.add_argument('--step-sizes',type=float,nargs='+',default=[.04,.12])
    p.add_argument('--particles',type=int,nargs='+',default=[16,32])
    p.add_argument('--warmup',type=int,default=160)
    p.add_argument('--chains',type=int,default=16)
    p.add_argument('--record-every',type=int,default=2)
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--no-figures',action='store_true')
    args=p.parse_args(argv)
    args.device=torch.device(args.device); args.dtype=torch.float64
    if args.device.type=='cuda' and not torch.cuda.is_available(): p.error('CUDA requested but unavailable; no fallback result is produced')
    if min(args.steps)<1 or min(args.particles)<2 or min(args.step_sizes)<=0 or not all(math.isfinite(h) for h in args.step_sizes) or min(args.chains,args.record_every,args.threads)<1 or args.warmup<0: p.error('positive finite budgets, particles >=2 and nonnegative warmup required')
    if len(set(args.seeds))!=len(args.seeds): p.error('replication seeds must be distinct')
    torch.set_num_threads(args.threads)
    args.output.mkdir(parents=True,exist_ok=True)
    common=dict(device=args.device,dtype=args.dtype)
    problem=Gaussian([.6,-.4],[[1.,.3],[.3,.7]],**common)
    latent=LatentGaussianLaplace(observations=[-.2,.7,1.1,1.4],**common)
    learned=learning_reference(latent)
    model=LinearGaussianStateSpace(**common)
    _,observations=model.simulate(3,seed=410)
    kalman=model.kalman_filter(observations)
    data=dict(schema_version=1,metadata=dict(device=str(args.device),dtype=str(args.dtype),torch_version=torch.__version__,python_version=platform.python_version(),threads=args.threads,cuda_executed=args.device.type=='cuda',seeds=args.seeds,
        families=list(FAMILIES),timing='Synchronized construction and algorithm execution; excludes reference solving, post-run moments and chain diagnostics. KSD timing includes direct draws, analytic scores and both pair estimators.',
        work_count_semantics='Observed target/gradient/prox/HVP callback counts; points are evaluated rows. Pair counts follow fully executed exact pair loops. Analytic derivative work is separately named; counts are not equivalent FLOPs.',
        uncertainty='Across independent algorithm seeds on fixed targets and fixed filtering observations; SD and standard error are descriptive finite-repetition quantities.',
        limitations='Small low-dimensional fixtures and finite budgets; no cross-objective ranking, full family coverage claim for every registered variant, GPU result, or convergence guarantee.'),
        references=dict(gaussian=dict(mean=problem.mean,covariance=problem.covariance),learning=learned,
                        filtering=dict(observations=observations,means=kalman.means,covariances=kalman.covariances,data_seed=410)),runs=[])
    started=time.perf_counter()
    posterior_methods=[('ULA','overdamped'),('MALA','overdamped'),('BAOAB','inertial_higher_order'),('HOLA','inertial_higher_order'),('AnchoredLangevin','specialized')]
    for seed in args.seeds:
        for steps in args.steps:
            for h in args.step_sizes:
                for method,family in posterior_methods:
                    data['runs'].append(posterior_run(problem,method,family,h,steps,args,seed))
        for index,n in enumerate(args.particles):
            steps=args.steps[min(index,len(args.steps)-1)]
            for method,family in [('SVGD','stein_particles'),('RepulsiveEnsemble','ensembles_hybrid'),('BirthDeathLangevin','ensembles_hybrid')]:
                data['runs'].append(empirical_run(problem,method,family,n,min(steps,180),args,seed))
            for method in ('MYIPLA','PIPGLA','PIPULA'):
                data['runs'].append(learning_run(latent,learned,method,n,steps,args,seed))
            for shifted in (False,True): data['runs'].append(diagnostic_run(problem,n*4,shifted,args,seed))
            data['runs'].append(filtering_run(model,observations,kalman,n,40+40*index,args,seed))
        print(json.dumps(dict(completed_seed=seed,runs=len(data['runs']),seconds=time.perf_counter()-started)),flush=True)
    data['summaries']=summaries(data['runs']); data['metadata']['driver_seconds']=time.perf_counter()-started
    (args.output/'families.json').write_text(json.dumps(strict_report(data),indent=2,allow_nan=False)+'\n')
    if not args.no_figures: figure(data,args.output)
    print(json.dumps(dict(output=str(args.output/'families.json'),runs=len(data['runs']),seconds=data['metadata']['driver_seconds'])))
    return data


if __name__=='__main__': main()
