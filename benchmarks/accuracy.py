"""Truth-based accuracy experiments; run from an installed lafa checkout.

python benchmarks/accuracy.py --output benchmark-results/accuracy --device cpu

JSON records actual run budgets, seeds, costs, moments, coordinate credible
intervals and mode occupancy. Correlated production samples are not called iid;
diagnostics retain independent chain/time axes and estimate finite-run uncertainty.
Figures require optional matplotlib; all target and reference calculations use
PyTorch only. The nonlinear two-coefficient model retains a full 2D spatial map.
"""
from pathlib import Path
import argparse
import json
import math
import os
import platform
import time
import torch
from lafa.problems import paper_gaussian, GaussianMixture2D, Rosenbrock2D, NonlinearTomography2D, quadrature_2d
from lafa.samplers import ULA, MALA


class CountedPotential:
    """Count actual callback invocations and independently evaluated rows.

    Gradient counts cover callbacks executed with differentiable tensor inputs;
    they count target gradients requested by these drivers, not FLOPs. Snapshot
    before postprocessing so reporting cannot contaminate sampling work.
    """
    def __init__(self, potential):
        self.potential = potential
        self.counts = dict(target_batches=0, target_points=0, gradient_batches=0,
                           gradient_points=0, hvp_calls=0, prox_batches=0, prox_points=0)

    @property
    def batches(self): return self.counts['target_batches']

    @property
    def chains(self): return self.counts['target_points']

    def __call__(self, *args):
        x = args[-1]
        points = math.prod(x.shape[:-1])
        self.counts['target_batches'] += 1
        self.counts['target_points'] += points
        if torch.is_grad_enabled() and any(t.requires_grad for t in args if isinstance(t, torch.Tensor)):
            self.counts['gradient_batches'] += 1
            self.counts['gradient_points'] += points
        return self.potential(*args)

    def snapshot(self): return dict(self.counts)


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def serializable(value):
    if isinstance(value, torch.Tensor): return serializable(value.detach().cpu().tolist())
    if isinstance(value, dict): return {k: serializable(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def strict_report(value):
    """Strict JSON with per-path explanations for nonfinite/undefined values."""
    undefined = {}
    def walk(item, path):
        if isinstance(item, torch.Tensor): item = item.detach().cpu().tolist()
        if isinstance(item, dict): return {k: walk(v, f'{path}.{k}') for k,v in item.items()}
        if isinstance(item, (tuple,list)): return [walk(v, f'{path}[{i}]') for i,v in enumerate(item)]
        if isinstance(item, float) and not math.isfinite(item):
            undefined[path] = 'Undefined/nonfinite finite-chain estimate (constant, degenerate, or insufficiently resolved draws); no convergence value is asserted.'
            return None
        return item
    result = walk(value, '$')
    if undefined: result['undefined_values'] = undefined
    return result


def chain_diagnostics(chains):
    """Always pass (independent chains, retained draws, dimensions) unchanged."""
    from lafa.diagnostics import (bulk_effective_sample_size, tail_effective_sample_size,
        mcse_mean, mcse_quantile, rank_normalized_split_rhat)
    methods = dict(bulk_ess=bulk_effective_sample_size, tail_ess=tail_effective_sample_size,
                   mean_mcse=mcse_mean, median_mcse=mcse_quantile,
                   rank_normalized_split_rhat=rank_normalized_split_rhat)
    values, reasons = {}, {}
    for name, method in methods.items():
        try:
            values[name] = method(chains)
            if not torch.isfinite(values[name]).all():
                reasons[name] = 'Undefined coordinates: constant, degenerate, or unresolved finite-chain diagnostic.'
        except ValueError as exc:
            values[name], reasons[name] = None, str(exc)
    return dict(applicable=True, input_shape=list(chains.shape), values=values,
                undefined_reasons=reasons,
                assumptions='Independent stationary chains required; finite-run estimates do not establish stationarity or convergence.')


def inapplicable_diagnostics(reason):
    return dict(applicable=False, input_shape=None, values=None, undefined_reasons=dict(all=reason))


def experiment(problem,cls,step_size,args,seed,reference):
    generator=torch.Generator(device=args.device).manual_seed(seed)
    initial=torch.zeros(args.chains,2,device=args.device,dtype=args.dtype)
    counted=CountedPotential(problem.potential)
    synchronize(args.device)
    started=time.perf_counter()
    sampler=cls(counted,initial,step_size=step_size,generator=generator)
    samples=[]
    accepted=initial.new_zeros(())
    attempts=0
    for i in range(args.warmup+args.iterations):
        result=sampler.step()
        if i>=args.warmup:
            if result.accepted is not None:
                accepted+=result.accepted.sum()
                attempts+=args.chains
            if (i-args.warmup)%args.record_every==0: samples.append(result.position)
    synchronize(args.device)
    seconds=time.perf_counter()-started
    work=counted.snapshot()
    chain_samples=torch.stack(samples,dim=1)
    diagnostic_started=time.perf_counter()
    diagnostics=chain_diagnostics(chain_samples)
    synchronize(args.device)
    diagnostic_seconds=time.perf_counter()-diagnostic_started
    samples=chain_samples.reshape(-1,initial.shape[-1])
    mean=samples.mean(0)
    centered=samples-mean
    cov=centered.T@centered/len(samples)
    intervals=torch.quantile(samples,torch.tensor([.025,.975],device=args.device,dtype=args.dtype),dim=0)
    report=dict(seed=seed,algorithm=cls.__name__,step_size=step_size,iterations=args.iterations,warmup=args.warmup,
        chains=args.chains,record_every=args.record_every,retained_states=len(samples),seconds=seconds,
        work=work,diagnostics=diagnostics,diagnostic_seconds=diagnostic_seconds,
        gradient_batch_evaluations=work['gradient_batches'],gradient_chain_evaluations=work['gradient_points'],
        gradient_count_semantics='one autodiff evaluation per call; each call contains chains independent gradients',
        sample_mean=mean,sample_covariance=cov,mean_l2_error=torch.linalg.vector_norm(mean-reference.mean),
        covariance_frobenius_error=torch.linalg.matrix_norm(cov-reference.covariance),
        coordinate_central_95_interval=intervals,
        acceptance_rate=(accepted/attempts if attempts else None))
    if isinstance(problem,GaussianMixture2D):
        nearest=(samples[:,None,:]-problem.means).square().sum(-1).argmin(-1)
        occupancy=torch.bincount(nearest,minlength=len(problem.weights))/len(samples)
        report.update(nearest_mode_occupancy=occupancy,reference_weights=problem.weights,visited_modes=(occupancy>0).sum())
    if hasattr(problem,'spatial'):
        fields=problem.field(samples[:4096]).reshape(min(len(samples),4096),-1)
        field_intervals=torch.quantile(fields,torch.tensor([.025,.975],device=args.device,dtype=args.dtype),dim=0)
        truth=problem.field(problem.truth).reshape(-1)
        report.update(spatial_truth_coverage=((truth>=field_intervals[0])&(truth<=field_intervals[1])).to(args.dtype).mean(),
                      spatial_field_mean=fields.mean(0).reshape(problem.grid_shape),
                      spatial_field_std=fields.std(0).reshape(problem.grid_shape))
    return report,samples[:2000].detach().cpu()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('benchmark-results/accuracy'))
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--dtype',choices=['float32','float64'],default='float64')
    parser.add_argument('--seeds',nargs='+',type=int,default=[7,19,31])
    parser.add_argument('--iterations',type=int,default=600)
    parser.add_argument('--warmup',type=int,default=200)
    parser.add_argument('--chains',type=int,default=64)
    parser.add_argument('--record-every',type=int,default=5)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args(argv)
    if min(args.iterations,args.chains,args.record_every,args.threads)<1 or args.warmup<0: parser.error('positive budgets and nonnegative warmup required')
    torch.set_num_threads(args.threads)
    args.device=torch.device(args.device)
    args.dtype=getattr(torch,args.dtype)
    if args.device.type=='cuda' and not torch.cuda.is_available(): parser.error('CUDA requested but unavailable; no GPU result can be produced')
    args.output.mkdir(parents=True,exist_ok=True)
    common=dict(device=args.device,dtype=args.dtype)
    nonlinear=NonlinearTomography2D(grid_shape=(4,4),seed=4,**common).reduced()
    q=quadrature_2d(nonlinear,((-2.,2.),(-2.,2.)),resolution=160)
    q2=quadrature_2d(nonlinear,((-2.,2.),(-2.,2.)),resolution=320)
    qwide=quadrature_2d(nonlinear,((-3.,3.),(-3.,3.)),resolution=480)
    targets=[('paper_gaussian',paper_gaussian(**common),.1),('mixture',GaussianMixture2D(**common),.1),
             ('paper_rosenbrock',Rosenbrock2D(**common),.002),('nonlinear_spatial_reduced',nonlinear,.003)]
    data=dict(metadata=dict(torch_version=torch.__version__,python_version=platform.python_version(),
        device=str(args.device),dtype=str(args.dtype),cpu=platform.processor(),platform=platform.platform(),threads=args.threads,
        gpu=torch.cuda.get_device_name(args.device) if args.device.type=='cuda' else None,
        cuda_runtime=torch.version.cuda,cuda_executed=args.device.type=='cuda',
        timing='seconds measures sampler construction and transitions only; diagnostic_seconds is measured separately; reference and moment postprocessing excluded.',
        notes='Finite-budget independent-chain diagnostics retain chain/time axes. No convergence guarantee. Mode occupancy uses nearest component.'),
        references=dict(nonlinear_spatial_reduced=dict(mean=q2.mean,covariance=q2.covariance,log_normalizer=q2.log_normalizer,
            bounds=q2.bounds,resolution=q2.resolution,boundary_mass=q2.boundary_mass,
            mean_change_on_doubling=torch.linalg.vector_norm(q2.mean-q.mean),
            covariance_change_on_doubling=torch.linalg.matrix_norm(q2.covariance-q.covariance),
            mean_change_on_expanding_bounds=torch.linalg.vector_norm(qwide.mean-q2.mean),
            covariance_change_on_expanding_bounds=torch.linalg.matrix_norm(qwide.covariance-q2.covariance))),runs=[])
    visuals=[]
    for name,problem,h in targets:
        reference=q2 if name=='nonlinear_spatial_reduced' else problem
        data['references'].setdefault(name,dict(mean=reference.mean,covariance=reference.covariance))
        for cls in (ULA,MALA):
            for seed in args.seeds:
                report,samples=experiment(problem,cls,h,args,seed,reference)
                report['target']=name
                data['runs'].append(report)
                if seed==args.seeds[0]: visuals.append((name,cls.__name__,samples))
    with (args.output/'accuracy.json').open('w') as f: json.dump(strict_report(data),f,indent=2,allow_nan=False)
    if not args.no_figures:
        os.environ.setdefault('MPLCONFIGDIR','/tmp/lafa-matplotlib-cache')
        os.environ.setdefault('XDG_CACHE_HOME','/tmp/lafa-cache')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
        for ax,(name,algorithm,samples) in zip(axes.T.reshape(-1),visuals):
            ax.scatter(samples[:,0],samples[:,1],s=2,alpha=.25)
            ax.set_title(f'{name}\n{algorithm}, seed {args.seeds[0]}'); ax.set_xlabel('coordinate 1'); ax.set_ylabel('coordinate 2')
        fig.savefig(args.output/'samples.png',dpi=140); plt.close(fig)
        fig,axes=plt.subplots(1,4,figsize=(13,3),constrained_layout=True)
        nonlinear_run=next(r for r in data['runs'] if r['target']=='nonlinear_spatial_reduced' and r['algorithm']=='MALA')
        truth=nonlinear.field(nonlinear.truth).cpu()
        field_mean=nonlinear_run['spatial_field_mean'].cpu()
        common_limits=dict(vmin=min(truth.min().item(),field_mean.min().item()),vmax=max(truth.max().item(),field_mean.max().item()))
        fields=[truth,nonlinear.spatial.basis[:,0].reshape(nonlinear.grid_shape).cpu(),field_mean,nonlinear_run['spatial_field_std'].cpu()]
        for i,(ax,field,title) in enumerate(zip(axes,fields,['Physical truth','First smooth basis','MALA field mean','MALA field SD'])):
            limits=common_limits if i in (0,2) else dict(vmin=0.,vmax=field.max().item())
            im=ax.imshow(field,origin='lower',extent=(0,1,0,1),**limits); ax.set_title(title); fig.colorbar(im,ax=ax)
        fig.savefig(args.output/'spatial_fields.png',dpi=140); plt.close(fig)
    print(json.dumps(dict(output=str(args.output/'accuracy.json'),runs=len(data['runs']),device=str(args.device),cuda_executed=args.device.type=='cuda')))
    return data


if __name__=='__main__': main()
