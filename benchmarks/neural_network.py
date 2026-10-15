"""Full-data Bayesian NN target, batch/VJP scaling, and independent MALA example.

Run with an installed lafa checkout. CPU RSS is a process high-water mark, not
per-configuration activation memory; CUDA requests never silently fall back.
"""
import argparse
import json
import math
import platform
from pathlib import Path
import resource
import time

import torch
from lafa.targets import ModulePotential
from lafa.particles import FunctionSpaceSVGD
from lafa.samplers import MALA


def network(width, device):
    return torch.nn.Sequential(torch.nn.Linear(1, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, 1)).to(device=device, dtype=torch.float64).eval()


def direct_prediction(theta, inputs, width):
    """Independent explicit formula; no ModulePotential/functional_call oracle."""
    hidden = (inputs*theta[:width] + theta[width:2*width]).tanh()
    return hidden@theta[2*width:3*width] + theta[-1]


class Posterior:
    """U = sum_n (y_n-f_w(x_n))²/(2 sigma²) + ||w||²/(2 prior_sd²).

    All observations enter every deterministic evaluation. Gaussian constants
    independent of w are omitted. No minibatches or stochastic model layers.
    """
    def __init__(self, model, inputs, observations, *, noise_sd=.3, prior_sd=.7,
                 batch_mode='vmap', chunk_size=None):
        self.target = ModulePotential(model, lambda out: ((out.squeeze(-1)-observations)/noise_sd).square().sum()/2,
                                      args=(inputs,), batch_mode=batch_mode, chunk_size=chunk_size)
        self.prior_sd = prior_sd
        self.evaluated_states = 0

    def __call__(self, x):
        return self.target(x) + x.square().sum(-1)/(2*self.prior_sd**2)

    def value_and_grad(self, x):
        value, gradient = self.target.value_and_grad(x)
        self.evaluated_states += 1 if x.ndim == 1 else len(x)
        owned = x.detach()
        return value + owned.square().sum(-1)/(2*self.prior_sd**2), gradient + owned/self.prior_sd**2


def synchronize(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)


def measure(callback, repetitions, device):
    synchronize(device)
    start = time.perf_counter()
    callback()
    synchronize(device)
    warmup = time.perf_counter()-start
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repetitions): callback()
    synchronize(device)
    return dict(warmup_seconds=warmup, timed_repetitions=repetitions,
                seconds_per_evaluation=(time.perf_counter()-start)/repetitions,
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)


def scaling(args, device):
    rows = []
    for width in args.widths:
        for data_count in args.data_sizes:
            inputs = torch.linspace(-2, 2, data_count, device=device, dtype=torch.float64)[:,None]
            observations = inputs[:,0].sin()
            model = network(width, device)
            for chains in args.chains:
                x = .7*torch.randn(chains, 3*width+1, device=device, dtype=torch.float64)
                reference_values, reference_gradients = [], []
                for row in x:
                    leaf = row.detach().clone().requires_grad_()
                    value = ((direct_prediction(leaf, inputs, width)-observations)/.3).square().sum()/2 + leaf.square().sum()/(2*.7**2)
                    reference_values.append(value.detach())
                    reference_gradients.append(torch.autograd.grad(value,leaf)[0])
                reference_values = torch.stack(reference_values)
                reference_gradients = torch.stack(reference_gradients)
                cotangents = torch.ones(chains,data_count,device=device,dtype=x.dtype)/data_count
                fmap = lambda theta: direct_prediction(theta, inputs, width)
                reference_vjps=[]
                for row, cotangent in zip(x,cotangents):
                    leaf=row.detach().clone().requires_grad_()
                    reference_vjps.append(torch.autograd.grad((fmap(leaf)*cotangent).sum(),leaf)[0])
                reference_vjps=torch.stack(reference_vjps)
                for mode in ['loop','vmap']:
                    for chunk in [None,4,16]:
                        target = Posterior(model,inputs,observations,batch_mode=mode,chunk_size=chunk)
                        values, gradients = target.value_and_grad(x)
                        torch.testing.assert_close(values,reference_values,atol=1e-8,rtol=1e-10)
                        torch.testing.assert_close(gradients,reference_gradients,atol=1e-8,rtol=1e-10)
                        timing=measure(lambda:target.value_and_grad(x),args.repetitions,device)
                        fs = FunctionSpaceSVGD(fmap,lambda z:z.square().sum(-1)/2,x,
                                               vectorized=mode=='vmap',chunk_size=chunk)
                        torch.testing.assert_close(fs._pullback(x,cotangents),reference_vjps,atol=1e-10,rtol=1e-10)
                        def outputs_and_vjp():
                            fs._outputs(x); fs._pullback(x,cotangents)
                        fs_timing=measure(outputs_and_vjp,args.repetitions,device)
                        rows.append(dict(width=width,parameters=x.shape[-1],data_count=data_count,chains=chains,
                                         batch_mode=mode,chunk_size=chunk,target_value_and_grad=timing,
                                         function_outputs_and_vjp=fs_timing,
                                         max_abs_value_error=float((values-reference_values).abs().max()),
                                         max_abs_gradient_error=float((gradients-reference_gradients).abs().max()),
                                         hidden_activation_tensor_bytes=min(chains,chunk or chains)*data_count*width*x.element_size(),
                                         process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    return rows


def diagnostics(samples):
    from lafa.diagnostics import rank_normalized_split_rhat,bulk_effective_sample_size,tail_effective_sample_size,mcse_mean
    result={}
    for name, fn in [('rank_normalized_split_rhat',rank_normalized_split_rhat),('bulk_ess',bulk_effective_sample_size),
                     ('tail_ess',tail_effective_sample_size),('mean_mcse',mcse_mean)]:
        try: result[name]=fn(samples)
        except ValueError as error: result[name]=dict(undefined=str(error))
    return result


def posterior_example(args,device):
    # Keep the simulated dataset independent of the requested scaling grid.
    torch.manual_seed(args.seed+100)
    width,chains,data_count=4,4,24
    noise_sd,prior_sd=.3,.7
    inputs=torch.linspace(-2,2,data_count,device=device,dtype=torch.float64)[:,None]
    model=network(width,device)
    truth=prior_sd*torch.randn(3*width+1,device=device,dtype=torch.float64)
    observed=direct_prediction(truth,inputs,width)+noise_sd*torch.randn(data_count,device=device,dtype=torch.float64)
    target=Posterior(model,inputs,observed,noise_sd=noise_sd,prior_sd=prior_sd,chunk_size=4)
    initial=prior_sd*torch.randn(chains,3*width+1,device=device,dtype=torch.float64)
    sampler=MALA(target,initial,args.step_size,generator=torch.Generator(device=device).manual_seed(args.seed+1))
    accepted=torch.zeros(chains,device=device,dtype=torch.float64)
    retained=[]; cumulative_seconds=[]
    synchronize(device); start=time.perf_counter()
    for iteration in range(args.burnin+args.draws):
        result=sampler.step()
        accepted+=result.accepted
        if iteration>=args.burnin:
            retained.append(result.position)
            synchronize(device); cumulative_seconds.append(time.perf_counter()-start)
    synchronize(device); seconds=time.perf_counter()-start
    samples=torch.stack(retained,dim=1)  # Independent chains, retained time, parameters.
    work=target.evaluated_states
    test_inputs=torch.linspace(-2.5,2.5,41,device=device,dtype=torch.float64)[:,None]
    with torch.no_grad():
        predictions=torch.vmap(torch.vmap(lambda theta:direct_prediction(theta,test_inputs,width)))(samples)
    latent_truth=direct_prediction(truth,test_inputs,width)
    quantiles=predictions.flatten(0,1).quantile(predictions.new_tensor([.05,.5,.95]),dim=0)
    prediction_draws=predictions+noise_sd*torch.randn_like(predictions)
    predictive_quantiles=prediction_draws.flatten(0,1).quantile(predictions.new_tensor([.05,.5,.95]),dim=0)
    result=dict(method='MALA',objective='Bayesian weight posterior with full-data Gaussian likelihood and independent Gaussian weight prior',
                width=width,parameters=3*width+1,chains=chains,data_count=data_count,noise_sd=noise_sd,prior_sd=prior_sd,
                seed=args.seed,data_and_initialization_seed=args.seed+100,
                proposal_seed=args.seed+1,burnin=args.burnin,retained_draws_per_chain=args.draws,step_size=args.step_size,
                seconds=seconds,acceptance_rate=accepted/(args.burnin+args.draws),
                evaluated_parameter_states=work,evaluated_observation_residuals=work*data_count,
                sample_shape=list(samples.shape),prediction_shape=list(predictions.shape),
                posterior_mean_function_rmse=(predictions.mean((0,1))-latent_truth).square().mean().sqrt(),
                latent_pointwise_90_percent_coverage=((latent_truth>=quantiles[0])&(latent_truth<=quantiles[2])).double().mean(),
                weight_diagnostics=diagnostics(samples),predictive_function_diagnostics=diagnostics(predictions[:,:,::10]),
                notes=['Finite-run diagnostics must be inspected; this run does not establish convergence.',
                       'Weight permutation symmetries can mix slowly even when predictions mix better.',
                       'Coverage is for one simulated dataset across correlated spatial locations, not calibrated repeated-dataset coverage.',
                       'No trusted exact high-dimensional posterior reference is available; no posterior bias estimate is asserted.'])
    torch.save(dict(samples=samples.cpu(),prediction_samples=predictions.cpu(),
                    retained_cumulative_seconds=torch.tensor(cumulative_seconds),truth_parameters=truth.cpu(),
                    training_inputs=inputs.cpu(),observations=observed.cpu(),test_inputs=test_inputs.cpu(),
                    true_function=latent_truth.cpu(),latent_pointwise_quantiles=quantiles.cpu(),
                    observation_predictive_quantiles=predictive_quantiles.cpu()),args.output/'posterior-traces.pt')
    return result


def serializable(value):
    if isinstance(value,torch.Tensor): return serializable(value.detach().cpu().tolist())
    if isinstance(value,dict): return {key:serializable(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)): return [serializable(item) for item in value]
    if isinstance(value,float) and not math.isfinite(value): return None
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,default=Path('benchmarks/results/neural-network'))
    parser.add_argument('--widths',nargs='+',type=int,default=[8,32,128])
    parser.add_argument('--data-sizes',nargs='+',type=int,default=[32,128])
    parser.add_argument('--chains',nargs='+',type=int,default=[1,8,32])
    parser.add_argument('--repetitions',type=int,default=5)
    parser.add_argument('--burnin',type=int,default=500)
    parser.add_argument('--draws',type=int,default=1000)
    parser.add_argument('--step-size',type=float,default=.0005)
    parser.add_argument('--seed',type=int,default=20261015)
    args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available(): parser.error('CUDA requested but unavailable; no CPU fallback')
    if min(args.widths+args.data_sizes+args.chains+[args.repetitions,args.draws])<1 or args.burnin<0:
        parser.error('sizes, repetitions and draws must be positive; burnin must be nonnegative')
    if not math.isfinite(args.step_size) or args.step_size<=0: parser.error('step size must be positive and finite')
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    device=torch.device(args.device)
    args.output.mkdir(parents=True,exist_ok=True)
    report=dict(device=str(device),dtype='float64',seed=args.seed,torch=torch.__version__,
                python=platform.python_version(),cpu_threads=torch.get_num_threads(),
                hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor() or platform.machine(),
                notes=['Warmup excluded from timing. Repeated evaluations are deterministic at fixed states.',
                       'CPU RSS is a shared-process cumulative high-water mark, not attributable peak allocation per row.',
                       'Hidden activation bytes count one tensor only, not total activation memory or autograd storage.',
                       'Function-space timings measure outputs and destination VJPs, excluding particle kernel interactions.',
                       'CUDA validation remains pending unless this report explicitly records device=cuda.'],
                scaling=scaling(args,device),posterior_example=posterior_example(args,device))
    output=args.output/'results.json'
    output.write_text(json.dumps(serializable(report),indent=2,allow_nan=False))
    print(json.dumps(dict(report=str(output),measurements=len(report['scaling']),
                          posterior_seconds=report['posterior_example']['seconds'])))


if __name__=='__main__': main()
