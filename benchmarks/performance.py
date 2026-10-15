#!/usr/bin/env python3
"""End-to-end sampler/optimizer timings with strict checks and owned outputs.

Optional --compile compiles the potential AND its gradient, never a supplied
noise-only kernel. Native transitions, RNG, validation and requested snapshots
are included in evaluation. First target compilation, warm-up and repeat timings
are reported separately. CUDA requests fail explicitly when unavailable.
"""
import argparse
import json
import platform
from pathlib import Path
import resource
import statistics
import tempfile
import time
import torch
from lafa import optim, samplers
from lafa.execution import compile_potential
from lafa.targets import value_and_grad
from lafa.particles import SVGD
from lafa.kernels import RBF
from lafa.recording import Recorder


def gaussian(x):
    return x.square().sum(-1)/2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', type=Path, default=Path('benchmarks/performance-results.json'))
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--warmup', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--chains', type=int, default=32)
    parser.add_argument('--dimensions', type=int, nargs='+', default=[8, 64, 256])
    parser.add_argument('--methods', nargs='+', choices=['ULA','MALA','LipULA','LipMALA','BAOAB','KineticLangevin'], default=['ULA','MALA','BAOAB'])
    parser.add_argument('--dtypes', nargs='+', choices=['float32','float64'], default=['float32','float64'])
    parser.add_argument('--targets', nargs='+', choices=['gaussian','nonlinear_dense'], default=['gaussian','nonlinear_dense'])
    parser.add_argument('--compile', action='store_true', help='Include end-to-end samplers with compiled potential and gradient')
    parser.add_argument('--compile-backend', default='inductor')
    parser.add_argument('--skip-auxiliary', action='store_true', help='Skip existing particle/recording and optimizer comparisons')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable; no CPU fallback')
    if min(args.steps, args.repeats, args.chains, *args.dimensions) < 1 or args.warmup < 0:
        parser.error('steps, repeats, chains and dimensions must be positive; warmup must be nonnegative')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    seed = 2026
    torch.manual_seed(seed)

    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    def measure(fn, count):
        synchronize()
        start = time.perf_counter()
        for _ in range(count):
            fn()
        synchronize()
        return time.perf_counter() - start

    def timing(fn, count=None):
        count = args.steps if count is None else count
        warmup = measure(fn, args.warmup)
        samples = [measure(fn, count) for _ in range(args.repeats)]
        median = statistics.median(samples)
        return dict(warmup_steps=args.warmup, warmup_seconds=warmup,
                    evaluation_steps=count, repeats=args.repeats, evaluation_seconds=median,
                    repeat_seconds=samples, seconds_per_step=median/count)

    rows, initializations = [], []
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for dimension in args.dimensions:
            x = torch.zeros((args.chains, dimension), device=device, dtype=dtype)
            # Fixed matrix couples coordinates within each chain, never chains.
            matrix = torch.randn((dimension, dimension), device=device, dtype=dtype,
                                 generator=torch.Generator(device=device).manual_seed(seed))/dimension**.5
            def nonlinear_dense(z):
                return z.square().sum(-1)/2 + .05*(z @ matrix).sin().square().sum(-1)
            for target_name in args.targets:
                potential = gaussian if target_name == 'gaussian' else nonlinear_dense
                targets = [('eager', potential)]
                if args.compile:
                    # Each configuration is measured independently. Reset Dynamo's
                    # graph cache to avoid its per-function recompilation limit.
                    # Inductor's disk cache can still make later runs faster.
                    torch.compiler.reset()
                    probe = torch.full_like(x, .37)
                    synchronize(); start = time.perf_counter()
                    compiled = compile_potential(potential, backend=args.compile_backend)
                    actual = value_and_grad(compiled, probe)
                    synchronize(); initialization = time.perf_counter()-start
                    expected = value_and_grad(potential, probe)
                    for left, right in zip(actual, expected):
                        torch.testing.assert_close(left, right)
                    initializations.append(dict(target=target_name, dtype=dtype_name, dimension=dimension,
                        chains=args.chains, backend=args.compile_backend,
                        first_target_call_including_compile_seconds=initialization,
                        initial_value_gradient_agreement=True))
                    targets.append(('compiled_target_gradient', compiled))
                for execution, target in targets:
                    for method in args.methods:
                        for output, fields in [('full', None), ('position', ('position',)), ('none', ())]:
                            sampler = getattr(samplers, method)(target, x, .02,
                                generator=torch.Generator(device=device).manual_seed(seed))
                            if device.type == 'cuda':
                                torch.cuda.reset_peak_memory_stats(device)
                            measured = timing(lambda: sampler.step(fields=fields))
                            rows.append(dict(kind='sampler', method=method, target=target_name,
                                execution=execution, output=output, dtype=dtype_name,
                                dimension=dimension, chains=args.chains,
                                transition_count_per_repeat=args.steps,
                                chain_transitions_per_repeat=args.steps*args.chains,
                                target_gradient_calls_per_repeat=args.steps,
                                strict_validation=True, includes_randomness=True, includes_owned_outputs=True,
                                live_position_tensor_bytes=x.numel()*x.element_size(),
                                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                                **measured))
    if not args.skip_auxiliary:
        for dtype_name in args.dtypes:
            dtype = getattr(torch, dtype_name)
            for dimension in args.dimensions:
                for method in ['ULA','MALA']:
                    for allocation in ['cached_scales','rebuild_scales_reference']:
                        parameter = torch.nn.Parameter(torch.zeros(dimension,device=device,dtype=dtype))
                        optimizer = getattr(optim,method)([parameter],lr=.02,
                            generator=torch.Generator(device=device).manual_seed(seed))
                        def closure():
                            optimizer.zero_grad(); value=parameter.square().sum()/2; value.backward(); return value
                        def update():
                            if allocation == 'rebuild_scales_reference':
                                # Same current implementation and transaction guarantees,
                                # force only scale buffers to be rebuilt for comparison.
                                optimizer._scale_key = None
                            optimizer.step(closure)
                        rows.append(dict(kind='optimizer',method=method,dtype=dtype_name,dimension=dimension,
                            allocation=allocation, target='gaussian', chains=1,
                            includes_closure_backward_rng_validation_and_rollback_snapshot=True,
                            **timing(update)))
        dtype = torch.float64
        for dimension in [8, 32]:
            for particles in [16, 64, 128]:
                x = torch.randn((particles,dimension),device=device,dtype=dtype)
                sampler = SVGD(gaussian,x,step_size=.02,kernel=RBF(2.),block_size=32)
                rows.append(dict(kind='particles',method='SVGD exact blocked',dtype='float64',
                                 dimension=dimension,particles=particles,pair_block=32,
                                 **timing(sampler.step,count=min(args.steps,20))))
        with tempfile.TemporaryDirectory(prefix='lafa-performance-') as temporary:
            for mode,every,asynchronous in [('off',1,False),('sync',1,False),('async',1,True),('sparse',10,False)]:
                samples=[]
                disk_bytes=0
                writer=None
                for repeat in range(args.repeats):
                    sampler=samplers.ULA(gaussian,torch.zeros((32,64),device=device,dtype=dtype),.02,
                        generator=torch.Generator(device=device).manual_seed(seed))
                    warmup=measure(lambda:sampler.step(fields=()),args.warmup)
                    directory=Path(temporary)/mode/str(repeat)
                    writer=None if mode=='off' else Recorder(directory,fields=['position'],every=every,
                            chunk_size=8,max_buffer_bytes=512*1024,async_write=asynchronous)
                    synchronize(); start=time.perf_counter()
                    for index in range(args.steps):
                        result=sampler.step(fields=() if writer is None else ('position',))
                        if writer is not None:
                            writer.record(index,position=result['position'])
                    if writer is not None:
                        writer.close()
                    synchronize(); samples.append(time.perf_counter()-start)
                    disk_bytes=sum(path.stat().st_size for path in directory.rglob('*') if path.is_file())
                elapsed=statistics.median(samples)
                rows.append(dict(kind='recording',mode=mode,dtype='float64',dimension=64,chains=32,every=every,
                    warmup_steps=args.warmup,warmup_seconds=warmup,evaluation_steps=args.steps,repeats=args.repeats,
                    repeat_seconds=samples,evaluation_seconds=elapsed,seconds_per_step=elapsed/args.steps,
                    includes_CPU_staging_and_flush=True,disk_bytes=disk_bytes,
                    writer_stats=writer.stats if writer else None))
    cpu_model = platform.processor() or platform.machine()
    cpuinfo=Path('/proc/cpuinfo')
    if device.type=='cpu' and cpuinfo.exists():
        cpu_model=next((line.split(':',1)[1].strip() for line in cpuinfo.read_text().splitlines()
                        if line.startswith('model name')),cpu_model)
    report=dict(schema_version=2,device=str(device),dtypes=args.dtypes,seed=seed,torch=torch.__version__,
        python=platform.python_version(),hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else cpu_model,
        cpu_threads=torch.get_num_threads(),cuda_available=torch.cuda.is_available(),
        process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        notes=['Single-process timings; median and all repeat intervals are retained. Not mixing/ESS measurements.',
               'Every sampler evaluation includes target, gradient, RNG, transition, strict validation, and selected owned outputs.',
               'Only potential plus gradient is compiled; warm-up and first target compile are excluded from evaluation.',
               'Dynamo cache resets per target configuration; Inductor disk cache is not cleared.',
               'Optimizer scale reference rebuilds buffers within the same implementation; it is not a previous-release benchmark.',
               'CPU RSS is process high-water mark, including compilation; position bytes exclude other state.',
               'Recording includes caller host staging and final close/flush.'],
        measurements=rows,compile_initializations=initializations)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(output=str(args.output),measurements=len(rows),compile_initializations=len(initializations),
                         device=str(device),dtypes=args.dtypes),indent=2))


if __name__=='__main__':
    main()
