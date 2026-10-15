#!/usr/bin/env python3
"""Bounded timings of actual eager updates, pair interactions and recording.

Warm-up and timed evaluation are separate. CUDA intervals are synchronized;
recording includes device-to-host staging and closing/flushing the writer.
"""
import argparse
import json
import platform
from pathlib import Path
import resource
import tempfile
import time
import torch
from lafa.samplers import ULA
from lafa.particles import SVGD
from lafa.kernels import RBF
from lafa.recording import Recorder


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    p.add_argument('--output', type=Path, default=Path('benchmarks/performance-results.json'))
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--compile', action='store_true', help='Also compile the functional ULA transition')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but unavailable')
    if args.steps < 1:
        p.error('--steps must be positive')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    dtype = torch.float64
    seed = 2026
    torch.manual_seed(seed)
    def synchronize():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    def measure(fn, count):
        synchronize()
        start = time.perf_counter()
        for _ in range(count): fn()
        synchronize()
        return time.perf_counter() - start
    rows = []
    for dimension in [8, 64, 256]:
        x = torch.zeros((32, dimension), device=device, dtype=dtype)
        sampler = ULA(lambda z: z.square().sum(-1)/2, x, .02,
                      generator=torch.Generator(device=device).manual_seed(seed))
        if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
        warmup = measure(sampler.step, 8)
        evaluation = measure(sampler.step, args.steps)
        rows.append(dict(kind='sampler', method='ULA eager', dimension=dimension, chains=32,
                         warmup_steps=8, warmup_seconds=warmup, evaluation_steps=args.steps,
                         evaluation_seconds=evaluation, seconds_per_step=evaluation/args.steps,
                         live_state_tensor_bytes=x.numel()*x.element_size(),
                         cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
    for dimension in [8, 32]:
        for particles in [16, 64, 128]:
            x = torch.randn((particles,dimension),device=device,dtype=dtype)
            sampler = SVGD(lambda z:z.square().sum(-1)/2,x,step_size=.02,kernel=RBF(2.),block_size=32)
            if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
            warmup = measure(sampler.step, 3)
            count = min(args.steps, 20)
            evaluation = measure(sampler.step, count)
            rows.append(dict(kind='particles',method='SVGD exact blocked',dimension=dimension,particles=particles,
                             pair_block=32,warmup_steps=3,warmup_seconds=warmup,evaluation_steps=count,
                             evaluation_seconds=evaluation,seconds_per_step=evaluation/count,
                             cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None))
    with tempfile.TemporaryDirectory(prefix='lafa-performance-') as temporary:
        for mode,every,asynchronous in [('off',1,False),('sync',1,False),('async',1,True),('sparse',10,False)]:
            x = torch.zeros((32,64),device=device,dtype=dtype)
            sampler = ULA(lambda z:z.square().sum(-1)/2,x,.02,
                          generator=torch.Generator(device=device).manual_seed(seed))
            warmup = measure(sampler.step,8)
            writer = None if mode=='off' else Recorder(Path(temporary)/mode,fields=['position'],every=every,
                            chunk_size=8,max_buffer_bytes=512*1024,async_write=asynchronous)
            synchronize(); start=time.perf_counter()
            for i in range(args.steps):
                result=sampler.step()
                if writer is not None: writer.record(i,position=result.position)
            if writer is not None: writer.close()
            synchronize(); elapsed=time.perf_counter()-start
            disk_bytes = sum(f.stat().st_size for f in (Path(temporary)/mode).rglob('*') if f.is_file()) if writer else 0
            rows.append(dict(kind='recording',mode=mode,dimension=64,chains=32,every=every,
                             warmup_steps=8,warmup_seconds=warmup,evaluation_steps=args.steps,
                             evaluation_seconds=elapsed,seconds_per_step=elapsed/args.steps,
                             includes_CPU_staging_and_flush=True,disk_bytes=disk_bytes,
                             writer_stats=writer.stats if writer else None))
    compiled_result = None
    if args.compile:
        from lafa.functional import ula_step
        x = torch.randn((32,64),device=device,dtype=dtype)
        g = x.clone(); noise=torch.randn_like(x)
        def functional(z, gradient, eta): return ula_step(z,gradient,.02,noise=eta)
        synchronize(); start=time.perf_counter()
        compiled = torch.compile(functional)
        actual = compiled(x,g,noise)
        synchronize(); initialization=time.perf_counter()-start
        torch.testing.assert_close(actual,functional(x,g,noise))
        compiled_result=dict(first_call_including_compile_seconds=initialization,
                             warmup_seconds=measure(lambda:compiled(x,g,noise),8),
                             compiled_evaluation_seconds=measure(lambda:compiled(x,g,noise),args.steps),
                             eager_evaluation_seconds=measure(lambda:functional(x,g,noise),args.steps),
                             steps=args.steps,note='Functional transition only: excludes potential, autograd and sampler snapshots')
    report = dict(device=str(device),dtype=str(dtype),seed=seed,torch=torch.__version__,python=platform.python_version(),
                  hardware=torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor() or platform.machine(),
                  cpu_threads=torch.get_num_threads(),process_peak_rss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  notes=['Single-process measurements, not speedup guarantees.','Warm-up is excluded from evaluation.',
                         'CPU RSS is process high-water mark; state tensor bytes are not total memory.',
                         'Async recording overlaps disk writing; host staging is included on the caller.'],
                  measurements=rows,functional_compile=compiled_result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
