# API and recording guide

Use the root `lafa` classes for optimizer-style parameter mutation, `lafa.samplers` for explicit independent chains, `lafa.particles` for interacting particles, and `lafa.specialized` for algorithms with their own inference state. The [algorithm register](research/algorithm-register.md) lists every public method and its assumptions. Class/function docstrings specify optional arguments; the [six notebooks](tutorials.md) show complete runs.

## Targets and devices

`potential(x)` returns `U=-log π`, with shape `x.shape[:-1]`. `from_log_prob` negates log density; `from_score` adapts a score for gradient-only methods, while MALA requires potential values. `ModulePotential` uses `torch.func.functional_call` to evaluate a flattened model state with explicit parameter ordering. `positive_transform` includes the change-of-variables Jacobian. The prior and likelihood must use the same convention; averaging a likelihood over a changing minibatch changes the target.

Construct the target, initial state, callback data and `torch.Generator(device=...)` on the same device. Float64 is useful for references; float32 performance and accuracy are workload dependent. Native and specialized sampler checkpoint loaders require matching shape, dtype and device. Particle loaders convert stored coordinates/weights to the receiving object's device/dtype; such conversion is not bitwise continuation. Exact continuation requires the same device/software and deterministic target, including any external model buffers/RNG.

## Results and gradients

Tensor samplers return `StepResult(position, potential, gradient, accepted, stats)`. Position, potential and gradient refer to the returned current state, including after rejection. `accepted` is absent for unadjusted transports. Statistics include step size and, for kinetic methods, momentum. Optimizer adapters expose their last snapshot as `.last_result`. Particle `.gradient` is the potential gradient at the returned state, not the Stein/KDE update direction; function-space methods pull gradients back to parameter coordinates. Specialized results expose `.position`, or `.theta` and `.particles`; filtering exposes state ensembles and learned parameters. Read that method's statistics before naming recorded fields.

`state_dict()` / `load_state_dict()` preserve numerical state, schedules/adaptation, owned generator state and configuration. Reconstruct callbacks and the same target before loading. Checkpoints are separate from traces; filtering checkpoints currently occur between observation stages. Saving checkpoints with `torch.save` is supported; load your own trusted tensor checkpoints with `torch.load(..., weights_only=True)`. Checkpoints do not serialize executable target/model definitions.

## Recording and checkpoints

`Recorder` accepts arbitrary named tensor fields. Select only required fields and a stride; derive spatial projections or predictions before passing them when full states are unnecessary. The recorder detaches and owns selected snapshots, preserves dtype by default, and writes atomic tensor chunks plus a JSON manifest. Use `storage_dtype` only when lossy precision is acceptable.

```python
from lafa.recording import Recorder

with Recorder('/tmp/lafa-trace', fields=['position', 'gradient'],
              every=10, chunk_size=32, max_buffer_bytes=8 * 1024**2,
              async_write=True) as trace:
    for step in range(1000):
        result = chain.step()   # an already constructed tensor sampler
        trace.record(step, position=result.position, gradient=result.gradient)
```

`max_buffer_bytes` bounds reserved tensor payloads including pending writer chunks; producer backpressure waits for capacity. A selected record larger than the limit is rejected. The cap excludes Python/serialization overhead and manifest entries, which grow with the number of committed chunks. Device-to-host staging is synchronous; `async_write` overlaps **disk** writing. `flush()` or context exit waits for persistence and propagates writer errors. Saving every field every step necessarily costs transfer, memory bandwidth and storage.

`TraceReader(path)` iterates chunks without loading the full run; `.read(field)` materializes that selected trace. `RingBuffer(capacity, fields)` retains only the latest records. `OnlineMoments` maintains running mean/variance without a history. `estimate_storage_bytes` estimates tensor payload, not filesystem/metadata overhead.

To resume, preserve the sampler checkpoint and last committed trace step together at a flush boundary, reconstruct the same target, load the checkpoint, and open `Recorder(..., resume=True)`. Record monotonically increasing step numbers. Recovery accepts committed manifest entries and rejects incompatible field/stride/dtype settings; it does not infer which sampler state belongs to a trace. Do not advance a checkpoint past data you still need to record.

Additional quantities can be recorded from public state/statistics or explicit user callbacks. Recomputing predictions or gradients has a cost; transient internal intermediates are not universally retained by default. Full diagnostic histories, higher derivative tensors and autograd graphs are intentionally not implicit sampler state.
