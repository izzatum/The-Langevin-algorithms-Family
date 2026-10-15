"""Opt-in execution for deterministic pure tensor potentials.

``ULA(compile_potential(potential), initial)`` compiles potential and its functional
input gradient together. Randomness, transition logic, finite validation and
owned results remain eager. Compilation is lazy and its first call can be costly.
There is no silent fallback: unsupported graphs raise with the original cause.
"""
import torch


class _CompiledPotential:
    def __init__(self, potential, *, backend, mode, fullgraph, dynamic):
        def summed_with_values(x):
            values = potential(x)
            return values.sum(), values

        gradient_and_value = torch.func.grad_and_value(summed_with_values, has_aux=True)

        def evaluate(x):
            gradient, (_, value) = gradient_and_value(x)
            return value, gradient

        self._evaluate = torch.compile(evaluate, backend=backend, mode=mode,
                                       fullgraph=fullgraph, dynamic=dynamic)

    def value_and_grad(self, x):
        try:
            # Native sampling is also valid inside torch.no_grad().
            with torch.enable_grad():
                return self._evaluate(x)
        except Exception as exc:
            raise RuntimeError('compiled potential/gradient evaluation failed; use the original '
                               'eager potential for unsupported targets') from exc


def compile_potential(potential, *, backend='inductor', mode=None, fullgraph=True, dynamic=False):
    """Compile a pure tensor callable ``U(x) -> values`` and its gradient.

    Values must have shape ``x.shape[:-1]`` and chains must be independent, just
    as for eager targets. Keep shapes/dtype/device fixed to avoid recompilation.
    Use deterministic, mutation-free tensor operations supported by torch.func
    and torch.compile. Score targets and custom value_and_grad implementations
    are unsupported. No RNG is compiled. Errors never silently fall back.
    ``backend='aot_eager'`` is useful to validate compatibility without codegen.
    Default finite/shape/dtype/device validation still occurs in the sampler.
    """
    if not callable(potential) or hasattr(potential, 'value_and_grad') or getattr(potential, 'score_only', False):
        raise TypeError('compile_potential requires a pure tensor callable, without a custom value_and_grad protocol')
    return _CompiledPotential(potential, backend=backend, mode=mode, fullgraph=fullgraph, dynamic=dynamic)
