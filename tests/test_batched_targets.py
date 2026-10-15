"""Independent single-state oracles for functional batching and destination VJPs."""
import pytest
import torch

from lafa.targets import ModulePotential, value_and_grad
from lafa.particles import FunctionSpaceSVGD, FunctionSpaceRepulsiveEnsemble


class TiedBuffered(torch.nn.Module):
    def __init__(self, dtype=torch.float64, device='cpu'):
        super().__init__()
        self.p = torch.nn.Parameter(torch.tensor([.4, -.6], dtype=dtype, device=device))
        self.q = self.p
        self.register_buffer('offset', torch.tensor([.2, -.1], dtype=dtype, device=device))
        self.register_buffer('alias', self.offset)

    def forward(self):
        self.offset.add_(self.p)
        return (self.p + self.q).sin() + self.alias


@pytest.mark.parametrize('mode', ['loop', 'vmap'])
@pytest.mark.parametrize('chunk', [None, 1, 2, 9])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_module_batch_matches_independent_tied_mutating_buffer_oracle(mode, chunk, dtype):
    model = TiedBuffered(dtype)
    target = ModulePotential(model, lambda y: y.square().sum()/2, batch_mode=mode, chunk_size=chunk)
    x = torch.tensor([[.1, .2], [.8, -.4], [-.3, .7]], dtype=dtype, requires_grad=True)
    before = x.detach().clone()
    expected_values, expected_grads = [], []
    for row in before:
        leaf = row.clone().requires_grad_()
        # Independent formula includes the same-object buffer and parameter ties.
        value = ((2*leaf).sin() + leaf + leaf.new_tensor([.2, -.1])).square().sum()/2
        expected_values.append(value.detach())
        expected_grads.append(torch.autograd.grad(value, leaf)[0])
    expected_values, expected_grads = torch.stack(expected_values), torch.stack(expected_grads)
    for _ in range(2):
        values, gradients = target.value_and_grad(x)
        torch.testing.assert_close(values, expected_values)
        torch.testing.assert_close(gradients, expected_grads)
        torch.testing.assert_close(target(x), expected_values)
        torch.testing.assert_close(torch.autograd.grad(target(x).sum(), x)[0], expected_grads)
        assert not values.requires_grad and not gradients.requires_grad
    torch.testing.assert_close(x, before)
    assert x.grad is None and model.p.grad is None
    torch.testing.assert_close(model.offset, x.new_tensor([.2, -.1]))
    assert model.p is model.q and model.offset is model.alias
    single_value, single_gradient = value_and_grad(target, x[0])
    torch.testing.assert_close(single_value, expected_values[0])
    torch.testing.assert_close(single_gradient, expected_grads[0])
    values.zero_(); gradients.zero_()
    torch.testing.assert_close(target.value_and_grad(x)[0], expected_values)


def test_module_value_grad_without_outer_grad_and_unused_parameters():
    model = torch.nn.Linear(2, 1).double()
    target = ModulePotential(model, lambda y: y.new_tensor(3.), args=(torch.ones(2, dtype=torch.float64),))
    x = target.flatten_parameters().repeat(3, 1)
    with torch.no_grad():
        values, gradients = target.value_and_grad(x)
    torch.testing.assert_close(values, x.new_full((3,), 3.))
    torch.testing.assert_close(gradients, torch.zeros_like(x))


@pytest.mark.parametrize('keyword,value', [('batch_mode','auto'), ('chunk_size',0), ('chunk_size',-1), ('chunk_size',True), ('chunk_size',1.5)])
def test_module_invalid_configuration(keyword, value):
    with pytest.raises(ValueError):
        ModulePotential(TiedBuffered(), lambda y:y.sum(), **{keyword:value})


@pytest.mark.parametrize('shape', [(2,2,2), (0,2), (3,), ()])
def test_module_invalid_positions(shape):
    target = ModulePotential(TiedBuffered(), lambda y:y.sum())
    with pytest.raises(ValueError):
        target(torch.zeros(shape, dtype=torch.float64))


def test_module_rejects_nonfinite_integer_and_loss_shape():
    target = ModulePotential(TiedBuffered(), lambda y:y.sum())
    for x in [torch.ones(2, dtype=torch.int64), torch.tensor([float('nan'), 0.])]:
        with pytest.raises(ValueError): target.value_and_grad(x)
    target = ModulePotential(TiedBuffered(), lambda y:y)
    with pytest.raises(ValueError, match='scalar'): target(torch.ones(2, dtype=torch.float64))


def test_vmap_unsupported_operation_has_explicit_loop_escape_hatch():
    class Branch(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        def forward(self):
            return self.p.square() if self.p.item() > 0 else self.p.sin()
    x = torch.tensor([[1.], [-1.]], dtype=torch.float64)
    with pytest.raises(RuntimeError, match="batch_mode='loop'"):
        ModulePotential(Branch(), lambda y:y.sum())(x)
    result = ModulePotential(Branch(), lambda y:y.sum(), batch_mode='loop')(x)
    torch.testing.assert_close(result, torch.stack([x.new_tensor(1.), x.new_tensor(-1.).sin()]))


@pytest.mark.parametrize('mode', ['vmap', 'loop'])
def test_parameter_mutation_cannot_change_input_or_model(mode):
    class Mutating(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.p = torch.nn.Parameter(torch.ones(1, dtype=torch.float64))
        def forward(self):
            self.p.add_(1.)
            return self.p.square().sum()
    model = Mutating()
    target = ModulePotential(model, lambda y:y, batch_mode=mode, chunk_size=2)
    x = torch.tensor([[1.], [2.], [3.]], dtype=torch.float64, requires_grad=True)
    expected = x.detach().clone()
    values, grads = target.value_and_grad(x)
    torch.testing.assert_close(values, (expected+1).square().sum(-1))
    torch.testing.assert_close(grads, 2*(expected+1))
    torch.testing.assert_close(x, expected)
    torch.testing.assert_close(model.p, torch.ones_like(model.p))


def test_cpu_target_rejects_wrong_dtype_and_broken_model_errors_are_preserved():
    target = ModulePotential(TiedBuffered(), lambda y:y.float().sum())
    with pytest.raises(ValueError, match='dtype'): target.value_and_grad(torch.ones(3,2,dtype=torch.float64))
    class Broken(TiedBuffered):
        def forward(self): raise RuntimeError('model implementation error')
    with pytest.raises(RuntimeError, match='^model implementation error$'):
        ModulePotential(Broken(), lambda y:y.sum())(torch.ones(3,2,dtype=torch.float64))


@pytest.mark.parametrize('cls', [FunctionSpaceSVGD, FunctionSpaceRepulsiveEnsemble])
@pytest.mark.parametrize('chunk', [None, 1, 2, 8])
def test_vectorized_functions_and_destination_vjps_match_independent_loops(cls, chunk):
    x = torch.tensor([[.1, -.3], [.8, .4], [-.6, .7]], dtype=torch.float64)
    fmap = lambda t: torch.stack([t[0].sin()+t[1].square(), t[0]*t[1], t.sum().cos()])
    potential = lambda z:z.square().sum(-1)/2
    obj = cls(fmap, potential, x, vectorized=True, chunk_size=chunk)
    reference = cls(fmap, potential, x)
    cotangent = torch.arange(9, dtype=x.dtype).reshape(3, 3)/9
    expected = []
    for row, c in zip(x, cotangent):
        leaf = row.detach().clone().requires_grad_()
        expected.append(torch.autograd.grad((fmap(leaf)*c).sum(), leaf)[0])
    torch.testing.assert_close(obj._pullback(x, cotangent), torch.stack(expected))
    torch.testing.assert_close(obj._outputs(x), torch.stack([fmap(row) for row in x]))
    for _ in range(2):
        torch.testing.assert_close(obj.step().position, reference.step().position)
    torch.testing.assert_close(obj.position, reference.position)
    torch.testing.assert_close(x, torch.tensor([[.1,-.3],[.8,.4],[-.6,.7]], dtype=x.dtype))


@pytest.mark.parametrize('kwargs', [{'vectorized':1}, {'chunk_size':0}, {'chunk_size':True}])
def test_function_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        FunctionSpaceSVGD(lambda t:t, lambda z:z.square().sum(-1), torch.ones(2,2), **kwargs)


def test_function_vmap_error_is_actionable():
    fmap = lambda t:t if t[0].item() > 0 else -t
    with pytest.raises(RuntimeError, match='vectorized=False'):
        FunctionSpaceSVGD(fmap, lambda z:z.square().sum(-1), torch.ones(2,2), vectorized=True)


@pytest.mark.parametrize('cls', [FunctionSpaceSVGD,FunctionSpaceRepulsiveEnsemble])
def test_function_checkpoint_restores_execution_controls(cls):
    x=torch.tensor([[.2,-.3],[.1,.6],[.8,.7]],dtype=torch.float64)
    fmap=lambda t:torch.stack([t.sum().sin(),t.square().sum()])
    potential=lambda z:z.square().sum(-1)/2
    original=cls(fmap,potential,x,vectorized=True,chunk_size=2)
    original.step()
    checkpoint=original.state_dict()
    restored=cls(fmap,potential,x)
    restored.load_state_dict(checkpoint)
    assert restored.vectorized is True and restored.chunk_size == 2
    torch.testing.assert_close(restored.step().position,original.step().position,rtol=0,atol=0)
    checkpoint['extra']={}
    cls(fmap,potential,x).load_state_dict(checkpoint)  # Legacy checkpoints remain readable.
    checkpoint['extra']={'vectorized':True,'chunk_size':0}
    with pytest.raises(ValueError,match='chunk_size'): restored.load_state_dict(checkpoint)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='requires real CUDA hardware')
def test_cuda_batched_target_and_device_validation():
    target = ModulePotential(TiedBuffered(device='cuda'), lambda y:y.square().sum())
    x = target.flatten_parameters().repeat(3,1)
    values, gradients = target.value_and_grad(x)
    assert values.device == gradients.device == x.device
    target = ModulePotential(TiedBuffered(device='cuda'), lambda y:y.cpu().sum())
    with pytest.raises(ValueError, match='device'): target.value_and_grad(x)
