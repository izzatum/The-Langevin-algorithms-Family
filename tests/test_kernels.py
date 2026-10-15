import pytest
import torch
from lafa.kernels import RBF, IMQ, median_rbf

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]

@pytest.mark.parametrize('device', DEVICES)
@pytest.mark.parametrize('kernel', [RBF(1.3), IMQ(0.7, -0.4)])
def test_derivatives_against_autograd(device, kernel):
    x = torch.tensor([0.3, -0.9, 0.4], dtype=torch.float64, device=device, requires_grad=True)
    y = torch.tensor([-0.2, 0.1, 1.1], dtype=torch.float64, device=device, requires_grad=True)
    k, b, trace = kernel.radial(((x-y)**2).sum(), 3)
    gx, gy = torch.autograd.grad(k, (x,y), create_graph=True)
    mixed = sum(torch.autograd.grad(gx[i], y, retain_graph=True)[0][i] for i in range(3))
    torch.testing.assert_close(gx, -b*(x-y))
    torch.testing.assert_close(gy, b*(x-y))
    torch.testing.assert_close(trace, mixed)
    torch.testing.assert_close(kernel(x[None], y[None])[0,0], k)

@pytest.mark.parametrize('kwargs', [{'length_scale':0}, {'length_scale':float('nan')}])
def test_rbf_invalid(kwargs):
    with pytest.raises(ValueError): RBF(**kwargs)

@pytest.mark.parametrize('scale,beta', [(0,-.5),(1,0),(1,-1),(1,float('nan'))])
def test_imq_invalid(scale,beta):
    with pytest.raises(ValueError): IMQ(scale,beta)

def test_bounded_median_reproducible_and_degenerate():
    x = torch.arange(150, dtype=torch.float64).reshape(50,3)
    assert median_rbf(x, max_pairs=13, seed=3) == median_rbf(x, max_pairs=13, seed=3)
    assert median_rbf(torch.zeros(1,3)).length_scale > 0
    assert median_rbf(torch.zeros(8,3)).length_scale > 0
