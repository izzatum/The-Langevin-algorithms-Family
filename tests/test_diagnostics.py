import math
import pytest
import torch
from lafa.kernels import RBF, IMQ
from lafa.diagnostics import ksd

DEVICES = ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable'))]

@pytest.mark.parametrize('device', DEVICES)
def test_ksd_exact_negative_u(device):
    x = torch.tensor([[-1.],[1.]], dtype=torch.float64, device=device)
    torch.testing.assert_close(ksd(x,-x,kernel=RBF(1)), x.new_tensor(1-4*math.exp(-2)))
    torch.testing.assert_close(ksd(x,-x,kernel=RBF(1),estimator='u'), x.new_tensor(-8*math.exp(-2)))
    with pytest.raises(ValueError): ksd(x,-x,estimator='u',squared=False)
    with pytest.raises(ValueError): ksd(x[:1],-x[:1],estimator='u')
    with pytest.raises(ValueError): ksd(x,-x,weights=x.new_tensor([1.,0.]),estimator='u')

@pytest.mark.parametrize('kernel', [RBF(.8), IMQ(.9,-.6)])
@pytest.mark.parametrize('block_size', [1,2,4,9])
def test_ksd_weighted_pair_oracle(kernel,block_size):
    g = torch.Generator().manual_seed(71)
    x = torch.randn(5,3,generator=g,dtype=torch.float64)
    s = -x**3
    w = torch.tensor([1,2,3,4,5],dtype=x.dtype); w /= w.sum()
    ref = x.new_zeros(())
    diag = x.new_zeros(())
    for i in range(5):
        for j in range(5):
            a,b = x[i].clone().requires_grad_(),x[j].clone().requires_grad_()
            t = ((a-b)**2).sum()
            k = torch.exp(-t/(2*kernel.length_scale**2)) if isinstance(kernel,RBF) else (kernel.scale**2+t)**kernel.beta
            ga,gb = torch.autograd.grad(k,(a,b),create_graph=True)
            tr = sum(torch.autograd.grad(ga[d],b,retain_graph=True)[0][d] for d in range(3))
            u = k*(s[i]*s[j]).sum()+(s[i]*gb).sum()+(s[j]*ga).sum()+tr
            ref = ref+w[i]*w[j]*u.detach()
            if i==j: diag = diag+w[i]*w[j]*u.detach()
    torch.testing.assert_close(ksd(x,lambda y:-y**3,kernel=kernel,weights=w,block_size=block_size),ref)
    torch.testing.assert_close(ksd(x,s,kernel=kernel,weights=w,block_size=block_size,estimator='u'),(ref-diag)/(1-w.square().sum()))


def test_autocorrelation_direct_oracle():
    from lafa.diagnostics import autocorrelation
    x=torch.tensor([[[1.],[3.],[2.],[5.],[4.]],[[2.],[1.],[5.],[3.],[4.]]],dtype=torch.float64)
    centered=x-x.mean(1,keepdim=True)
    expected=torch.stack([(centered[:,:5-lag]*centered[:,lag:]).sum(1)/centered.square().sum(1) for lag in range(4)],dim=1)
    torch.testing.assert_close(autocorrelation(x,max_lag=3),expected)


def test_rank_normalized_split_rhat_manual_ties_oracle():
    from lafa.diagnostics import rank_normalized_split_rhat
    x=torch.tensor([[[0.],[1.],[1.],[3.],[4.],[6.]],[[2.],[3.],[4.],[4.],[5.],[8.]]],dtype=torch.float64)
    split=torch.cat((x[:,:3],x[:,-3:]),dim=0)
    def oracle(data):
        flat=data.flatten()
        ranks=torch.stack([1+(flat<v).sum()+.5*((flat==v).sum()-1) for v in flat]).to(data.dtype)
        z=torch.distributions.Normal(0,1).icdf((ranks-.375)/(len(flat)+.25)).reshape(4,3,1)
        w=z.var(1,unbiased=True).mean(0)
        b=3*z.mean(1).var(0,unbiased=True)
        return ((2*w/3+b/3)/w).sqrt()
    # Median is the mean of the two middle order statistics for even counts.
    median=split.flatten().sort().values[5:7].mean()
    expected=torch.maximum(oracle(split),oracle((split-median).abs()))
    torch.testing.assert_close(rank_normalized_split_rhat(x),expected)


def test_chain_ess_stationary_ar1_and_shifted_rhat():
    from lafa.diagnostics import effective_sample_size,rank_normalized_split_rhat
    rng=torch.Generator().manual_seed(44)
    noise=torch.randn(4,4100,1,dtype=torch.float64,generator=rng)
    x=noise.clone()
    for t in range(1,x.shape[1]): x[:,t]=.7*x[:,t-1]+noise[:,t]
    x=x[:,100:]
    ess=effective_sample_size(x)
    theoretical=x.shape[0]*x.shape[1]*(1-.7)/(1+.7)
    assert .7*theoretical<ess[0]<1.3*theoretical
    assert rank_normalized_split_rhat(x)[0]<1.02
    shifted=x+torch.arange(4,dtype=x.dtype)[:,None,None]*4
    assert rank_normalized_split_rhat(shifted)[0]>1.2
    assert effective_sample_size(shifted)[0]<ess[0]/10


def test_chain_diagnostics_reject_particles_and_undefined_constant():
    from lafa.diagnostics import effective_sample_size,rank_normalized_split_rhat
    with pytest.raises(ValueError): effective_sample_size(torch.ones(5,3))
    with pytest.raises(ValueError): rank_normalized_split_rhat(torch.ones(1,10,3))
    assert torch.isnan(rank_normalized_split_rhat(torch.ones(2,10,3))).all()


def test_float32_ksd_translation_invariance():
    x=torch.tensor([[-1.,0.],[1.,0.],[0.,1.]],dtype=torch.float32)
    scores=torch.tensor([[.3,-.2],[-.7,.2],[.9,-.6]],dtype=x.dtype)
    baseline=ksd(x,scores,block_size=2)
    shifted=ksd(x+10000,scores,block_size=2)
    torch.testing.assert_close(shifted,baseline,atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('block_size',[1,2,3])
def test_mixed_scale_ksd_pair_residuals(block_size):
    x=torch.tensor([[-1e8],[0.],[1.]],dtype=torch.float32)
    scores=torch.tensor([[0.],[1.],[2.]],dtype=x.dtype)
    # Offdiagonal pair (0,1) of the two nearby points has u=3 exp(-1/2),
    # and diagonal values are 1,2,5; all distant pairs vanish.
    expected=x.new_tensor((8+6*math.exp(-.5))/9)
    torch.testing.assert_close(ksd(x,scores,block_size=block_size),expected,rtol=1e-6,atol=1e-7)


@pytest.mark.parametrize('small_mass',[1e-6,1e-8])
@pytest.mark.parametrize('reverse',[False,True])
def test_ksd_u_stable_positive_pair_mass(small_mass,reverse):
    x=torch.tensor([[-1.],[1.]],dtype=torch.float32)
    weights=x.new_tensor([1.,small_mass])
    if reverse: weights=weights.flip(0)
    torch.testing.assert_close(ksd(x,-x,weights=weights,estimator='u'),x.new_tensor(-8*math.exp(-2)),rtol=1e-6,atol=1e-7)


def test_callable_ksd_score_graphs_respect_block_size():
    calls=[]
    def score(x):
        calls.append(len(x))
        return -x
    x=torch.randn(7,3,dtype=torch.float64,generator=torch.Generator().manual_seed(9))
    expected=ksd(x,-x,block_size=2)
    torch.testing.assert_close(ksd(x,score,block_size=2),expected)
    assert calls and max(calls)<=2


def test_chain_scale_invariance_before_fft_and_variance():
    from lafa.diagnostics import autocorrelation,effective_sample_size
    rng=torch.Generator().manual_seed(44)
    noise=torch.randn(4,4100,1,generator=rng)
    chain=noise.clone()
    for t in range(1,chain.shape[1]): chain[:,t]=.9*chain[:,t-1]+noise[:,t]
    chain=chain[:,100:]
    reference=effective_sample_size(chain)
    for scale in (1e17,1e-25):
        torch.testing.assert_close(effective_sample_size(chain*scale),reference,rtol=2e-5,atol=1e-3)
        torch.testing.assert_close(autocorrelation(chain*scale,max_lag=10),autocorrelation(chain,max_lag=10),rtol=2e-5,atol=1e-6)
    shifted=chain+torch.arange(4,dtype=chain.dtype)[:,None,None]*4
    torch.testing.assert_close(effective_sample_size(shifted*1e17),effective_sample_size(shifted),rtol=2e-5,atol=1e-3)
