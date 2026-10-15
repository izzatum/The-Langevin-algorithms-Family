"""Statistical tests use independent chain endpoints, avoiding fictitious iid ESS.

Six-standard-error tolerances below are fixed from the independent endpoint
budget. Seeds are repeated experiments, not searched for favorable trajectories.
"""
import math
import torch
from lafa.problems import Gaussian

DT=torch.float64
SEEDS=(13,29,47)
CHAINS=4096


def endpoints(cls,p,*,step_size,steps):
    draws=[]
    for seed in SEEDS:
        sampler=cls(p.potential,torch.zeros(CHAINS,p.dimension,dtype=DT),step_size=step_size,generator=torch.Generator().manual_seed(seed))
        for _ in range(steps): result=sampler.step()
        draws.append(result.position)
    return torch.cat(draws)


def test_ula_gaussian_discrete_stationary_covariance():
    from lafa.samplers import ULA
    p=Gaussian([.5,-.7],[[1.,0.],[0.,2.]])
    h=.4
    samples=endpoints(ULA,p,step_size=h,steps=90)
    # Eigenwise solution of Sigma_h=(I-hP)Sigma_h(I-hP)+2hI.
    expected=p.covariance.diag()/(1-h/(2*p.covariance.diag()))
    n=len(samples)
    mean_tolerance=6*torch.sqrt(expected/n)
    variance_tolerance=6*expected*math.sqrt(2/(n-1))
    assert ((samples.mean(0)-p.mean).abs()<mean_tolerance).all()
    assert ((samples.var(0)-expected).abs()<variance_tolerance).all()
    # Budget is large enough to distinguish discretization bias from target truth.
    assert abs(samples.var(0)[0]-p.covariance[0,0])>variance_tolerance[0]


def test_mala_exact_gaussian_moments_multiple_seeds():
    from lafa.samplers import MALA
    p=Gaussian([.5,-.7],[[1.,.5],[.5,2.]])
    samples=endpoints(MALA,p,step_size=.5,steps=100)
    n=len(samples)
    centered=samples-p.mean
    covariance=centered.T@centered/n
    covariance_se=torch.sqrt((p.covariance.diag()[:,None]*p.covariance.diag()[None,:]+p.covariance.square())/n)
    assert ((samples.mean(0)-p.mean).abs()<6*torch.sqrt(p.covariance.diag()/n)).all()
    assert ((covariance-p.covariance).abs()<6*covariance_se).all()


def test_nonlinear_spatial_mala_against_independent_quadrature():
    from lafa.problems import NonlinearTomography2D, quadrature_2d
    from lafa.samplers import MALA
    problem=NonlinearTomography2D(grid_shape=(4,4),seed=4).reduced()
    reference=quadrature_2d(problem,((-2.,2.),(-2.,2.)),resolution=200)
    draws=[]
    for seed in SEEDS:
        sampler=MALA(problem,torch.zeros(1024,2,dtype=DT),step_size=.003,generator=torch.Generator().manual_seed(seed))
        for _ in range(180): final=sampler.step()
        draws.append(final.position)
    samples=torch.cat(draws)
    n=len(samples)
    variance=reference.covariance.diag()
    mu=reference.mean
    # Fourth central moment from independently integrated raw moments.
    moments=reference.moments
    central4=moments['fourth']-4*mu*moments['third']+6*mu.square()*moments['second']-3*mu.pow(4)
    mean_se=(variance/n).sqrt()
    variance_se=((central4-variance.square())/n).sqrt()
    assert ((samples.mean(0)-mu).abs()<6*mean_se).all()
    assert ((samples.var(0)-variance).abs()<6*variance_se).all()
