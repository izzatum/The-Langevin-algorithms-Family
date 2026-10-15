"""Independent target identities, finite differences and spatial geometry checks."""
import math
import pytest
import torch

DT = torch.float64


def finite_difference(potential, x, eps=1e-6):
    eye = torch.eye(x.numel(), dtype=x.dtype, device=x.device)
    return torch.stack([(potential(x + eps * e) - potential(x - eps * e)) / (2 * eps) for e in eye])


def test_gaussian_reference_and_paper_fixture():
    from lafa.problems import Gaussian, paper_gaussian
    p = Gaussian([1., -2.], [[2., .4], [.4, 1.]])
    x = torch.tensor([.3, -.8], dtype=DT, requires_grad=True)
    grad = torch.autograd.grad(p.potential(x), x)[0]
    torch.testing.assert_close(grad, torch.linalg.solve(p.covariance, x-p.mean))
    torch.testing.assert_close(grad, finite_difference(p.potential, x), atol=1e-8, rtol=1e-7)
    torch.testing.assert_close(p.score(x), -grad)
    assert p.potential(torch.zeros(2, 3, 2, dtype=DT)).shape == (2, 3)
    a = torch.tensor([[2., .5], [.5, 2.]], dtype=DT)
    l = 1e-3 * torch.tensor([[.5, 0.], [2., 0.]], dtype=DT)
    exact = paper_gaussian()
    torch.testing.assert_close(exact.mean, torch.linalg.solve(a.T@a+l.T@l, a.T@torch.ones(2,dtype=DT)))
    g1=torch.Generator().manual_seed(42); g2=torch.Generator().manual_seed(42)
    assert torch.equal(p.sample((7,),generator=g1), p.sample((7,),generator=g2))


def test_mixture_total_covariance_and_stability():
    from lafa.problems import GaussianMixture2D
    p = GaussianMixture2D(weights=[1., 2., 3., 4.])
    assert p.dimension == 2 and torch.isclose(p.weights.sum(), torch.tensor(1.,dtype=DT))
    expected = torch.einsum('k,ki,kj->ij',p.weights,p.means-p.mean,p.means-p.mean)+p.component_covariance
    torch.testing.assert_close(p.covariance, expected)
    x=torch.tensor([100., -200.],dtype=DT,requires_grad=True)
    assert torch.isfinite(p(x))
    torch.testing.assert_close(torch.autograd.grad(p(x),x)[0],finite_difference(p,x),atol=1e-4,rtol=1e-6)
    samples=p.sample((20000,),generator=torch.Generator().manual_seed(15))
    torch.testing.assert_close(samples.mean(0),p.mean,atol=.08,rtol=0.)
    with pytest.raises(ValueError): GaussianMixture2D(weights=[1., 0., 1., 1.])


def test_rosenbrock_quadrature_matches_corrected_paper_moments():
    from lafa.problems import Rosenbrock2D, quadrature_2d
    p=Rosenbrock2D()
    assert abs(p.mean[1].item()-.4004891200336424)<1e-12
    assert abs(p.covariance[1,1].item()-.27026063)<1e-7
    q=quadrature_2d(p,((-2.5,3.),(-1.5,9.)),resolution=240)
    q2=quadrature_2d(p,((-2.5,3.),(-1.5,9.)),resolution=480)
    torch.testing.assert_close(q.mean,p.mean,atol=2e-6,rtol=0.)
    torch.testing.assert_close(q2.covariance,p.covariance,atol=2e-6,rtol=0.)
    torch.testing.assert_close(q.mean,q2.mean,atol=1e-6,rtol=0.)
    assert abs(q2.log_normalizer.item()-p.log_normalizer)<2e-6
    assert q2.boundary_mass < 1e-8


def test_quadrature_cell_area_and_chunking():
    from lafa.problems import quadrature_2d
    flat=lambda x:x.sum(-1)*0
    q=quadrature_2d(flat,((0.,2.),(-1.,3.)),resolution=80,chunk_size=137)
    torch.testing.assert_close(q.mean,torch.tensor([1.,1.],dtype=DT))
    assert abs(q.log_normalizer.item()-math.log(8))<1e-12
    assert set(q.moments)=={'second','third','fourth'}


@pytest.mark.parametrize('shape',[(4,4),(8,8),(16,16)])
def test_sparse_ray_geometry_and_linear_reference(shape):
    from lafa.problems import TraveltimeTomography2D
    p=TraveltimeTomography2D(grid_shape=shape,seed=9)
    assert p.operator.layout==torch.sparse_coo
    assert (p.operator.values()>0).all()
    torch.testing.assert_close(p.forward(torch.ones(p.dimension,dtype=DT)),p.ray_lengths)
    x=torch.linspace(.1,.4,p.dimension,dtype=DT).repeat(3,1).requires_grad_()
    sparse=p.forward(x)
    dense=x@p.operator.to_dense().T
    torch.testing.assert_close(sparse,dense)
    torch.testing.assert_close(torch.autograd.grad(sparse.square().sum(),x)[0],2*dense@p.operator.to_dense())
    assert p.field(x).shape==(3,*shape)
    if shape==(4,4):
        posterior=p.posterior()
        torch.testing.assert_close(torch.autograd.grad(p(posterior.mean.requires_grad_()),posterior.mean)[0],torch.zeros(p.dimension,dtype=DT),atol=1e-9,rtol=0.)
        torch.testing.assert_close(posterior.covariance,posterior.covariance.T)


def test_ray_cell_exact_diagonal_and_outside():
    from lafa.problems import ray_cell_operator
    rays=torch.tensor([[[0.,0.],[1.,1.]],[[-1.,.25],[2.,.25]],[[-1.,2.],[2.,2.]]],dtype=DT)
    a=ray_cell_operator((2,2),rays)
    expected=torch.tensor([[math.sqrt(.5),0.,0.,math.sqrt(.5)],[.5,.5,0.,0.],[0.,0.,0.,0.]],dtype=DT)
    torch.testing.assert_close(a.to_dense(),expected)


def test_nonlinear_spatial_and_reduced_quadrature():
    from lafa.problems import NonlinearTomography2D, quadrature_2d
    p=NonlinearTomography2D(grid_shape=(4,4),seed=4)
    assert p.truth.shape==(16,) and p.field(p.truth).shape==(4,4)
    x=p.truth.detach().clone().requires_grad_()
    torch.testing.assert_close(torch.autograd.grad(p(x),x)[0],finite_difference(p,x),atol=2e-6,rtol=2e-6)
    zero=torch.zeros_like(x)
    assert not torch.allclose(p.forward(x)+p.forward(-x),2*p.forward(zero))
    assert ((p.forward(x)>0)&(p.forward(x)<1)).all()
    r=p.reduced()
    assert r.dimension==2 and r.field(r.truth).shape==(4,4)
    torch.testing.assert_close(r.forward(r.truth),p.forward(p.truth))
    bounds=((-2.,2.),(-2.,2.))
    q=quadrature_2d(r,bounds,resolution=100,chunk_size=700)
    q2=quadrature_2d(r,bounds,resolution=200,chunk_size=1000)
    torch.testing.assert_close(q.mean,q2.mean,atol=3e-5,rtol=0.)
    torch.testing.assert_close(q.covariance,q2.covariance,atol=3e-5,rtol=0.)
    assert q2.boundary_mass<1e-7
    wider=quadrature_2d(r,((-3.,3.),(-3.,3.)),resolution=300,chunk_size=1000)
    torch.testing.assert_close(q2.mean,wider.mean,atol=1e-7,rtol=0.)
    torch.testing.assert_close(q2.covariance,wider.covariance,atol=1e-7,rtol=0.)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable; CPU is not GPU validation')
@pytest.mark.parametrize('name',['GaussianMixture2D','Rosenbrock2D','TraveltimeTomography2D','NonlinearTomography2D'])
def test_cuda_same_input_forward_gradient(name):
    import lafa.problems as problems
    cls=getattr(problems,name)
    cpu=cls(device='cpu'); gpu=cls(device='cuda')
    x=torch.linspace(-.2,.3,2*cpu.dimension,dtype=DT).reshape(2,-1).requires_grad_()
    y=x.detach().cuda().requires_grad_()
    vc=cpu(x); vg=gpu(y)
    torch.testing.assert_close(vc,vg.cpu(),atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(torch.autograd.grad(vc.sum(),x)[0],torch.autograd.grad(vg.sum(),y)[0].cpu(),atol=1e-8,rtol=1e-8)


def test_latent_soft_threshold_and_state_space_kalman():
    from lafa.problems import LatentGaussianLaplace, LinearGaussianStateSpace
    p=LatentGaussianLaplace(observations=[1.,-1.],penalty=2.)
    x=torch.tensor([-.4,.1,1.],dtype=DT)
    torch.testing.assert_close(p.prox_latent(x,.1),torch.tensor([-.2,0.,.8],dtype=DT))
    model=LinearGaussianStateSpace(transition=[[1.]],observation=[[1.]],process_covariance=[[.2]],observation_covariance=[[.3]],initial_mean=[0.],initial_covariance=[[1.]])
    result=model.kalman_filter(torch.tensor([[1.]],dtype=DT))
    # First observation is of initial state (no hidden prediction before t=0).
    torch.testing.assert_close(result.means,torch.tensor([[1/1.3]],dtype=DT))
    torch.testing.assert_close(result.covariances,torch.tensor([[[.3/1.3]]],dtype=DT))
    states,obs=model.simulate(5,seed=7)
    assert states.shape==obs.shape==(5,1)


def test_latent_marginal_likelihood_against_independent_integration():
    from lafa.problems import LatentGaussianLaplace
    p=LatentGaussianLaplace(observations=[.7],noise_scale=.6,penalty=1.4)
    theta=torch.tensor([.2],dtype=DT,requires_grad=True)
    z=torch.linspace(-12.,12.,200001,dtype=DT)
    density=(p.penalty/2)*torch.exp(-p.penalty*z.abs())*torch.exp(-.5*((.7-theta-z)/p.noise_scale).square())/(p.noise_scale*math.sqrt(2*math.pi))
    numerical=torch.trapezoid(density,z).log()
    analytic=p.log_marginal_likelihood(theta)
    torch.testing.assert_close(analytic,numerical,atol=1e-8,rtol=0.)
    assert torch.isfinite(torch.autograd.grad(analytic,theta)[0]).all()
    torch.testing.assert_close(p.latent_map(theta),torch.tensor([0.],dtype=DT))


def test_kalman_multi_time_against_joint_gaussian_conditioning():
    from lafa.problems import LinearGaussianStateSpace
    model=LinearGaussianStateSpace()
    y=torch.tensor([[.5],[-.2],[1.]],dtype=DT)
    result=model.kalman_filter(y)
    # Construct entire joint covariance by independent noise propagation.
    f=model.F.item(); q=model.Q.item(); r=model.R.item()
    load=torch.tensor([[1.,0.,0.],[f,1.,0.],[f*f,f,1.]],dtype=DT)
    sx=load@torch.diag(torch.tensor([1.,q,q],dtype=DT))@load.T
    sy=sx+r*torch.eye(3,dtype=DT)
    expected_mean=sx[-1]@torch.linalg.solve(sy,y[:,0])
    expected_var=sx[-1,-1]-sx[-1]@torch.linalg.solve(sy,sx[:,-1])
    torch.testing.assert_close(result.means[-1,0],expected_mean)
    torch.testing.assert_close(result.covariances[-1,0,0],expected_var)
    distribution=torch.distributions.MultivariateNormal(torch.zeros(3,dtype=DT),sy)
    torch.testing.assert_close(result.log_likelihood,distribution.log_prob(y[:,0]))


@pytest.mark.parametrize('name',['Rosenbrock2D','Banana2D'])
def test_banana_gradient_and_distinct_reference_definition(name):
    import lafa.problems as problems
    p=getattr(problems,name)()
    x=torch.tensor([.4,-.2],dtype=DT,requires_grad=True)
    torch.testing.assert_close(torch.autograd.grad(p(x),x)[0],finite_difference(p,x),atol=1e-8,rtol=1e-7)
    assert p(torch.ones(2,3,2,dtype=DT)).shape==(2,3)
    if name=='Banana2D':
        g=torch.Generator().manual_seed(3)
        samples=p.sample((20000,),generator=g)
        inverse=torch.stack((samples[:,0],samples[:,1]-p.bend*(samples[:,0].square()-1)),dim=-1)
        torch.testing.assert_close(inverse.mean(0),torch.zeros(2,dtype=DT),atol=.03,rtol=0.)
        torch.testing.assert_close(inverse.var(0),torch.ones(2,dtype=DT),atol=.04,rtol=0.)


def test_latent_nonsmooth_joint_prox_per_particle_theta_contract():
    from lafa.problems import LatentGaussianLaplace
    p=LatentGaussianLaplace(observations=[.5,-.8])
    theta=torch.tensor([.2],dtype=DT)
    latent=torch.tensor([[.3,-.2],[1.,-.4],[-1.,2.]],dtype=DT)
    pt,pz=p.prox_joint(theta,latent,.1)
    assert pt.shape==(3,1) and pz.shape==(3,2)
    torch.testing.assert_close(pt,theta.expand(3,1))
    torch.testing.assert_close(pz,p.prox_latent(latent,.1))


@pytest.mark.parametrize('h',[1e-4,.1,2.,20.])
@pytest.mark.parametrize('theta_scale',[math.inf,.8])
def test_full_gaussian_laplace_joint_prox_kkt(h,theta_scale):
    from lafa.problems import LatentGaussianLaplace
    p=LatentGaussianLaplace(observations=[.5,-.8,.1],noise_scale=.6,penalty=1.4,theta_scale=theta_scale)
    theta=torch.tensor([.2],dtype=DT)
    latent=torch.tensor([[.3,-.2,0.],[1.,-.4,-.1],[-1.,2.,.2],[0.,0.,0.]],dtype=DT)
    pt,pz=p.full_joint_prox(theta,latent,h)
    assert pt.shape==(4,1) and pz.shape==latent.shape
    likelihood_residual=(pt+pz-p.observations)/p.noise_scale**2
    theta_kkt=likelihood_residual.sum(-1,keepdim=True)+pt/p.theta_scale**2+(pt-theta)/h
    latent_smooth_kkt=likelihood_residual+(pz-latent)/h
    torch.testing.assert_close(theta_kkt,torch.zeros_like(pt),atol=2e-10,rtol=0.)
    # Subgradient is sign for nonzero coordinates; [-lambda,lambda] at zero.
    active=pz!=0
    torch.testing.assert_close(latent_smooth_kkt[active]+p.penalty*pz[active].sign(),torch.zeros_like(pz[active]),atol=2e-10,rtol=0.)
    assert (latent_smooth_kkt[~active].abs()<=p.penalty+2e-10).all()
    objective=lambda t,z:p.potential(t,z)+((t-theta).square().sum(-1)+(z-latent).square().sum(-1))/(2*h)
    assert (objective(pt,pz)<=objective(theta,latent)+1e-12).all()


@pytest.mark.parametrize('name',['MYIPLA','PIPULA','PIPGLA'])
def test_latent_fixture_direct_proximal_driver_integration(name):
    from lafa.problems import LatentGaussianLaplace
    import lafa.specialized as specialized
    p=LatentGaussianLaplace(observations=[.5,-.8],theta_scale=.8)
    theta=torch.tensor([.2],dtype=DT)
    latent=torch.tensor([[.3,-.2],[1.,-.4],[-1.,2.]],dtype=DT)
    cls=getattr(specialized,name)
    if name=='PIPULA': sampler=cls(p.full_joint_prox,theta,latent,step_size=.03)
    else: sampler=cls(p.smooth_potential,p.prox_joint,theta,latent,step_size=.03)
    result=sampler.step(theta_noise=torch.zeros_like(theta),particle_noise=torch.zeros_like(latent))
    assert result.particles.shape==latent.shape and result.theta.shape==theta.shape
    assert torch.isfinite(result.particles).all() and torch.isfinite(result.theta).all()


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),-1.,0.])
@pytest.mark.parametrize('name,keyword',[('TraveltimeTomography2D','noise_scale'),('TraveltimeTomography2D','prior_scale'),('NonlinearTomography2D','noise_scale'),('NonlinearTomography2D','prior_scale'),('LatentGaussianLaplace','noise_scale'),('LatentGaussianLaplace','penalty')])
def test_proper_problem_scale_rejects_nonfinite_and_nonpositive(name,keyword,bad):
    import lafa.problems as problems
    with pytest.raises(ValueError): getattr(problems,name)(**{keyword:bad})


@pytest.mark.parametrize('name,keyword',[('TraveltimeTomography2D','prior_mean'),('NonlinearTomography2D','offset')])
@pytest.mark.parametrize('bad',[float('nan'),float('inf')])
def test_spatial_problem_finite_offsets(name,keyword,bad):
    import lafa.problems as problems
    with pytest.raises(ValueError): getattr(problems,name)(**{keyword:bad})


def test_mixture_large_finite_weights_and_empty_components():
    from lafa.problems import GaussianMixture2D,NonlinearTomography2D,LatentGaussianLaplace
    mixture=GaussianMixture2D(weights=[1e308]*4)
    torch.testing.assert_close(mixture.weights,torch.full((4,),.25,dtype=DT))
    assert torch.isfinite(mixture(torch.zeros(2,dtype=DT)))
    with pytest.raises(ValueError): GaussianMixture2D(means=torch.empty(0,2,dtype=DT),weights=[])
    with pytest.raises(ValueError): NonlinearTomography2D().reduced(prior_scale=float('inf'))
    with pytest.raises(ValueError): LatentGaussianLaplace(theta_scale=float('nan'))
