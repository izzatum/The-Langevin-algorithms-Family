"""Likelihood-informed pSVGD for isotropic Gaussian prior coordinates."""
import math
import torch
from lafa.targets import value_and_grad
from lafa.state import StepResult
from lafa.kernels import RBF, validate_ensemble, validate_block_size
from .svgd import SVGD, direction, kde_direction, _validate_score
class ProjectedSVGD(SVGD):
    """Chen--Ghattas practical pSVGD with N(0,I) prior (whitened coordinates).

    log_likelihood is the batched LOG LIKELIHOOD, excluding prior. The empirical
    likelihood-gradient information basis is orthonormal. One frozen complement
    per particle supplies the paper's approximate profile score; this is NOT an
    unbiased estimator of grad log E_b[likelihood]. Initial particles must supply
    prior-distributed complements (typically initialize from N(0,I)); no silent
    zeroing or resampling occurs. At basis refresh, reproject CURRENT positions.
    General priors require the caller's explicit whitening and transformed
    likelihood. This implementation does not claim the optimal-profile KL bound.

    basis_refresh=None fixes the basis; a positive integer gives outer refresh
    frequency. The thin SVD stores O(ND) work, never an information Hessian.
    """
    approximation = 'frozen_complement_profile_score'

    def __init__(self,log_likelihood,initial,*,rank,basis_refresh=None,**kwargs):
        if isinstance(rank,bool) or not isinstance(rank,int) or not 1<=rank<=initial.shape[-1]:
            raise ValueError('rank must be in [1,D]')
        if basis_refresh is not None: validate_block_size(basis_refresh)
        self.log_likelihood,self.rank,self.basis_refresh = log_likelihood,rank,basis_refresh
        self.basis_refresh_count=0
        super().__init__(lambda x:x.square().sum(-1)/2-log_likelihood(x),initial,**kwargs)
        self._refresh_basis()

    def _likelihood_gradient(self,x):
        _,gradient=self._batched_value_and_grad(lambda y:-self.log_likelihood(y),x)
        _validate_score(x,gradient)
        return -gradient

    def _refresh_basis(self):
        x=self._position
        h=self._likelihood_gradient(x)
        weighted=h*self.weights.sqrt()[:,None]
        _,singular,vh=torch.linalg.svd(weighted,full_matrices=False)
        basis=vh[:self.rank].T
        if basis.shape[1]<self.rank:
            # Only needed when rank>N. Complete the null eigenspace with a
            # deterministic orthonormal basis. Requested O(D*rank) state remains.
            basis=torch.linalg.qr(torch.cat((basis,torch.eye(x.shape[1],self.rank,device=x.device,dtype=x.dtype)),dim=1),mode='reduced').Q[:,:self.rank]
        self.basis=basis.detach()
        self.eigenvalues=x.new_zeros((self.rank,))
        self.eigenvalues[:min(self.rank,len(singular))]=singular[:self.rank].square()
        self.coefficients=x@self.basis
        self.complement=(x-self.coefficients@self.basis.T).detach()
        self.basis_refresh_count+=1
        residual=weighted.T@(weighted@self.basis)-self.basis*self.eigenvalues
        self.basis_residual=residual.norm().detach()

    def step(self):
        if self.basis_refresh is not None and self.iteration and self.iteration%self.basis_refresh==0:
            self._refresh_basis()
        h=self._likelihood_gradient(self._position)
        coeff_score=h@self.basis-self.coefficients
        phi=direction(self.coefficients,coeff_score,kernel=self.kernel,weights=self.weights,block_size=self.block_size)
        coefficients=(self.coefficients+self.step_size*phi).detach()
        candidate=coefficients@self.basis.T+self.complement
        validate_ensemble(candidate)
        values,gradient=self._evaluate(candidate)
        self.coefficients=coefficients
        self._position=candidate.detach()
        self.iteration+=1
        stats={'iteration':candidate.new_tensor(self.iteration,dtype=torch.int64),
               'basis_residual':self.basis_residual.clone(),
               'basis_refresh_count':candidate.new_tensor(self.basis_refresh_count,dtype=torch.int64)}
        return StepResult(self.position,values.clone(),gradient.clone(),stats=stats)

    def _extra_state(self):
        return dict(rank=self.rank,basis_refresh=self.basis_refresh,basis_refresh_count=self.basis_refresh_count,
                    basis=self.basis.clone(),eigenvalues=self.eigenvalues.clone(),
                    coefficients=self.coefficients.clone(),complement=self.complement.clone(),
                    basis_residual=self.basis_residual.clone(),approximation=self.approximation)

    def _load_extra(self,extra,x):
        rank=extra['rank']
        if not isinstance(rank,int) or not 1<=rank<=x.shape[1]: raise ValueError('invalid projection rank')
        refresh=extra['basis_refresh']
        if refresh is not None: validate_block_size(refresh)
        converted={key:extra[key].to(device=x.device,dtype=x.dtype).detach().clone() for key in ('basis','eigenvalues','coefficients','complement','basis_residual')}
        shapes={'basis':(x.shape[1],rank),'eigenvalues':(rank,),'coefficients':(len(x),rank),'complement':x.shape,'basis_residual':()}
        if any(converted[k].shape!=shape or not torch.isfinite(converted[k]).all() for k,shape in shapes.items()):
            raise ValueError('invalid projection checkpoint shapes or values')
        if not torch.allclose(converted['basis'].T@converted['basis'],torch.eye(rank,device=x.device,dtype=x.dtype),rtol=1e-4,atol=1e-5):
            raise ValueError('checkpoint basis is not orthonormal')
        if not torch.allclose(converted['coefficients']@converted['basis'].T+converted['complement'],x):
            raise ValueError('checkpoint projection does not reconstruct position')
        self.rank,self.basis_refresh=rank,refresh
        self.basis_refresh_count=extra['basis_refresh_count']
        for key,value in converted.items(): setattr(self,key,value)
