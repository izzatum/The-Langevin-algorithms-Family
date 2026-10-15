"""Matrix-free published block-diagonal Stein variational Newton."""
import math
import torch
from lafa.targets import value_and_grad
from lafa.state import StepResult
from lafa.kernels import RBF, validate_ensemble, validate_block_size
from .svgd import SVGD, direction, kde_direction, _validate_score
from lafa.kernels import squared_distances
class SteinVariationalNewton(SVGD):
    """Published BLOCK-DIAGONAL SVN, Detommaso et al. Eq.(17).

    Each coefficient solves H_ss alpha_s=phi_s, then W_i=sum_s K_is alpha_s
    WITHOUT another 1/N. H_ss is evaluated matrix-free from potential HVPs and
    source kernel-gradient outer products; no parameter Hessian is allocated.
    Main-text positive transport sign is used (Appendix B sign differs).
    Full coupled cross-block Hessian conventions in the source are ambiguous;
    those cross blocks are outside this explicitly block-diagonal approximation.

    curvature='exact' uses exact U HVPs; ordinary CG requires an SPD block.
    damping adds lambda*I in coefficient coordinates. Negative curvature or
    nonconvergence raises without moving particles. For guaranteed PSD target
    curvature use curvature='gauss_newton' and provide curvature_hvp(x,v), which
    returns batched per-particle PSD likelihood-plus-prior HVPs; it is a caller
    contract that this operator is symmetric PSD. Positive damping then makes
    the blocks SPD. CG checks the actual configured-operator residual.
    """
    approximation='block_diagonal'

    def __init__(self,potential,initial,*,damping=1e-6,cg_tolerance=1e-6,
                 cg_absolute_tolerance=1e-10,cg_max_iterations=100,
                 curvature='exact',curvature_hvp=None,**kwargs):
        if not math.isfinite(damping) or damping<0: raise ValueError('damping must be finite and nonnegative')
        if not math.isfinite(cg_tolerance) or cg_tolerance<=0: raise ValueError('cg_tolerance must be positive')
        if not math.isfinite(cg_absolute_tolerance) or cg_absolute_tolerance<0: raise ValueError('cg_absolute_tolerance must be nonnegative')
        validate_block_size(cg_max_iterations)
        if curvature not in ('exact','gauss_newton'): raise ValueError('curvature must be exact or gauss_newton')
        if curvature=='gauss_newton' and curvature_hvp is None: raise ValueError('gauss_newton requires curvature_hvp')
        if curvature=='exact' and curvature_hvp is not None: raise ValueError('curvature_hvp requires gauss_newton label')
        super().__init__(potential,initial,**kwargs)
        self.damping,self.cg_tolerance,self.cg_absolute_tolerance=damping,cg_tolerance,cg_absolute_tolerance
        self.cg_max_iterations,self.curvature,self.curvature_hvp=cg_max_iterations,curvature,curvature_hvp

    def _target_hvp(self,x,v):
        if self.curvature_hvp is not None:
            result=self.curvature_hvp(x.detach(),v.detach())
            _validate_score(x,result)
            return result.detach()
        with torch.enable_grad():
            leaf=x.detach().clone().requires_grad_(True)
            value=self.potential(leaf)
            if not isinstance(value,torch.Tensor) or value.shape!=x.shape[:-1]: raise ValueError('potential must return batched values for HVP')
            grad=torch.autograd.grad(value.sum(),leaf,create_graph=True,allow_unused=True)[0] if value.requires_grad else None
            result=None
            if grad is not None and grad.requires_grad:
                result=torch.autograd.grad((grad*v).sum(),leaf,allow_unused=True)[0]
            if result is None: result=torch.zeros_like(x)
        _validate_score(x,result)
        return result.detach()

    def block_operator(self,position,center,vector):
        """Apply configured H_ss+lambda I to one vector (also useful for audit)."""
        x=position.detach()
        s=x[center]
        out=self.damping*vector
        for l in range(0,len(x),self.block_size):
            xl=x[l:l+self.block_size]
            with torch.no_grad():
                k,b,_=self.kernel.radial(squared_distances(s[None],xl),x.shape[1])
                a=(s-xl)*b[0,:,None]
                masses=self.weights[l:l+self.block_size]
            hv=self._target_hvp(xl,vector.expand_as(xl))
            out=out+(masses*k[0].square())@hv+(masses*(a@vector))@a
        if not torch.isfinite(out).all(): raise RuntimeError('nonfinite SVN curvature operator')
        return out.detach()

    def _cg(self,operator,rhs):
        solution=torch.zeros_like(rhs)
        residual=rhs.clone()
        norm=rhs.norm()
        threshold=max(self.cg_absolute_tolerance,self.cg_tolerance*float(norm))
        if norm<=threshold: return solution,0,norm,norm/ norm.clamp_min(torch.finfo(rhs.dtype).tiny),0
        search=residual.clone(); rr=residual@residual
        calls=0
        for iteration in range(1,self.cg_max_iterations+1):
            applied=operator(search); calls+=1
            curvature=search@applied
            if not torch.isfinite(curvature) or curvature<=0:
                raise RuntimeError('SVN CG encountered nonpositive curvature; configure damping or PSD curvature')
            alpha=rr/curvature
            solution=solution+alpha*search
            residual=residual-alpha*applied
            new_rr=residual@residual
            if residual.norm()<=threshold:
                actual=rhs-operator(solution); calls+=1
                absolute=actual.norm(); relative=absolute/norm
                if absolute<=threshold:
                    return solution,iteration,absolute,relative,calls
                # Residual replacement/restart avoids claiming recursive-only convergence.
                residual=actual; new_rr=residual@residual; search=residual.clone(); rr=new_rr
                continue
            search=residual+(new_rr/rr)*search
            rr=new_rr
        actual=(rhs-operator(solution)).norm()
        raise RuntimeError(f'SVN CG failed to converge: actual residual={float(actual):.3g}, tolerance={threshold:.3g}')

    def _direction(self,x,gradient):
        rhs=direction(x,-gradient,kernel=self.kernel,weights=self.weights,block_size=self.block_size)
        coefficients=[];counts=[];absolutes=[];relatives=[];operator_calls=0
        for s in range(len(x)):
            alpha,count,absolute,relative,calls=self._cg(lambda v:self.block_operator(x,s,v),rhs[s])
            coefficients.append(alpha);counts.append(count);absolutes.append(absolute);relatives.append(relative);operator_calls+=calls
        coefficients=torch.stack(coefficients)
        descent=(rhs*coefficients).sum()
        if rhs.square().sum()>self.cg_absolute_tolerance**2 and descent<0:
            raise RuntimeError('SVN coefficient direction failed descent check')
        field=torch.zeros_like(x)
        with torch.no_grad():
            for i in range(0,len(x),self.block_size):
                for j in range(0,len(x),self.block_size):
                    field[i:i+self.block_size]+=self.kernel(x[i:i+self.block_size],x[j:j+self.block_size])@coefficients[j:j+self.block_size]
        stats={'cg_iterations':torch.tensor(counts,device=x.device,dtype=torch.int64),
               'cg_absolute_residual':torch.stack(absolutes),'cg_relative_residual':torch.stack(relatives),
               'cg_operator_calls':x.new_tensor(operator_calls,dtype=torch.int64),
               'cg_solves':x.new_tensor(len(x),dtype=torch.int64),
               'cg_converged':torch.ones(len(x),device=x.device,dtype=torch.bool),
               'target_hvp_calls':x.new_tensor(operator_calls*math.ceil(len(x)/self.block_size),dtype=torch.int64),
               'damping':x.new_tensor(self.damping),
               'coefficient_descent':descent.detach()}
        return field,stats

    def _extra_state(self):
        return dict(damping=self.damping,cg_tolerance=self.cg_tolerance,cg_absolute_tolerance=self.cg_absolute_tolerance,
                    cg_max_iterations=self.cg_max_iterations,curvature=self.curvature,approximation=self.approximation)

    def _load_extra(self,extra,x):
        if extra.get('approximation')!=self.approximation: raise ValueError('SVN approximation mismatch')
        if extra['curvature']!=self.curvature: raise ValueError('reattach the matching curvature callback/mode before restore')
        for key in ('damping','cg_absolute_tolerance'):
            if not math.isfinite(extra[key]) or extra[key]<0: raise ValueError('invalid solver checkpoint')
        if not math.isfinite(extra['cg_tolerance']) or extra['cg_tolerance']<=0: raise ValueError('invalid solver tolerance')
        validate_block_size(extra['cg_max_iterations'])
        for key in ('damping','cg_tolerance','cg_absolute_tolerance','cg_max_iterations'): setattr(self,key,extra[key])
