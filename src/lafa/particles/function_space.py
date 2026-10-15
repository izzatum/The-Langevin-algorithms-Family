"""Function-space SVGD with destination parameter VJPs."""
import math
import torch
from lafa.targets import value_and_grad, _validate_chunk_size, _transform_error
from lafa.state import StepResult
from lafa.kernels import RBF, validate_ensemble, validate_block_size
from .svgd import SVGD, direction, kde_direction, _validate_score
class FunctionSpaceSVGD(SVGD):
    """Wang et al. (2019) Eq.(3)/(4), finite function-space destination VJP.

    function_map maps ONE parameter vector to a fixed ordered 1D output vector.
    It must be deterministic, differentiable and use the same evaluation inputs
    and buffers throughout each step. function_potential accepts batched outputs
    and specifies the entire finite function posterior, including an explicit
    function prior. A weight prior gradient is not a function prior score.
    from_score(score_function) may supply an analytic function posterior score.

    Frozen function transport is pulled back by J_i.T at destination i. It is
    NOT an inverse-Jacobian update: induced output drift is J_i J_i.T phi_i.
    checkpoint restore requires the same function map and function target.
    ``vectorized=True`` uses vmap for function values and destination VJPs;
    ``chunk_size`` bounds the number of particles transformed at once. The map
    must support torch.func transforms and must not mutate captured state.
    Unsupported vmap operations require the explicit default loop mode.
    """
    def __init__(self,function_map,function_potential,initial,*,vectorized=False,chunk_size=None,**kwargs):
        if not isinstance(vectorized, bool): raise ValueError('vectorized must be a boolean')
        _validate_chunk_size(chunk_size)
        self.vectorized,self.chunk_size = vectorized,chunk_size
        self.function_map,self.function_potential = function_map,function_potential
        super().__init__(function_potential,initial,**kwargs)
        self._outputs(initial)

    def _outputs(self,x):
        with torch.no_grad():
            if self.vectorized:
                size = self.chunk_size or len(x)
                try:
                    z = torch.cat([torch.vmap(self.function_map)(x[i:i+size].detach().clone())
                                   for i in range(0,len(x),size)])
                except RuntimeError as error:
                    _transform_error(error, 'vectorized=False')
            else:
                z = torch.stack([self.function_map(row.detach().clone()) for row in x])
        validate_ensemble(z)
        if z.dtype != x.dtype or z.device != x.device: raise ValueError('function_map must preserve dtype and device')
        return z.detach().clone()

    def _pullback(self,x,cotangents):
        if self.vectorized:
            def pullback(row,cotangent):
                _,vjp = torch.func.vjp(self.function_map,row)
                return vjp(cotangent)[0]
            size = self.chunk_size or len(x)
            try:
                with torch.enable_grad():
                    result=torch.cat([torch.vmap(pullback)(x[i:i+size].detach().clone(),cotangents[i:i+size].detach().clone()).detach()
                                      for i in range(0,len(x),size)])
            except RuntimeError as error:
                _transform_error(error, 'vectorized=False')
            _validate_score(x,result)
            return result
        result=[]
        for row,cotangent in zip(x,cotangents):
            with torch.enable_grad():
                _,vjp = torch.func.vjp(self.function_map,row.detach().clone())
                result.append(vjp(cotangent.detach())[0].detach())
        result=torch.stack(result)
        _validate_score(x,result)
        return result

    def _evaluate(self,x):
        z = self._outputs(x)
        values,gradient = self._batched_value_and_grad(self.function_potential,z)
        if not getattr(self.function_potential,'score_only',False) and not torch.isfinite(values).all():
            raise ValueError('nonfinite function potential values')
        _validate_score(z,gradient)
        return values,self._pullback(x,gradient)

    def _direction(self,x,gradient):
        z = self._outputs(x)
        _,grad_z = self._batched_value_and_grad(self.function_potential,z)
        phi = direction(z,-grad_z,kernel=self.kernel,weights=self.weights,block_size=self.block_size)
        return self._pullback(x,phi),{'function_direction_norm':phi.norm()}

    def _extra_state(self):
        return {'vectorized':self.vectorized,'chunk_size':self.chunk_size}

    def _load_extra(self,extra,x):
        # Earlier function-space checkpoints used empty metadata.
        if not extra: return
        if set(extra) != {'vectorized','chunk_size'} or not isinstance(extra['vectorized'],bool):
            raise ValueError('invalid function-space checkpoint metadata')
        _validate_chunk_size(extra['chunk_size'])
        self.vectorized,self.chunk_size = extra['vectorized'],extra['chunk_size']
