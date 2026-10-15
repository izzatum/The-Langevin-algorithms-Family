"""Gaussian KDE-WGD in parameters and finite function coordinates."""
import math
import torch
from lafa.targets import value_and_grad
from lafa.state import StepResult
from lafa.kernels import RBF, validate_ensemble, validate_block_size
from .svgd import SVGD, direction, kde_direction, _validate_score
from .function_space import FunctionSpaceSVGD
class RepulsiveEnsemble(SVGD):
    """D'Angelo--Fortuin (2021) Eq.(15), Gaussian KDE-WGD.

    Own-particle posterior score minus KDE query score, including self mass.
    This differs from the kernel-averaged attraction of SVGD. Fixed bandwidth
    and finite particles do not imply exact posterior sampling.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(self.kernel,RBF): raise ValueError('KDE-WGD requires RBF')

    def _direction(self,x,gradient):
        return kde_direction(x,-gradient,kernel=self.kernel,weights=self.weights,block_size=self.block_size),{}

    def load_state_dict(self,state):
        if state['kernel'].get('kind') != 'rbf': raise ValueError('KDE-WGD requires RBF')
        return super().load_state_dict(state)


class FunctionSpaceRepulsiveEnsemble(FunctionSpaceSVGD):
    """Gaussian KDE-fWGD, D'Angelo--Fortuin Eq.(20); destination VJP."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if not isinstance(self.kernel,RBF): raise ValueError('KDE-fWGD requires RBF')

    def _direction(self,x,gradient):
        z=self._outputs(x)
        _,grad_z=self._batched_value_and_grad(self.function_potential,z)
        field=kde_direction(z,-grad_z,kernel=self.kernel,weights=self.weights,block_size=self.block_size)
        return self._pullback(x,field),{'function_direction_norm':field.norm()}

    def load_state_dict(self,state):
        if state['kernel'].get('kind') != 'rbf': raise ValueError('KDE-fWGD requires RBF')
        return super().load_state_dict(state)
