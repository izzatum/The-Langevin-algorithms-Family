"""Deterministic interacting posterior approximations; not IID MCMC chains."""
from .svgd import SVGD, WeightSpaceSVGD, direction, kde_direction
from .projected import ProjectedSVGD
from .newton import SteinVariationalNewton
from .function_space import FunctionSpaceSVGD
from .repulsive import RepulsiveEnsemble, FunctionSpaceRepulsiveEnsemble

__all__ = ['SVGD','WeightSpaceSVGD','direction','kde_direction','ProjectedSVGD',
           'SteinVariationalNewton','FunctionSpaceSVGD','RepulsiveEnsemble',
           'FunctionSpaceRepulsiveEnsemble']
