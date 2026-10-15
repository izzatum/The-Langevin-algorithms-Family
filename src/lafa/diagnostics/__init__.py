"""Diagnostics with explicit empirical-particle versus independent-chain scope."""
from .stein import ksd
from .chains import autocorrelation, effective_sample_size, rank_normalized_split_rhat
__all__ = ['ksd','autocorrelation','effective_sample_size','rank_normalized_split_rhat']
