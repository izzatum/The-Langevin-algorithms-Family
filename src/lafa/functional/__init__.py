"""Pure tensor kernels. Randomness is supplied by callers; inputs must be valid.

step_size broadcasts over coordinates: use (chains,1) for per-chain sizes.
Mobility is fixed positive diagonal; no state-dependent correction is implied.
"""
import torch


def ula_step(position, gradient, step_size, *, temperature=1., mobility=1., noise):
    h = torch.as_tensor(step_size, dtype=position.dtype, device=position.device)
    m = torch.as_tensor(mobility, dtype=position.dtype, device=position.device)
    return position - h*m*gradient + (2*h*temperature*m).sqrt()*noise


def mala_log_acceptance(position, proposal, value, proposal_value, gradient, proposal_gradient,
                        step_size, *, temperature=1., mobility=1.):
    """Unclipped log MH ratio, using the same frozen covariance both ways."""
    h = torch.as_tensor(step_size, dtype=position.dtype, device=position.device)
    m = torch.as_tensor(mobility, dtype=position.dtype, device=position.device)
    forward = proposal-position+h*m*gradient
    reverse = position-proposal+h*m*proposal_gradient
    return (value-proposal_value)/temperature + ((forward.square()-reverse.square())/(4*h*temperature*m)).sum(-1)


def lip_step_size(position, previous_position, gradient, previous_gradient, step_size, alpha,
                  *, mobility=1., lc=1., max_step_size=1.):
    """Safeguarded recurrence; zero displacement retains h; zero slope uses growth cap."""
    dx = torch.linalg.vector_norm(position-previous_position, dim=-1)
    dg = torch.linalg.vector_norm(mobility*(gradient-previous_gradient), dim=-1)
    cap = torch.full_like(step_size, max_step_size)
    slope = lc*dx/dg.clamp_min(torch.finfo(position.dtype).tiny)
    slope = torch.where(dg > 0, slope, cap)
    candidate = torch.minimum(torch.minimum((1+alpha).sqrt()*step_size, slope), cap)
    return torch.where((dx > 0) & torch.isfinite(candidate) & (candidate > 0), candidate, step_size)


def ou_step(momentum, step_size, *, damping=1., mass=1., temperature=1., noise):
    """Exact OU momentum step for dp=-damping*p dt+sqrt(2*damping*mass*T)dW."""
    h = torch.as_tensor(step_size, dtype=momentum.dtype, device=momentum.device)
    decay = torch.exp(-damping*h)
    return decay*momentum + (mass*temperature*(-torch.expm1(-2*damping*h))).sqrt()*noise
