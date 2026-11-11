"""torch.optim adapters sharing the native transition kernels.

ULA accepts existing .grad or a closure. Other adapters require a deterministic
closure that zeroes gradients, evaluates scalar U, calls backward, and returns U.
MALA returns the accepted-state potential. ULA returns its optional closure value.
No module buffers or external RNGs are checkpointed: freeze stochastic/mutable
model behavior yourself. Parameter groups use lr as diagonal covariance scaling;
MALA accepts/rejects the full concatenated vector together. Parameter groups
are fixed at construction; add_param_group afterward is unsupported. Failed or
interrupted steps (any BaseException, including KeyboardInterrupt) restore
parameters, gradients, numerical sampler history, last_result and this
optimizer's generator. Mutable closure state and external RNGs are caller-owned.

MALA, LipULA, LipMALA and BAOAB keep the potential and gradient of the current
state between steps. When the same closure object is passed again (identity, so
a bound method looked up anew does not count) and the parameters are
bit-identical to the state the previous step left, the closure runs once per
step, at the proposal; otherwise the current state is re-evaluated first, as on
the first step. Reuse therefore assumes the closure depends only on this
optimizer's parameters between steps: after changing anything else it reads
(data, other parameters, buffers) behind the same closure object, call
optimizer.invalidate_cache() or pass a new closure object. The last closure
stays referenced, through the sampler target, until the next step or
load_state_dict().

lr changes apply from the next step. Fixed-step adapters reset their step to lr.
LipULA/LipMALA keep the adapted step and multiply it by new/old group-0 lr;
mobility is group lr over group-0 lr, so below the cap each group's effective
step scales by its own lr ratio. The product is clamped at max_step_size, the cap
the adaptation itself enforces. A clamped increase is not undone by a later
decrease, and because lr ratios between groups are always honoured, a clamped
increase of group 0 lowers the other groups' effective steps instead of raising
its own. The alpha history is left alone. The lr the sampler step corresponds to
is saved as the optional checkpoint entry 'sampler_lr'; checkpoints without it
take their saved group-0 lr as that reference.
"""
import copy
import math
import torch
from ..functional import ula_step
from .. import samplers


class ULA(torch.optim.Optimizer):
    sampler_cls = None

    def __init__(self, params, lr=.01, *, temperature=1., generator=None, **sampler_kwargs):
        if isinstance(params,torch.Tensor):
            raise TypeError('params must be an iterable of Tensors or dicts, not a single '+torch.typename(params))
        # Materialize once so duplicate detection happens before Optimizer warnings.
        params=list(params)
        flat=[]
        for entry in params:
            if isinstance(entry,dict):
                # As in torch.optim, a group may hold one bare Tensor; list() would iterate its rows.
                entry['params']=[entry['params']] if isinstance(entry['params'],torch.Tensor) else list(entry['params'])
                flat.extend(entry['params'])
            else:
                flat.append(entry)
        if not flat:
            raise ValueError('at least one parameter is required')
        seen=set()
        for p in flat:
            if not isinstance(p,torch.Tensor) or p.layout != torch.strided or p.dtype not in (torch.float32,torch.float64) or p.numel()==0:
                raise ValueError('parameters must be nonempty dense float32/float64 tensors')
            storage=(p.device,p.untyped_storage().data_ptr())
            if storage in seen:
                raise ValueError('duplicate or aliased parameter storage is unsupported')
            seen.add(storage)
            if p.device != flat[0].device or p.dtype != flat[0].dtype or not torch.isfinite(p).all():
                raise ValueError('parameters must be finite on one device with one dtype')
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('temperature must be finite and positive')
        self._groups_initialized = False
        super().__init__(params,dict(lr=lr))
        self._groups_initialized = True
        self.temperature=float(temperature)
        self.generator=generator if generator is not None else torch.Generator(device=flat[0].device)
        if torch.device(self.generator.device) != flat[0].device:
            raise ValueError('generator device must match parameters')
        if generator is None: self.generator.seed()
        self.sampler_kwargs=dict(sampler_kwargs)
        if self.sampler_cls is None and sampler_kwargs:
            raise ValueError('ULA optimizer accepts lr, temperature and generator; group lr defines mobility')
        self._sampler=None
        self.last_result=None
        self._pending_sampler_state=None
        self._cached_closure=None  # closure whose current-state potential/gradient the sampler holds
        self._sampler_lr=None  # group-0 lr the sampler step size corresponds to
        self._scales()

    def invalidate_cache(self):
        """Re-evaluate the current state on the next step, e.g. after changing data behind the same closure."""
        self._cached_closure=None

    def add_param_group(self, param_group):
        """Groups must be supplied at construction to keep joint state/history fixed."""
        if self._groups_initialized:
            raise ValueError('adding parameter groups after initialization is unsupported; construct a new optimizer')
        return super().add_param_group(param_group)

    def _parameters(self):
        return [p for group in self.param_groups for p in group['params']]

    def _flat(self):
        return torch.cat([p.detach().reshape(-1) for p in self._parameters()])

    def _scales(self):
        rates = tuple(float(group['lr']) for group in self.param_groups)
        if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
            raise ValueError('group lr must be finite and positive')
        # Read group rates every time; cached buffers never contain gradients.
        signature = tuple((id(p), p.shape, p.dtype, p.device) for p in self._parameters())
        key = (rates, signature)
        if getattr(self, '_scale_key', None) != key:
            pieces = [torch.full_like(p, rate).reshape(-1)
                      for group, rate in zip(self.param_groups, rates) for p in group['params']]
            self._scale_buffer = torch.cat(pieces)
            self._scale_key = key
        return self._scale_buffer

    def _grads(self):
        pieces=[]
        for p in self._parameters():
            if p.grad is None or p.grad.layout != torch.strided or p.grad.is_complex():
                raise ValueError('each parameter needs a dense real gradient')
            pieces.append(p.grad.detach().reshape(-1))
        return torch.cat(pieces)

    @torch.no_grad()
    def _assign(self,x,gradient=None):
        offset=0
        for p in self._parameters():
            size=p.numel()
            p.copy_(x[offset:offset+size].view_as(p))
            if gradient is not None:
                p.grad=gradient[offset:offset+size].view_as(p).clone()
            offset+=size

    def step(self,closure=None):
        # Sampler tensors are replaced, never mutated in-place by transitions.
        # A shallow state snapshot therefore supports transactional recovery
        # without cloning all numerical history on every successful step.
        initial = self._flat()
        previous_grads = [None if p.grad is None else p.grad.detach().clone() for p in self._parameters()]
        rng = self.generator.get_state()
        previous_sampler = self._sampler
        sampler_state = None if previous_sampler is None else previous_sampler.__dict__.copy()
        pending = self._pending_sampler_state
        previous_result = self.last_result
        cached_closure, sampler_lr = self._cached_closure, self._sampler_lr
        try:
            return self._step(closure, initial)
        except BaseException:
            # BaseException: an interrupted closure must not leave an unaccepted proposal in the parameters.
            self._assign(initial)
            for p, gradient in zip(self._parameters(), previous_grads):
                p.grad = gradient
            self.generator.set_state(rng)
            self._sampler = previous_sampler
            if previous_sampler is not None:
                previous_sampler.__dict__.clear()
                previous_sampler.__dict__.update(sampler_state)
            self._pending_sampler_state = pending
            self.last_result = previous_result
            self._cached_closure, self._sampler_lr = cached_closure, sampler_lr
            self._sampler_scale_buffer = None
            raise

    @staticmethod
    def _identical(a,b):
        # torch.equal alone equates -0. with 0.; comparing sign bits too makes finite values bitwise equal.
        return torch.equal(a,b) and torch.equal(torch.signbit(a),torch.signbit(b))

    def _step(self,closure,initial):
        if self.sampler_cls is None:
            value=None
            if closure is not None:
                with torch.enable_grad(): value=closure()
            # A closure may deliberately update parameter values.
            x,g=(self._flat() if closure is not None else initial),self._grads()
            if not torch.isfinite(g).all(): raise ValueError('gradients must be finite')
            z=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=self.generator)
            y=ula_step(x,g,self._scales(),temperature=self.temperature,noise=z)
            if not torch.isfinite(y).all(): raise ValueError('nonfinite ULA transition')
            self._assign(y)
            return value
        if closure is None:
            raise ValueError(f'{type(self).__name__} requires a repeatable potential-and-backward closure')
        initial=self._flat()  # read external parameter changes on every call
        # Only a completed step (or the rollback of a failed one) attributes the cache to a closure.
        cached,self._cached_closure=self._cached_closure,None
        optimizer=self
        class ClosureTarget:
            def value_and_grad(self,x):
                optimizer._assign(x)
                with torch.enable_grad(): value=closure()
                if not isinstance(value,torch.Tensor) or value.ndim != 0:
                    raise ValueError('closure must return a scalar tensor potential')
                return value.detach(),optimizer._grads()
        target=ClosureTarget()
        scales=self._scales()
        h=float(self.param_groups[0]['lr'])
        kinetic=issubclass(self.sampler_cls,samplers.BAOAB)
        lip=issubclass(self.sampler_cls,(samplers.LipULA,samplers.LipMALA))
        if kinetic and any(float(group['lr']) != h for group in self.param_groups):
            raise ValueError('BAOAB optimizer requires equal group lr')
        if self._sampler is None:
            # Separate a common h from block mobility; Lip adapts this common h.
            kwargs=dict(self.sampler_kwargs)
            if not kinetic:
                kwargs['mobility']=scales/h
            pending=self._pending_sampler_state
            # A resumed Lip sampler is built with its checkpointed step: after a
            # clamped lr increase, lr itself may exceed max_step_size.
            start=pending['state']['step_size'] if lip and pending is not None else h
            self._sampler=self.sampler_cls(target,initial,step_size=start,temperature=self.temperature,
                                           generator=self.generator,**kwargs)
            if pending is None:
                self._sampler_lr=h  # a fresh sampler steps at lr by construction
            else:
                # Restore history/RNG while retaining freshly evaluated external
                # parameters. A pending checkpoint must not overwrite edits made
                # after load_state_dict and before the first resumed step.
                value,gradient=self._sampler._value,self._sampler._gradient
                self._sampler.load_state_dict(pending)
                self._sampler._position=initial
                self._sampler._value,self._sampler._gradient=value,gradient
                self._pending_sampler_state=None
        else:
            self._sampler.potential_fn=target
            # Same closure object at bit-identical parameters: the sampler's cached
            # potential and gradient are current, so only the proposal is evaluated.
            if closure is not cached or not self._identical(initial,self._sampler._position):
                self._sampler._position=initial
                self._sampler.invalidate_cache()
        if getattr(self, '_sampler_scale_buffer', None) is not scales:
            if not kinetic:
                self._sampler.mobility=scales/h
            if not lip:
                self._sampler.step_size=torch.full_like(self._sampler.step_size,h)
            elif self._sampler_lr is not None and h != self._sampler_lr:
                # Lip keeps its adapted step and follows the group-0 lr ratio.
                # With mobility=scales/h every group's effective step
                # (step_size*mobility) then scales by its own lr ratio. Clamped
                # at max_step_size, the cap the adaptation itself enforces;
                # replaced, not mutated, so a failed step restores the old tensor.
                step_size=(self._sampler.step_size*(h/self._sampler_lr)).clamp(max=self._sampler.max_step_size)
                if not (step_size > 0).all():
                    raise ValueError('lr change underflows the adapted step size')
                self._sampler.step_size=step_size
            self._sampler_lr=h
        self._sampler_scale_buffer=scales
        self.last_result=self._sampler.step()
        self._assign(self.last_result.position,self.last_result.gradient)
        self._cached_closure=closure
        return self.last_result.potential.clone()

    def state_dict(self):
        result=copy.deepcopy(super().state_dict())
        result['lafa']={'schema_version':1,'algorithm':type(self).__name__, 'position':self._flat().clone(),
                        'generator_state':self.generator.get_state().clone(),'temperature':self.temperature,
                        'sampler_kwargs':copy.deepcopy(self.sampler_kwargs),
                        'sampler':self._sampler.state_dict() if self._sampler is not None else copy.deepcopy(self._pending_sampler_state),
                        # Optional key (absent from older schema-1 checkpoints): group-0 lr the sampler step corresponds to.
                        'sampler_lr':self._sampler_lr}
        return result

    def load_state_dict(self,state_dict):
        data=copy.deepcopy(state_dict)
        own=data.pop('lafa')
        x=self._flat()
        if own['schema_version']!=1 or own['algorithm']!=type(self).__name__ or own['position'].shape!=x.shape or own['position'].dtype!=x.dtype or own['position'].device!=x.device:
            raise ValueError('incompatible optimizer checkpoint')
        reference=own.get('sampler_lr')  # optional: absent from checkpoints written before lr tracking
        if reference is not None and (not isinstance(reference,float) or not math.isfinite(reference) or reference <= 0):
            raise ValueError('checkpoint sampler_lr must be None or a finite positive float')
        super().load_state_dict(data)
        self.temperature=own['temperature']
        self.sampler_kwargs=own['sampler_kwargs']
        # Written before bool adapt_steps was rejected: the equal int continues identically.
        if isinstance(self.sampler_kwargs.get('adapt_steps'),bool):
            self.sampler_kwargs['adapt_steps']=int(self.sampler_kwargs['adapt_steps'])
        self._assign(own['position'])
        self.generator.set_state(own['generator_state'])
        self._sampler=None
        self._pending_sampler_state=own['sampler']
        self.last_result=None
        self._sampler_scale_buffer=None
        self._cached_closure=None
        # Checkpoints without the key resume as 'lr unchanged': their saved
        # group-0 lr is the reference, so only a later lr edit rescales a Lip step.
        self._sampler_lr=float(self.param_groups[0]['lr']) if reference is None and own['sampler'] is not None else reference


class MALA(ULA):
    sampler_cls=samplers.MALA

class LipULA(ULA):
    sampler_cls=samplers.LipULA

class LipMALA(ULA):
    sampler_cls=samplers.LipMALA

class BAOAB(ULA):
    sampler_cls=samplers.BAOAB
