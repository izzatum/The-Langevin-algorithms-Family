"""torch.optim adapters sharing the native transition kernels.

ULA accepts existing .grad or a closure. Other adapters require a deterministic
closure that zeroes gradients, evaluates scalar U, calls backward, and returns U.
MALA returns the accepted-state potential. ULA returns its optional closure value.
No module buffers or external RNGs are checkpointed: freeze stochastic/mutable
model behavior yourself. Parameter groups use lr as diagonal covariance scaling;
MALA accepts/rejects the full concatenated vector together. Parameter groups
are fixed at construction; add_param_group afterward is unsupported. Failed steps
restore parameters, gradients, numerical sampler history, last_result and this
optimizer's generator. Mutable closure state and external RNGs are caller-owned.
"""
import copy
import math
import torch
from ..functional import ula_step
from .. import samplers


class ULA(torch.optim.Optimizer):
    sampler_cls = None

    def __init__(self, params, lr=.01, *, temperature=1., generator=None, **sampler_kwargs):
        # Materialize once so duplicate detection happens before Optimizer warnings.
        params=list(params)
        flat=[]
        for entry in params:
            if isinstance(entry,dict):
                entry['params']=list(entry['params'])
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
        self._scales()

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
        try:
            return self._step(closure, initial)
        except Exception:
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
            self._sampler_scale_buffer = None
            raise

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
        if kinetic and any(float(group['lr']) != h for group in self.param_groups):
            raise ValueError('BAOAB optimizer requires equal group lr')
        if self._sampler is None:
            # Separate a common h from block mobility; Lip adapts this common h.
            kwargs=dict(self.sampler_kwargs)
            if not kinetic:
                kwargs['mobility']=scales/h
            self._sampler=self.sampler_cls(target,initial,step_size=h,temperature=self.temperature,
                                           generator=self.generator,**kwargs)
            if self._pending_sampler_state is not None:
                # Restore history/RNG while retaining freshly evaluated external
                # parameters. A pending checkpoint must not overwrite edits made
                # after load_state_dict and before the first resumed step.
                value,gradient=self._sampler._value,self._sampler._gradient
                self._sampler.load_state_dict(self._pending_sampler_state)
                self._sampler._position=initial
                self._sampler._value,self._sampler._gradient=value,gradient
                self._pending_sampler_state=None
        else:
            self._sampler.potential_fn=target
            self._sampler._position=initial
            self._sampler.invalidate_cache()
        if getattr(self, '_sampler_scale_buffer', None) is not scales:
            if not kinetic:
                self._sampler.mobility=scales/h
            if not isinstance(self._sampler,(samplers.LipULA,samplers.LipMALA)):
                self._sampler.step_size=torch.full_like(self._sampler.step_size,h)
        self._sampler_scale_buffer=scales
        self.last_result=self._sampler.step()
        self._assign(self.last_result.position,self.last_result.gradient)
        return self.last_result.potential.clone()

    def state_dict(self):
        result=copy.deepcopy(super().state_dict())
        result['lafa']={'schema_version':1,'algorithm':type(self).__name__, 'position':self._flat().clone(),
                        'generator_state':self.generator.get_state().clone(),'temperature':self.temperature,
                        'sampler_kwargs':copy.deepcopy(self.sampler_kwargs),
                        'sampler':self._sampler.state_dict() if self._sampler is not None else copy.deepcopy(self._pending_sampler_state)}
        return result

    def load_state_dict(self,state_dict):
        data=copy.deepcopy(state_dict)
        own=data.pop('lafa')
        x=self._flat()
        if own['schema_version']!=1 or own['algorithm']!=type(self).__name__ or own['position'].shape!=x.shape or own['position'].dtype!=x.dtype or own['position'].device!=x.device:
            raise ValueError('incompatible optimizer checkpoint')
        super().load_state_dict(data)
        self.temperature=own['temperature']
        self.sampler_kwargs=own['sampler_kwargs']
        self._assign(own['position'])
        self.generator.set_state(own['generator_state'])
        self._sampler=None
        self._pending_sampler_state=own['sampler']
        self.last_result=None
        self._sampler_scale_buffer=None


class MALA(ULA):
    sampler_cls=samplers.MALA

class LipULA(ULA):
    sampler_cls=samplers.LipULA

class LipMALA(ULA):
    sampler_cls=samplers.LipMALA

class BAOAB(ULA):
    sampler_cls=samplers.BAOAB
