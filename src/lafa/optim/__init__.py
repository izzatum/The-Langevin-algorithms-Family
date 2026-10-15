"""torch.optim adapters sharing the native transition kernels.

ULA accepts existing .grad or a closure. Other adapters require a deterministic
closure that zeroes gradients, evaluates scalar U, calls backward, and returns U.
MALA returns the accepted-state potential. ULA returns its optional closure value.
No module buffers or external RNGs are checkpointed: freeze stochastic/mutable
model behavior yourself. Parameter groups use lr as diagonal covariance scaling;
MALA accepts/rejects the full concatenated vector together. Parameter groups
are fixed at construction; add_param_group afterward is unsupported.
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
        pieces=[]
        for group in self.param_groups:
            h=float(group['lr'])
            if not math.isfinite(h) or h <= 0:
                raise ValueError('group lr must be finite and positive')
            pieces.extend(torch.full_like(p,h).reshape(-1) for p in group['params'])
        return torch.cat(pieces)

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
        if self.sampler_cls is None:
            value=None
            if closure is not None:
                with torch.enable_grad(): value=closure()
            x,g=self._flat(),self._grads()
            if not torch.isfinite(g).all(): raise ValueError('gradients must be finite')
            z=torch.randn(x.shape,device=x.device,dtype=x.dtype,generator=self.generator)
            y=ula_step(x,g,self._scales(),temperature=self.temperature,noise=z)
            if not torch.isfinite(y).all(): raise ValueError('nonfinite ULA transition')
            self._assign(y)
            return value
        if closure is None:
            raise ValueError(f'{type(self).__name__} requires a repeatable potential-and-backward closure')
        initial=self._flat()
        old_grads=[None if p.grad is None else p.grad.detach().clone() for p in self._parameters()]
        optimizer=self
        class ClosureTarget:
            def value_and_grad(self,x):
                optimizer._assign(x)
                with torch.enable_grad(): value=closure()
                if not isinstance(value,torch.Tensor) or value.ndim != 0:
                    raise ValueError('closure must return a scalar tensor potential')
                return value.detach(),optimizer._grads()
        target=ClosureTarget()
        try:
            scales=self._scales()
            if self._sampler is None:
                # Separate a common h from block mobility; Lip adapts this common h.
                h=float(self.param_groups[0]['lr'])
                kwargs=dict(self.sampler_kwargs)
                if issubclass(self.sampler_cls,samplers.BAOAB):
                    if not torch.equal(scales,torch.full_like(scales,h)):
                        raise ValueError('BAOAB optimizer requires equal group lr')
                else:
                    kwargs['mobility']=scales/h
                self._sampler=self.sampler_cls(target,initial,step_size=h,temperature=self.temperature,
                                               generator=self.generator,**kwargs)
                if self._pending_sampler_state is not None:
                    self._sampler.load_state_dict(self._pending_sampler_state)
                    self._pending_sampler_state=None
            else:
                self._sampler.potential_fn=target
                self._sampler._position=initial.clone()
                self._sampler.invalidate_cache()
                h=float(self.param_groups[0]['lr'])
                if isinstance(self._sampler,samplers.BAOAB):
                    if not torch.equal(scales,torch.full_like(scales,h)):
                        raise ValueError('BAOAB optimizer requires equal group lr')
                    self._sampler.step_size.fill_(h)
                else:
                    self._sampler.mobility=scales/h
                    if not isinstance(self._sampler,(samplers.LipULA,samplers.LipMALA)):
                        self._sampler.step_size.fill_(h)
            self.last_result=self._sampler.step()
            self._assign(self.last_result.position,self.last_result.gradient)
            return self.last_result.potential.clone()
        except Exception:
            self._assign(initial)
            for p,g in zip(self._parameters(),old_grads): p.grad=g
            raise

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


class MALA(ULA):
    sampler_cls=samplers.MALA

class LipULA(ULA):
    sampler_cls=samplers.LipULA

class LipMALA(ULA):
    sampler_cls=samplers.LipMALA

class BAOAB(ULA):
    sampler_cls=samplers.BAOAB
