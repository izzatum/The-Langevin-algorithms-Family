"""ReplicaExchange target reuse, schedule completion and step-count validation."""
import pytest
import torch
from lafa.targets import value_and_grad
from lafa.specialized import AnnealedLangevin,NoiseAnnealedLangevin,ReplicaExchange,ScheduleComplete

DT=torch.float64


class Well:
    """Elementwise double well counting target batches.

    Only exactly rounded elementwise operations, so a row's value and gradient
    do not depend on its batch position and reuse can be compared bit for bit.
    """
    def __init__(self,tilt=0.):
        self.tilt,self.batches=tilt,0
    def __call__(self,x):
        self.batches+=1
        return (x.square().square()/4-x.square()/2+self.tilt*x).sum(-1)


def replicas(R,seed=0,potential=None):
    generator=torch.Generator().manual_seed(seed)
    x=1.5*torch.randn((R,3),dtype=DT,generator=generator)
    potential=Well() if potential is None else potential
    return ReplicaExchange(potential,x,[1.5**k for k in range(R)],.2,generator=generator),potential


def supplied(R,iteration):
    generator=torch.Generator().manual_seed(1000+iteration)
    pairs=1 if R==2 else len(range(iteration%2,R-1,2))
    return {'noise':torch.randn((R,3),dtype=DT,generator=generator),
            'local_uniforms':torch.rand(R,dtype=DT,generator=generator),
            'swap_uniforms':torch.rand(pairs,dtype=DT,generator=generator)}


def assert_same(actual,expected):
    assert torch.equal(actual.position,expected.position)
    assert torch.equal(actual.stats['local_accepted'],expected.stats['local_accepted'])
    assert torch.equal(actual.stats['swapped'],expected.stats['swapped'])


@pytest.mark.parametrize('R',[2,3,5])
def test_replica_exchange_costs_one_target_batch_per_step(R):
    sampler,potential=replicas(R)
    assert potential.batches==1
    for k in range(1,13):
        sampler.step()
        assert potential.batches==k+1
    # Invalidation is lazy: paid once by the next step, not by the call itself.
    sampler.invalidate_cache()
    assert potential.batches==13
    sampler.step()
    assert potential.batches==15
    sampler.step()
    assert potential.batches==16


@pytest.mark.parametrize('mode',['generator','supplied'])
@pytest.mark.parametrize('R',[2,3,5])
def test_replica_exchange_reuse_equals_reevaluating_twin(R,mode):
    sampler,potential=replicas(R,seed=R)
    twin,twin_potential=replicas(R,seed=R)
    steps=30; accepted=swaps=0
    for k in range(steps):
        kwargs=supplied(R,k) if mode=='supplied' else {}
        twin.invalidate_cache()
        actual,expected=sampler.step(**kwargs),twin.step(**kwargs)
        assert_same(actual,expected)
        accepted+=int(actual.stats['local_accepted'].sum()); swaps+=int(actual.stats['swapped'].sum())
    # Local accepts, local rejects and real swaps all fed later steps.
    assert 0<accepted<steps*R and swaps>=5
    assert potential.batches==steps+1 and twin_potential.batches==2*steps+1


def test_replica_exchange_carried_pair_follows_swapped_rows():
    sampler,_=replicas(5,seed=11)
    reference=Well(); swaps=0
    for _ in range(20):
        result=sampler.step()
        swaps+=int(result.stats['swapped'].sum())
        value,gradient=value_and_grad(reference,result.position)
        assert torch.equal(sampler._cache[1],value) and torch.equal(sampler._cache[2],gradient)
    assert swaps>=5


def test_replica_exchange_load_state_dict_drops_cache_and_continues_exactly():
    first,_=replicas(3,seed=7)
    for _ in range(5):
        first.step()
    checkpoint=first.state_dict()
    assert set(checkpoint)=={'schema_version','algorithm','state','generator_state'}
    assert set(checkpoint['state'])=={'_position','iteration','step_size','temperatures'}
    uninterrupted=[first.step() for _ in range(8)]
    # The receiver carries values of unrelated replicas until the checkpoint arrives.
    second,potential=replicas(3,seed=8)
    for _ in range(2):
        second.step()
    before=potential.batches
    second.load_state_dict(checkpoint)
    assert potential.batches==before
    for k,expected in enumerate(uninterrupted):
        assert_same(second.step(),expected)
        assert potential.batches==before+k+2


def test_replica_exchange_reassigned_potential_is_reevaluated():
    sampler,_=replicas(3,seed=3)
    twin,_=replicas(3,seed=3)
    for _ in range(3):
        sampler.step(); twin.step()
    tilted,twin_tilted=Well(.7),Well(.7)
    sampler.potential=tilted
    twin.potential=twin_tilted; twin.invalidate_cache()
    for _ in range(6):
        assert_same(sampler.step(),twin.step())
    assert tilted.batches==twin_tilted.batches==7


def test_replica_exchange_instance_without_cache_entry_reevaluates():
    # A whole sampler pickled before the cache existed has no _cache entry.
    sampler,potential=replicas(3,seed=9)
    twin,_=replicas(3,seed=9)
    sampler.step(); twin.step()
    del sampler.__dict__['_cache']
    twin.invalidate_cache()
    assert_same(sampler.step(),twin.step())
    assert potential.batches==4


def test_replica_exchange_rejected_call_keeps_carried_evaluation():
    sampler,potential=replicas(3,seed=5)
    twin,_=replicas(3,seed=5)
    sampler.step(); twin.step()
    position=sampler.position
    # Every local move would be accepted, then the swap uniforms are refused.
    bad=supplied(3,1)|{'local_uniforms':torch.zeros(3,dtype=DT),'swap_uniforms':torch.zeros(2,dtype=DT)}
    with pytest.raises(ValueError,match='swap uniforms'):
        sampler.step(**bad)
    assert sampler.iteration==1 and torch.equal(sampler.position,position)
    batches=potential.batches
    twin.invalidate_cache()
    assert_same(sampler.step(**supplied(3,1)),twin.step(**supplied(3,1)))
    assert potential.batches==batches+1
    for _ in range(4):
        twin.invalidate_cache()
        assert_same(sampler.step(),twin.step())


def test_replica_exchange_interrupt_after_position_commit_drops_carried_pair(monkeypatch):
    import lafa.specialized._common as common
    sampler,potential=replicas(4,seed=13)
    twin,_=replicas(4,seed=13)
    for _ in range(3):
        sampler.step(); twin.step()
    real=common.SpecializedResult
    def interrupted(*arguments,**keywords):
        raise KeyboardInterrupt
    # _finish builds the result after it has committed the new position.
    monkeypatch.setattr(common,'SpecializedResult',interrupted)
    with pytest.raises(KeyboardInterrupt):
        sampler.step()
    monkeypatch.setattr(common,'SpecializedResult',real)
    expected=twin.step()
    assert sampler.iteration==4 and torch.equal(sampler.position,expected.position)
    # The pair of the replicas before the interrupted step must not survive it.
    assert sampler._cache is None
    batches=potential.batches
    for _ in range(6):
        twin.invalidate_cache()
        assert_same(sampler.step(),twin.step())
    assert potential.batches==batches+7


def annealed():
    return AnnealedLangevin(lambda x:x.square().sum(-1)/2,torch.tensor([1.,2.],dtype=DT),[2.,1.],.1,
                            steps_per_temperature=2,generator=torch.Generator().manual_seed(1))


def noise_annealed():
    return NoiseAnnealedLangevin(lambda x,sigma:-x/sigma**2,torch.tensor([1.,2.],dtype=DT),[2.,1.,.5],.1,
                                 steps_per_sigma=2,generator=torch.Generator().manual_seed(1))


SCHEDULED=[(annealed,4,'temperature schedule complete'),(noise_annealed,6,'noise schedule complete')]


@pytest.mark.parametrize('factory,total,message',SCHEDULED)
def test_schedule_complete_surfaces_unchanged_from_generator(factory,total,message):
    sampler=factory()
    def chain():
        while True:
            yield sampler.step().position
    seen=[]
    with pytest.raises(ScheduleComplete,match=message) as caught:
        for position in chain():
            seen.append(position)
    assert type(caught.value) is ScheduleComplete and caught.value.__cause__ is None
    assert len(seen)==total
    assert issubclass(ScheduleComplete,RuntimeError) and not issubclass(ScheduleComplete,StopIteration)


@pytest.mark.parametrize('factory,total,message',SCHEDULED)
def test_schedule_complete_is_not_swallowed_by_iterator_protocol(factory,total,message):
    sampler=factory()
    class Steps:
        def __iter__(self):return self
        def __next__(self):return sampler.step().position
    with pytest.raises(ScheduleComplete,match=message):
        list(Steps())
    assert sampler.iteration==total


@pytest.mark.parametrize('factory,total,message',SCHEDULED)
def test_finished_flips_with_last_scheduled_step_and_round_trips(factory,total,message):
    sampler=factory()
    for k in range(total):
        assert sampler.finished is False
        if k==total-1:
            pending=sampler.state_dict()
        sampler.step()
    assert sampler.finished is True and sampler.iteration==total
    position=sampler.position
    with pytest.raises(ScheduleComplete,match=message):
        sampler.step()
    assert sampler.finished and sampler.iteration==total and torch.equal(sampler.position,position)
    with pytest.raises(AttributeError):
        sampler.finished=False
    assert 'finished' not in sampler.state_dict()['state']
    restored=factory(); restored.load_state_dict(sampler.state_dict())
    assert restored.finished
    with pytest.raises(ScheduleComplete,match=message):
        restored.step()
    # Derived from the loaded state in both directions, never latched.
    restored.load_state_dict(pending)
    assert not restored.finished
    assert torch.equal(restored.step().position,position) and restored.finished
    # The exception-free loop the property exists for.
    loop=factory(); count=0
    while not loop.finished:
        loop.step(); count+=1
    assert count==total and torch.equal(loop.position,position)


def test_bool_and_non_positive_step_counts_rejected():
    x=torch.tensor([1.,2.],dtype=DT)
    # A zero count would never advance the stage, so `finished` could not turn True.
    for bad in (True,False,0,-1,1.,'2',None):
        with pytest.raises(ValueError,match='steps_per_temperature must be a positive integer'):
            AnnealedLangevin(lambda x:x.square().sum(-1)/2,x,[2.,1.],steps_per_temperature=bad)
        with pytest.raises(ValueError,match='steps_per_sigma must be a positive integer'):
            NoiseAnnealedLangevin(lambda x,sigma:-x,x,[2.,1.],steps_per_sigma=bad)
    assert AnnealedLangevin(lambda x:x.square().sum(-1)/2,x,[2.,1.],steps_per_temperature=1).steps_per_temperature==1
    assert NoiseAnnealedLangevin(lambda x,sigma:-x,x,[2.,1.],steps_per_sigma=1).steps_per_sigma==1
