"""Report contracts and observed work; no one-seed convergence assertions."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest
import torch
from lafa.problems import Gaussian, LatentGaussianLaplace, LinearGaussianStateSpace

BENCHMARKS=Path(__file__).resolve().parents[1]/'benchmarks'
# The scripts remain directly runnable without making benchmarks a runtime package.
spec=importlib.util.spec_from_file_location('lafa_benchmark_accuracy',BENCHMARKS/'accuracy.py')
accuracy=importlib.util.module_from_spec(spec); spec.loader.exec_module(accuracy)
sys.modules.setdefault('accuracy',accuracy)
spec=importlib.util.spec_from_file_location('lafa_benchmark_families',BENCHMARKS/'families.py')
families=importlib.util.module_from_spec(spec); spec.loader.exec_module(families)


def args():
    return SimpleNamespace(device=torch.device('cpu'),dtype=torch.float64,chains=3,
                           warmup=3,record_every=2,iterations=9)


def gaussian(): return Gaussian([.6,-.4],[[1.,.3],[.3,.7]])


def test_json_undefined_is_null_with_path_reason():
    report=accuracy.strict_report(dict(diagnostics=torch.tensor([float('nan'),float('inf'),1.])))
    encoded=json.dumps(report,allow_nan=False)
    decoded=json.loads(encoded)
    assert decoded['diagnostics']==[None,None,1.]
    assert set(decoded['undefined_values'])=={'$.diagnostics[0]','$.diagnostics[1]'}
    assert all(decoded['undefined_values'].values())


def test_existing_accuracy_preserves_chain_time_axes(monkeypatch):
    captured=[]
    def diagnose(chains):
        captured.append(chains.clone())
        return dict(applicable=True,input_shape=list(chains.shape))
    monkeypatch.setattr(accuracy,'chain_diagnostics',diagnose)
    from lafa.samplers import ULA
    report,samples=accuracy.experiment(gaussian(),ULA,.04,args(),7,gaussian())
    assert report['diagnostics']['input_shape']==[3,5,2]
    torch.testing.assert_close(samples,captured[0].reshape(-1,2))
    assert report['work']['gradient_points']==(1+3+9)*3
    assert report['work']['target_batches']==1+3+9


@pytest.mark.parametrize('method',['ULA','MALA','BAOAB','AnchoredLangevin','HOLA'])
def test_actual_posterior_work_and_diagnostic_axes(method):
    a=args(); steps=9; total=a.warmup+steps
    row=families.posterior_run(gaussian(),method,'test',.04,steps,a,7)
    assert row['diagnostics']['input_shape']==[3,5,2]
    work=row['work']
    if method=='HOLA':
        assert work['target_batches']==0
        assert work['gradient_points']==total*3
        assert work['hvp_calls']==total*3*3
        assert work['grad_laplacian_calls']==total*3
    elif method=='AnchoredLangevin':
        assert work['target_batches']==2*(total+1)
        assert work['gradient_batches']==total+1
        assert work['gradient_points']==(total+1)*3
    else:
        assert work['target_batches']==total+1
        assert work['gradient_batches']==total+1
        assert work['gradient_points']==(total+1)*3
    assert row['seconds']>0


def test_work_snapshot_does_not_change_on_later_evaluation():
    counted=accuracy.CountedPotential(lambda x:x.square().sum(-1))
    x=torch.ones((4,2),dtype=torch.float64,requires_grad=True)
    counted(x); snapshot=counted.snapshot()
    with torch.no_grad(): counted(x)
    assert snapshot['target_points']==4
    assert counted.snapshot()['target_points']==8
    assert snapshot['gradient_points']==4
    assert counted.snapshot()['gradient_points']==4


@pytest.mark.parametrize('method',['MYIPLA','PIPGLA','PIPULA'])
def test_learning_counts_and_integrated_reference_objective(method):
    problem=LatentGaussianLaplace(observations=[-.2,.7,1.1,1.4])
    reference=families.learning_reference(problem)
    theta=reference['theta'].clone().requires_grad_()
    derivative=torch.autograd.grad(-problem.log_marginal_likelihood(theta),theta)[0]
    assert derivative.abs().max()<1e-12
    row=families.learning_run(problem,reference,method,4,9,args(),7)
    assert row['work']['prox_batches']==12
    assert row['work']['prox_points']==48
    assert row['work']['gradient_batches']==(0 if method=='PIPULA' else 12)
    assert row['objective']=='shared_parameter_learning'
    assert not row['diagnostics']['applicable']
    assert row['diagnostics']['undefined_reasons']['all']


def test_birth_death_counts_include_all_rate_recomputations():
    row=families.empirical_run(gaussian(),'BirthDeathLangevin','ensembles_hybrid',4,3,args(),7)
    assert row['work']['target_batches']==3*(1+4)
    assert row['work']['gradient_batches']==3
    assert row['work']['target_points']==3*(1+4)*4
    assert row['work']['ordered_pair_evaluations']==3*4**3
    assert not row['diagnostics']['applicable']


def test_filter_counts_and_kalman_comparison_are_sequential():
    model=LinearGaussianStateSpace()
    _,observations=model.simulate(2,seed=410)
    reference=model.kalman_filter(observations)
    row=families.filtering_run(model,observations,reference,3,4,args(),7)
    work=row['work']
    assert work['gradient_batches']==4
    assert work['gradient_points']==12
    assert work['conditional_prior_analytic_gradient_points']==12
    assert work['transition_batches']==1+4*(3+1)
    assert work['transition_points']==3+4*(3*6+3)
    assert row['metrics']['means'].shape==(2,1)
    assert row['objective']=='sequential_filtering'
    assert not row['diagnostics']['applicable']


def test_complete_small_report_has_six_families_and_seed_uncertainty(tmp_path):
    data=families.main(['--output',str(tmp_path),'--steps','4','8','--step-sizes','.04',
                        '--particles','2','3','--chains','2','--warmup','2','--seeds','7','19','--no-figures'])
    report=json.loads((tmp_path/'families.json').read_text(),parse_constant=lambda value:pytest.fail(value))
    assert {r['family'] for r in report['runs']}==set(families.FAMILIES)
    assert {r['objective'] for r in report['runs']}=={'posterior_moments','final_empirical_distribution',
                'shared_parameter_learning','diagnostic_discrepancy','sequential_filtering'}
    assert all(r['repetitions']==2 and r['error_seed_sd'] is not None for r in report['summaries'])
    assert all(r['seconds']>0 and r['work']['hvp_calls']>=0 for r in report['runs'])
    diagnostic=next(r for r in data['runs'] if r['method']=='IMQ KSD direct iid')
    assert diagnostic['work']['ordered_pair_evaluations']==2*diagnostic['budget']['direct_draws']**2
    assert report['metadata']['cuda_executed'] is False


@pytest.mark.parametrize('driver',[accuracy,families])
def test_unavailable_cuda_rejected_before_output(driver,monkeypatch,tmp_path):
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    output=tmp_path/'must-not-exist'
    with pytest.raises(SystemExit) as exc:
        driver.main(['--device','cuda','--output',str(output)])
    assert exc.value.code==2
    assert not output.exists()
