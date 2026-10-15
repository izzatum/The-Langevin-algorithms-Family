"""Full-grid nonlinear calibration with a finite elliptical-slice reference.

Run: python benchmarks/nonlinear_calibration.py --output benchmarks/results/cpu
All latent cells are inferred. Datasets are independent prior-predictive draws;
chains within a dataset estimate one conditional posterior. Coordinate coverage
uses datasets as binomial trials, never correlated cells. Reference Monte Carlo
uncertainty and mixing warnings are reported, not assumed away.
"""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import time
import torch
from lafa.problems import NonlinearTomography2D
from lafa.samplers import MALA
from reference_sampling import elliptical_slice


def prior_predictive_problem(grid, *, seed):
    """x~N(-.6,.7² I); y,y_rep independently ~N(exp(-A exp(x)),.04² I).

    Replaces only this benchmark instance's synthetic data, preserving the
    public constructor's historical smooth-truth/reduced-model defaults.
    """
    problem = NonlinearTomography2D((grid, grid), seed=seed)
    rng = torch.Generator().manual_seed(seed)
    problem.truth = problem.offset + problem.prior_scale * torch.randn(problem.dimension, generator=rng, dtype=problem.dtype)
    expected = problem.forward(problem.truth)
    problem.observations = expected + problem.noise_scale * torch.randn(expected.shape, generator=rng, dtype=problem.dtype)
    replicate = expected + problem.noise_scale * torch.randn(expected.shape, generator=rng, dtype=problem.dtype)
    return problem, replicate


def wilson_interval(successes, trials, z=1.959963984540054):
    """Wilson 95% binomial interval across independent datasets."""
    p = successes / trials
    denominator = 1 + z*z/trials
    center = (p + z*z/(2*trials)) / denominator
    radius = z * math.sqrt(p*(1-p)/trials + z*z/(4*trials*trials)) / denominator
    # The analytic interval contains p. Preserve this at roundoff-sensitive
    # boundaries (e.g. k=n=10), avoiding negative plotted error-bar lengths.
    return [max(0., min(p, center-radius)), min(1., max(p, center+radius))]


def aggregate_coverage(events):
    """Rows are independent datasets; columns are correlated field cells/rays."""
    events = torch.as_tensor(events, dtype=torch.bool)
    n = events.shape[0]
    fractions = events.double().mean(-1)
    simultaneous = events.all(-1)
    return dict(independent_datasets=n, coordinate_coverage=events.double().mean(0).tolist(),
                coordinate_wilson_95=[wilson_interval(int(k), n) for k in events.sum(0)],
                field_fraction_by_dataset=fractions.tolist(), field_fraction_mean=float(fractions.mean()),
                field_fraction_dataset_standard_error=float(fractions.std()/math.sqrt(n)) if n > 1 else None,
                simultaneous_field_coverage=float(simultaneous.double().mean()),
                simultaneous_field_wilson_95=wilson_interval(int(simultaneous.sum()), n),
                note='Simultaneous coverage is intersection of marginal 95% intervals, not a calibrated 95% joint region. Dataset SE clusters correlated coordinates.')


def diagnostics(samples):
    from lafa.diagnostics import (rank_normalized_split_rhat, bulk_effective_sample_size,
                                  tail_effective_sample_size, mcse_mean, mcse_quantile)
    rhat = rank_normalized_split_rhat(samples)
    bulk, tail = bulk_effective_sample_size(samples), tail_effective_sample_size(samples)
    okay = bool(torch.isfinite(rhat).all() and (rhat < 1.01).all() and (bulk > 400).all() and (tail > 400).all())
    return dict(rhat=rhat, bulk_ess=bulk, tail_ess=tail, mean_mcse=mcse_mean(samples),
                interval_endpoint_mcse=torch.stack([mcse_quantile(samples, p) for p in (.025, .975)]),
                mixing_screen_passed=okay,
                mixing_screen='all coordinate rank/folded split Rhat <1.01, bulk/tail ESS >400; heuristic, no guarantee')


def summarize(samples, problem, replicate, predictive_seed):
    from lafa.diagnostics import mcse_mean, mcse_quantile
    flat = samples.reshape(-1, samples.shape[-1])
    probabilities = torch.tensor([.025, .975], dtype=samples.dtype)
    bounds = torch.quantile(flat, probabilities, dim=0)
    fields = samples.exp()
    # Monotonic exponentiation preserves the interval events for attenuation.
    field_bounds = bounds.exp()
    field_truth = problem.truth.exp()
    generator = torch.Generator().manual_seed(predictive_seed)
    predicted = problem.forward(samples)
    predicted += problem.noise_scale * torch.randn(predicted.shape, generator=generator, dtype=samples.dtype)
    predictive_bounds = torch.quantile(predicted.reshape(-1, predicted.shape[-1]), probabilities, dim=0)
    # Only compute transformed/predictive diagnostics actually reported. Full
    # rank/Rhat/ESS summaries are retained for the latent posterior below.
    field_mean_mcse = mcse_mean(fields)
    field_quantile_mcse = torch.stack([mcse_quantile(fields, p) for p in (.025, .975)])
    predictive_quantile_mcse = torch.stack([mcse_quantile(predicted, p) for p in (.025, .975)])
    return dict(sample_shape=list(samples.shape), mean=flat.mean(0), sd=flat.std(0), interval_95=bounds,
                chain_means=samples.mean(1), chain_interval_95=torch.quantile(samples, probabilities, dim=1),
                diagnostics=diagnostics(samples), field_mean=fields.mean((0, 1)), field_sd=fields.flatten(0, 1).std(0),
                field_interval_95=field_bounds, field_mean_mcse=field_mean_mcse,
                field_interval_endpoint_mcse=field_quantile_mcse,
                field_covered=(field_truth >= field_bounds[0]) & (field_truth <= field_bounds[1]),
                predictive_interval_95=predictive_bounds,
                predictive_interval_endpoint_mcse=predictive_quantile_mcse,
                predictive_covered=(replicate >= predictive_bounds[0]) & (replicate <= predictive_bounds[1]),
                predictive_seed=predictive_seed)


def candidate(problem, *, seeds, draws, warmup, step_size):
    generators = [torch.Generator().manual_seed(s) for s in seeds]
    initial = torch.stack([problem.offset + problem.prior_scale * torch.randn(problem.dimension, dtype=problem.dtype, generator=g) for g in generators])
    # Batch independent chains: elementwise iid Gaussian/uniform draws from one
    # explicitly seeded transition stream, with separate prior-initialization
    # seeds. There is no interacting-particle term or cross-chain adaptation.
    calls = 0

    def counted(x):
        nonlocal calls
        calls += math.prod(x.shape[:-1])
        return problem.potential(x)

    started = time.perf_counter()
    transition_seed = seeds[0] + 10000
    sampler = MALA(counted, initial, step_size=step_size,
                   generator=torch.Generator().manual_seed(transition_seed))
    samples = torch.empty(len(seeds), draws, problem.dimension, dtype=problem.dtype)
    elapsed = torch.empty(draws, dtype=torch.float64)
    accepted = 0
    for i in range(warmup + draws):
        result = sampler.step()
        if i >= warmup:
            samples[:, i-warmup] = result.position
            accepted += int(result.accepted.sum())
            elapsed[i-warmup] = time.perf_counter()-started
    return dict(samples=samples, elapsed_seconds=elapsed, seconds=time.perf_counter()-started,
                acceptance_rate=accepted/(len(seeds)*draws), gradient_chain_evaluations=calls,
                initialization_seeds=list(seeds), transition_seed=transition_seed,
                warmup=warmup, draws=draws, step_size=step_size)


def gaussian_reference_validation(draws=3000):
    mean = torch.tensor([.4, -.8], dtype=torch.float64)
    chol = torch.tensor([[1., 0.], [.3, .8]], dtype=torch.float64)
    a = torch.tensor([[1., .4], [-.2, .9]], dtype=torch.float64)
    y = torch.tensor([.7, -.1], dtype=torch.float64)
    prior_precision = torch.cholesky_inverse(chol)
    covariance = torch.linalg.inv(prior_precision + a.T @ a / .6**2)
    expected = covariance @ (prior_precision @ mean + a.T @ y / .6**2)
    run = elliptical_slice(lambda x: -.5*((x @ a.T-y)/.6).square().sum(-1), mean, chol,
                           seeds=[2, 3, 5, 7], draws=draws, warmup=500)
    samples = run.pop('samples')
    run.pop('elapsed_seconds')
    estimated_covariance = torch.cov(samples.flatten(0, 1).T)
    error = samples.mean((0, 1))-expected
    diag = diagnostics(samples)
    return dict(**run, exact_mean=expected, exact_covariance=covariance,
                estimated_mean=samples.mean((0, 1)), estimated_covariance=estimated_covariance,
                mean_error=error, mean_error_in_mcse=error/diag['mean_mcse'], diagnostics=diag,
                numerical_check_passed=bool((error.abs()<.035).all() and ((estimated_covariance-covariance).abs()<.025).all()))


def serializable(value):
    if isinstance(value, torch.Tensor):
        return serializable(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {key: serializable(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def figures(data, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/lafa-matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(data['grids']), 4, figsize=(13, 3.2*len(data['grids'])), squeeze=False, constrained_layout=True)
    for row, grid in zip(axes, data['grids']):
        first = grid['datasets'][0]
        n = grid['grid']
        truth = torch.tensor(first['truth']).exp().reshape(n, n)
        mean = torch.tensor(first['reference']['field_mean']).reshape(n, n)
        sd = torch.tensor(first['reference']['field_sd']).reshape(n, n)
        covered = torch.tensor(first['reference']['field_covered']).reshape(n, n)
        for ax, field, title in zip(row, [truth, mean, sd, covered], ['Prior-drawn truth', 'Reference posterior mean', 'Reference posterior SD', 'Truth inside marginal 95% interval']):
            options = dict(vmin=min(float(truth.min()), float(mean.min())), vmax=max(float(truth.max()), float(mean.max()))) if title in ('Prior-drawn truth', 'Reference posterior mean') else {}
            im = ax.imshow(field, origin='lower', **options)
            ax.set_title(f'{n}×{n}: {title}', fontsize=10)
            fig.colorbar(im, ax=ax)
    fig.savefig(output/'nonlinear-fields.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, len(data['grids']), figsize=(6*len(data['grids']), 11), squeeze=False, constrained_layout=True)
    for column, grid in enumerate(data['grids']):
        ax = axes[0, column]
        for run in grid['datasets'][0]['candidates']:
            path = run['error_path']
            ax.plot([p['seconds'] for p in path], [p['mean_rmse_to_reference'] for p in path], marker='o', label=f"MALA h={run['step_size']}")
        reference = grid['datasets'][0]['reference']
        ax.axhline(reference['mean_mcse_rms'], color='black', linestyle='--', label='Reference RMS mean MCSE')
        ax.set(xlabel='Measured CPU seconds including warmup', ylabel='Latent mean RMSE to finite reference', title=f"{grid['grid']}×{grid['grid']}, dataset 0")
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax = axes[2, column]
        steps = [run['step_size'] for run in grid['datasets'][0]['candidates']]
        errors = torch.tensor([[run['error_path'][-1]['mean_rmse_to_reference'] for run in record['candidates']] for record in grid['datasets']])
        standard_error = errors.std(0)/math.sqrt(len(errors)) if len(errors)>1 else torch.zeros(len(steps))
        ax.errorbar(steps, errors.mean(0), yerr=standard_error, marker='o', capsize=4)
        ax.set(xlabel='Fixed MALA step size', ylabel='Mean latent RMSE to finite reference',
               title=f"{grid['grid']}×{grid['grid']}: dataset mean ± dataset SE")
        ax.set_xscale('log')
        ax = axes[1, column]
        coverage = grid['reference_calibration']['field']
        p = coverage['coordinate_coverage']
        intervals = coverage['coordinate_wilson_95']
        ax.errorbar(range(len(p)), p, yerr=[[v-b[0] for v,b in zip(p, intervals)], [b[1]-v for v,b in zip(p, intervals)]], fmt='o', label='Reference, Wilson 95%')
        for run in grid['candidate_calibration']:
            ax.plot(range(len(p)), run['field']['coordinate_coverage'], '.', label=f"MALA h={run['step_size']}")
        ax.axhline(.95, color='black', linestyle='--')
        ax.set(xlabel='Grid cell', ylabel='Coverage across independent datasets', ylim=(0, 1.03), title=f"{coverage['independent_datasets']} datasets; cells are correlated")
        ax.legend(fontsize=8)
    fig.savefig(output/'nonlinear-error-coverage.png', dpi=150)
    plt.close(fig)


def refine_references(args):
    """Recompute finite references without repeating already measured candidates.

    Existing short-reference summaries are retained as a budget history. This
    requires saved raw candidate chains to recompute every prefix comparison.
    The new run has independent seeds and a fixed user-visible finite cap.
    """
    if args.raw_output is None:
        raise ValueError('--refine-reference-draws requires --raw-output')
    with (args.output/'nonlinear-calibration.json').open() as stream:
        data = json.load(stream)
    raw = torch.load(args.raw_output/'nonlinear-chains.pt', weights_only=True)
    data['metadata']['reference_refinement'] = dict(draws=args.refine_reference_draws,
        warmup=args.reference_warmup, grids=args.grids, seed_offset=args.reference_seed_offset,
        policy='Fixed additional reference budget; retain old diagnostics and report remaining failures')
    for grid in data['grids']:
        n = grid['grid']
        if n not in args.grids:
            continue
        for record in grid['datasets']:
            seed = record['data_seed']
            dataset = record['dataset']
            problem, replicate = prior_predictive_problem(n, seed=seed)
            mean = torch.full((n*n,), problem.offset, dtype=problem.dtype)
            chol = problem.prior_scale * torch.eye(n*n, dtype=problem.dtype)
            ll = lambda x: -.5*((problem.forward(x)-problem.observations)/problem.noise_scale).square().sum(-1)
            chains = record['reference']['sample_shape'][0]
            ref = elliptical_slice(ll, mean, chol, seeds=[seed+args.reference_seed_offset+10+c for c in range(chains)],
                                   draws=args.refine_reference_draws, warmup=args.reference_warmup)
            samples, elapsed = ref.pop('samples'), ref.pop('elapsed_seconds')
            ref.update(summarize(samples, problem, replicate, seed+args.reference_seed_offset+100))
            ref['mean_mcse_rms'] = ref['diagnostics']['mean_mcse'].square().mean().sqrt()
            record.setdefault('reference_budget_history', []).append(record['reference'])
            record['reference'] = ref
            raw[f'grid{n}/dataset{dataset}/reference'] = dict(samples=samples, elapsed_seconds=elapsed)
            for index, run in enumerate(record['candidates']):
                stored = raw[f'grid{n}/dataset{dataset}/mala{index}']
                candidate_samples = stored['samples']
                candidate_mcse = torch.tensor([float('nan') if value is None else value
                                               for value in run['diagnostics']['mean_mcse']], dtype=samples.dtype)
                combined = (candidate_mcse.square()+ref['diagnostics']['mean_mcse'].square()).sqrt()
                run['mean_difference_in_combined_mcse'] = (candidate_samples.mean((0, 1))-ref['mean'])/combined
                for point in run['error_path']:
                    point['mean_rmse_to_reference'] = (candidate_samples[:, :point['draws']].mean((0, 1))-ref['mean']).square().mean().sqrt()
            print(f'refined grid={n} dataset={dataset} Rhat max={float(ref["diagnostics"]["rhat"].max()):.3f}, ESS min={float(ref["diagnostics"]["bulk_ess"].min()):.1f}', flush=True)
        grid['reference_calibration'] = dict(field=aggregate_coverage(torch.stack([torch.as_tensor(r['reference']['field_covered']) for r in grid['datasets']])),
            posterior_predictive=aggregate_coverage(torch.stack([torch.as_tensor(r['reference']['predictive_covered']) for r in grid['datasets']])))
        with (args.output/'nonlinear-calibration.json').open('w') as stream:
            json.dump(serializable(data), stream, indent=2, allow_nan=False)
        torch.save(raw, args.raw_output/'nonlinear-chains.pt')
    if not args.no_figures:
        figures(serializable(data), args.output)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmarks/results/cpu'))
    parser.add_argument('--raw-output', type=Path, help='Optional separate directory for potentially large chain/time tensors')
    parser.add_argument('--datasets', type=int, default=8)
    parser.add_argument('--grids', nargs='+', type=int, default=[3, 4])
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--reference-draws', type=int, default=2000)
    parser.add_argument('--reference-warmup', type=int, default=1000)
    parser.add_argument('--refine-reference-draws', type=int, help='Rerun only references from existing JSON/raw output at this longer budget')
    parser.add_argument('--reference-seed-offset', type=int, default=50000000,
                        help='Refinement seed offset; change for independent budget comparisons')
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--steps', nargs='+', type=float, default=[.001, .003, .008])
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args(argv)
    if min(args.datasets, args.chains, args.reference_draws, args.draws, *args.grids) < 1 or args.chains < 2 or min(args.reference_draws, args.draws) < 20 or min(args.warmup, args.reference_warmup) < 0 or any(not math.isfinite(h) or h <= 0 for h in args.steps):
        parser.error('positive finite budgets/steps, >=2 chains and >=20 draws required')
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.refine_reference_draws is not None:
        if args.refine_reference_draws < 20:
            parser.error('--refine-reference-draws must be >=20')
        return refine_references(args)
    validation = gaussian_reference_validation()
    if not validation['numerical_check_passed']:
        raise RuntimeError('independent reference failed analytic Gaussian validation')
    data = dict(metadata=dict(torch_version=torch.__version__, python_version=platform.python_version(),
                platform=platform.platform(), device='cpu', dtype='float64', threads=1, cuda_executed=False,
                arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                reference_source='Murray, Adams & MacKay (2010), https://proceedings.mlr.press/v9/murray10a.html',
                model='x~N(-0.6,0.7²I); y=exp(-A exp(x))+N(0,0.04²I); every grid cell inferred',
                limitations='Finite reference and candidate mixing; 8 datasets by default give coarse calibration. No exact finite-run reference or calibration guarantee. Nonfinite diagnostics serialize as null.',
                raw_axes='samples=(independent chains, retained draws, full-grid coordinates); elapsed_seconds=(retained draws), includes warmup'),
                gaussian_reference_validation=validation, grids=[])
    raw = {}
    for grid in args.grids:
        records = []
        for dataset in range(args.datasets):
            seed = args.seed + 100000*grid + 1000*dataset
            problem, replicate = prior_predictive_problem(grid, seed=seed)
            mean = torch.full((problem.dimension,), problem.offset, dtype=problem.dtype)
            chol = problem.prior_scale * torch.eye(problem.dimension, dtype=problem.dtype)
            ll = lambda x: -.5*((problem.forward(x)-problem.observations)/problem.noise_scale).square().sum(-1)
            ref = elliptical_slice(ll, mean, chol, seeds=[seed+10+c for c in range(args.chains)],
                                   draws=args.reference_draws, warmup=args.reference_warmup)
            reference_samples = ref.pop('samples')
            reference_times = ref.pop('elapsed_seconds')
            ref.update(summarize(reference_samples, problem, replicate, seed+100))
            ref['mean_mcse_rms'] = ref['diagnostics']['mean_mcse'].square().mean().sqrt()
            record = dict(dataset=dataset, data_seed=seed, truth=problem.truth, observations=problem.observations,
                          independent_replicate=replicate, reference=ref, candidates=[])
            if args.raw_output is not None:
                raw[f'grid{grid}/dataset{dataset}/reference'] = dict(samples=reference_samples, elapsed_seconds=reference_times)
            for index, step in enumerate(args.steps):
                run = candidate(problem, seeds=[seed+200+20*index+c for c in range(args.chains)],
                                draws=args.draws, warmup=args.warmup, step_size=step)
                samples = run.pop('samples')
                elapsed = run.pop('elapsed_seconds')
                run.update(summarize(samples, problem, replicate, seed+400+index))
                combined_mcse = (run['diagnostics']['mean_mcse'].square()+ref['diagnostics']['mean_mcse'].square()).sqrt()
                run['mean_difference_in_combined_mcse'] = (run['mean']-ref['mean'])/combined_mcse
                run['error_path'] = [dict(draws=k, seconds=elapsed[k-1], mean_rmse_to_reference=(samples[:, :k].mean((0, 1))-ref['mean']).square().mean().sqrt())
                                     for k in sorted(set([max(1, args.draws//8), args.draws//4, args.draws//2, args.draws]))]
                record['candidates'].append(run)
                if args.raw_output is not None:
                    raw[f'grid{grid}/dataset{dataset}/mala{index}'] = dict(samples=samples, elapsed_seconds=elapsed)
            records.append(record)
            print(f'grid={grid} dataset={dataset} reference Rhat max={float(ref["diagnostics"]["rhat"].max()):.3f}, ESS min={float(ref["diagnostics"]["bulk_ess"].min()):.1f}', flush=True)
        calibration = lambda runs: dict(field=aggregate_coverage(torch.stack([r['field_covered'] for r in runs])),
                                        posterior_predictive=aggregate_coverage(torch.stack([r['predictive_covered'] for r in runs])))
        data['grids'].append(dict(grid=grid, dimension=grid*grid, datasets=records,
                                 reference_calibration=calibration([r['reference'] for r in records]),
                                 candidate_calibration=[dict(step_size=step, **calibration([r['candidates'][i] for r in records])) for i,step in enumerate(args.steps)]))
        # Save completed grids so a bounded interrupted experiment remains inspectable.
        with (args.output/'nonlinear-calibration.json').open('w') as stream:
            json.dump(serializable(data), stream, indent=2, allow_nan=False)
        if args.raw_output is not None:
            args.raw_output.mkdir(parents=True, exist_ok=True)
            torch.save(raw, args.raw_output/'nonlinear-chains.pt')
    if not args.no_figures:
        figures(serializable(data), args.output)
    return data


if __name__ == '__main__':
    main()
