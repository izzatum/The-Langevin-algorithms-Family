"""Build public documentation from the maintained Markdown and executed notebooks.

Notebook outputs are converted, never executed. Source-only files stay on
GitHub; citation links and tutorial links resolve to the corresponding HTML.
"""
from pathlib import Path, PurePosixPath
import posixpath
import re
from urllib.parse import urlsplit, urlunsplit
import xml.etree.ElementTree as ET

import nbformat
from nbconvert import MarkdownExporter
from mkdocs.structure.files import File

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = 'https://github.com/izzatum/The-Langevin-algorithms-Family'
RAW = 'https://raw.githubusercontent.com/izzatum/The-Langevin-algorithms-Family/main/'
NOTEBOOKS = {
    '01_overdamped': ('Overdamped Langevin tutorial: ULA, MALA and adaptive steps',
        'Compare ULA, MALA and Lipschitz-adaptive Langevin on Gaussian and multimodal targets, with executed CPU figures and independent-chain diagnostics.'),
    '02_inertial': ('Underdamped Langevin, BAOAB and HOLA tutorial',
        'Compare kinetic Langevin, BAOAB splitting and higher-order overdamped HOLA using executed PyTorch examples, derivative costs and finite-step checks.'),
    '03_specialized': ('Annealed, tempered and proximal Langevin tutorial',
        'Explore temperature and noise annealing, replica exchange and proximal interacting particle learning with explicit objectives and reference comparisons.'),
    '04_stein_particles': ('Stein variational inference and neural-network tutorial',
        'Run SVGD, projected SVGD, Stein variational Newton and function-space methods, including chunked PyTorch neural-network targets and actual figures.'),
    '05_ksd': ('Kernel Stein discrepancy tutorial: RBF and IMQ',
        'Compare RBF and IMQ kernel Stein discrepancies, signed U and V statistics, exact blocked computation and examples where a small KSD misses modes.'),
    '06_ensemble_hybrid': ('Ensemble Langevin and 2D nonlinear inverse-problem tutorial',
        'Compare repulsive and birth-death ensembles, SA-LEnKF filtering and 2D nonlinear tomography, with posterior uncertainty figures and finite-run limits.'),
}

# Descriptions of the committed figures, in output order. Require an update
# when a notebook changes its figure count instead of publishing alt="png".
FIGURES = {
    '01_overdamped': [
        'Gaussian MALA samples, ULA and MALA coordinate traces, and adaptive step-size histories',
        'MALA samples on a four-mode Gaussian mixture and a Gaussian banana target',
        'Four independent MALA chains traced across both Gaussian coordinates',
    ],
    '02_inertial': [
        'Position, momentum and autocorrelation traces for kinetic and higher-order methods',
        'Covariance error versus step size for kinetic Euler, BAOAB and tamed or untamed HOLA',
    ],
    '03_specialized': [
        'Final particles from annealing and replica-exchange workflows on a four-mode mixture',
        'Particle coordinate traces alongside temperature and noise schedules',
        'Proximal shared-parameter trajectories and the integrated log-likelihood reference',
    ],
    '04_stein_particles': [
        'Particle trajectories and final positions for SVGD, projected SVGD and Stein Newton',
        'Weight-space and function-space regression predictions compared with an analytic Gaussian reference',
        'Short neural-network SVGD run: data, particle mean prediction and empirical 5–95 percent band',
    ],
    '05_ksd': [
        'RBF and IMQ kernel Stein discrepancy estimates and runtime versus sample count',
        'Signed U-statistic KSD estimates, including negative values, from independent null samples',
    ],
    '06_ensemble_hybrid': [
        'KDE repulsive and Wasserstein–Fisher–Rao birth–death particles on a Gaussian mixture',
        'SA-LEnKF state estimates against exact Kalman filtering and a separate parameter-learning trajectory',
        'Reduced nonlinear tomography: true attenuation, posterior mean, standard deviation and coefficient samples',
        'Full-grid 3-by-3 nonlinear tomography: true attenuation and finite-run posterior mean and standard deviation',
    ],
}


def on_files(files, config):
    exporter = MarkdownExporter()
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    for stem, (title, description) in NOTEBOOKS.items():
        source = ROOT / 'notebooks' / (stem + '.ipynb')
        notebook = nbformat.read(source, as_version=4)
        code = [cell for cell in notebook.cells if cell.cell_type == 'code']
        if not code or any(cell.execution_count is None for cell in code):
            raise ValueError(f'{source.name} must contain executed code cells')
        if any(output.output_type == 'error' for cell in code for output in cell.outputs):
            raise ValueError(f'{source.name} contains an error output')
        body, resources = exporter.from_notebook_node(notebook, resources={
            'unique_key': stem, 'output_files_dir': f'assets/{stem}',
        })
        figures = FIGURES[stem]
        if body.count('![png](') != len(figures):
            raise ValueError(f'{source.name}: update the figure descriptions for changed outputs')
        descriptions = iter(figures)
        body = re.sub(r'!\[png\]\(', lambda _: '![' + next(descriptions) + '](', body)
        # Notebook source links use paths relative to the repository notebooks/.
        body = body.replace('../docs/', '../')
        # Keep the notebook's own H1; put provenance directly below it.
        first_line, separator, rest = body.partition('\n')
        provenance = (
            '\n\n!!! note "Recorded execution"\n'
            '    These are the committed CPU outputs, not a new execution of this page. '
            'Short budgets illustrate behavior and may not mix. '
            f'[Download or run the notebook]({REPOSITORY}/blob/main/notebooks/{stem}.ipynb) '
            'and read the [validation limits](../validation.md).\n\n'
        )
        body = first_line + provenance + rest if separator else body
        # Values are controlled strings; JSON quoting is valid YAML quoting.
        import json
        metadata = f'---\ntitle: {json.dumps(title)}\ndescription: {json.dumps(description)}\n---\n\n'
        files.append(File.generated(config, f'notebooks/{stem}.md', content=metadata + body))
        for name, data in resources.get('outputs', {}).items():
            files.append(File.generated(config, f'notebooks/{name}', content=data))
    return files


# The maintained Markdown uses ordinary inline links/images. Match destination
# only so titles, image markers, labels, anchors and reference prose survive.
LINK = re.compile(r'(!?\[[^\]\n]*\]\()([^\s)]+)([^)\n]*\))')


def on_page_markdown(markdown, page, config, files):
    generated_notebook = page.file.src_uri.startswith('notebooks/')
    source_dir = PurePosixPath(page.file.src_uri).parent

    def rewrite(match):
        start, destination, end = match.groups()
        url = urlsplit(destination)
        if url.scheme or url.netloc or not url.path or url.path.startswith('/'):
            return match.group(0)
        resolved = posixpath.normpath(str(source_dir / url.path))
        # The tutorial guide links to repository notebooks; publish their HTML.
        if resolved.startswith('../notebooks/') and resolved.endswith('.ipynb'):
            published = resolved[3:-6] + '.md'
            relative = posixpath.relpath(published, str(source_dir))
            return start + urlunsplit(('', '', relative, url.query, url.fragment)) + end
        if not resolved.startswith('../'):
            # Original notebook links to benchmarks/ resolve relative to the
            # repository, while converted output assets resolve within docs/.
            if generated_notebook and files.get_file_from_path(resolved) is None:
                candidate = ROOT / resolved
                if candidate.is_file():
                    base = RAW if start.startswith('!') else REPOSITORY + '/blob/main/'
                    return start + base + urlunsplit(('', '', resolved, url.query, url.fragment)) + end
            return match.group(0)
        # Repo evidence and code outside docs/ remain authoritative GitHub files.
        repository_path = resolved[3:]
        if not (ROOT / repository_path).is_file():
            raise ValueError(f'{page.file.src_uri}: missing source link {destination}')
        base = RAW if start.startswith('!') else REPOSITORY + '/blob/main/'
        return start + base + urlunsplit(('', '', repository_path, url.query, url.fragment)) + end

    markdown = LINK.sub(rewrite, markdown)
    if generated_notebook:
        page.edit_url = REPOSITORY + '/edit/main/' + page.file.src_uri[:-3] + '.ipynb'
    return markdown


def on_post_build(config):
    # MkDocs' default sitemap dates are build dates, not content revisions.
    # Omit them rather than advertise unchanged pages as freshly modified.
    sitemap = Path(config.site_dir) / 'sitemap.xml'
    ET.register_namespace('', 'http://www.sitemaps.org/schemas/sitemap/0.9')
    tree = ET.parse(sitemap)
    for entry in tree.getroot():
        for child in list(entry):
            if child.tag.endswith('}lastmod'):
                entry.remove(child)
    tree.write(sitemap, encoding='utf-8', xml_declaration=True)
    import gzip
    with gzip.open(str(sitemap) + '.gz', 'wb') as output:
        output.write(sitemap.read_bytes())
