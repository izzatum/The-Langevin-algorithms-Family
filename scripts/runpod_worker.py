#!/usr/bin/env python3
"""Bounded Serverless CUDA validation worker; importing this module starts nothing."""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback

MAX_ARCHIVE_BYTES = 6_000_000
MAX_EXTRACT_BYTES = 128_000_000


def safe_extract(data, destination, max_bytes=MAX_EXTRACT_BYTES):
    """Extract only regular files/directories, with path, count and size limits."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as archive:
        members = archive.getmembers()
        if len(members) > 10_000 or sum(m.size for m in members) > max_bytes:
            raise ValueError('Archive size/member limit exceeded')
        seen = set()
        for member in members:
            path = PurePosixPath(member.name)
            target = root.joinpath(*path.parts)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name
                    or not path.parts or member.name in seen
                    or not (member.isfile() or member.isdir())
                    or not target.resolve().is_relative_to(root)):
                raise ValueError('Unsafe or duplicate archive path/type')
            seen.add(member.name)
        for member in members:
            target = root.joinpath(*PurePosixPath(member.name).parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Existing links/files are never overwritten by archive extraction.
                with target.open('xb') as output, archive.extractfile(member) as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)


def run_command(command, cwd, output, deadline, env=None):
    """Persist stdout/stderr and return the actual command exit or timeout status."""
    started = time.monotonic()
    record = {'command': command, 'log': Path(output).name, 'timed_out': False}
    with Path(output).open('wb') as log:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log.write(b'Validation wall deadline exhausted before command\n')
            record.update(returncode=124, timed_out=True, seconds=0)
            return record
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            record['returncode'] = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            record['timed_out'] = True
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            record['returncode'] = 124
            log.write(b'\nValidation wall deadline exceeded; process group killed\n')
    record['seconds'] = time.monotonic() - started
    return record


def bundle_results(directory):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for file in sorted(Path(directory).rglob('*')):
            if file.is_symlink():
                raise ValueError('Results must not contain symbolic links')
            if file.is_file():
                archive.add(file, arcname=file.relative_to(directory).as_posix(), recursive=False)
    data = buffer.getvalue()
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ValueError('Results exceed six MB transport limit')
    return {'artifact_b64': base64.b64encode(data).decode(),
            'artifact_sha256': hashlib.sha256(data).hexdigest()}


def handler(job):
    """Return artifacts on failure too; SDK1.12 maps the error field to FAILED."""
    started = time.monotonic()
    deadline = started + 840  # Leave one minute for packaging before provider900s timeout.
    input_data = job.get('input', {})
    run_id = input_data.get('run_id', '')
    metadata = {'run_id': run_id, 'success': False, 'stages': [],
                'python': platform.python_version(), 'cuda_executed': False,
                'worker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix='lafa-gpu-') as temporary:
        work = Path(temporary)
        repo, outputs = work / 'repo', work / 'outputs'
        outputs.mkdir()
        try:
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', run_id):
                raise ValueError('Invalid run identity')
            encoded = input_data.get('payload_b64', '')
            if len(encoded) > 9_000_000:
                raise ValueError('Input payload too large')
            data = base64.b64decode(encoded, validate=True)
            if hashlib.sha256(data).hexdigest() != input_data.get('payload_sha256'):
                raise ValueError('Payload checksum mismatch')
            if input_data.get('worker_sha256') != metadata['worker_sha256']:
                raise ValueError('Worker changed after payload approval')
            safe_extract(data, repo)
            if sys.version_info < (3, 11):
                raise RuntimeError('The project requires Python>=3.11; use compatible image')
            import torch
            version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
            if version < (2, 6) or not torch.cuda.is_available():
                raise RuntimeError('Preinstalled torch>=2.6 with CUDA required; torch is not reinstalled')
            # Actual allocation, kernel execution and synchronization, not device-label inference.
            value = torch.arange(64, device='cuda', dtype=torch.float64).square().sum()
            torch.cuda.synchronize()
            assert value.is_cuda and value.item() == 85344
            metadata.update(torch=torch.__version__, cuda_runtime=torch.version.cuda,
                            gpu=torch.cuda.get_device_name(0), cuda_executed=True)
            environment = os.environ.copy()
            environment.update(PYTHONPATH=str(repo / 'src'), LAFA_DEVICE='cuda',
                               LAFA_QUICK='1', MPLBACKEND='Agg', OMP_NUM_THREADS='1',
                               MPLCONFIGDIR=str(work / 'matplotlib'), XDG_CACHE_HOME=str(work / 'cache'))
            # Install optional tooling only; no command resolves or replaces PyTorch.
            stages = [
                ('dependencies', [sys.executable, '-m', 'pip', 'install', '--disable-pip-version-check',
                                  '--no-cache-dir', 'setuptools>=77', 'wheel', 'pytest>=8,<9', 'nbclient>=0.10,<1',
                                  'nbformat>=5.10,<6', 'ipykernel>=6,<7',
                                  'matplotlib>=3.8,<4', 'numpy>=1.26,<3']),
                ('install-project', [sys.executable, '-m', 'pip', 'install', '--no-deps',
                                     '--no-build-isolation', '-e', '.']),
                ('pytest', [sys.executable, '-m', 'pytest', '-q', '--junitxml=' + str(outputs / 'pytest.xml')]),
                ('notebooks', [sys.executable, 'scripts/execute_notebooks.py', '--device', 'cuda',
                               '--output', str(outputs / 'notebooks'), '--timeout', '120']),
            ]
            notebooks = sorted((repo / 'notebooks').glob('*.ipynb'))
            if len(notebooks) != 6:
                raise RuntimeError(f'Expected six tutorial notebooks; found {len(notebooks)}')
            if input_data.get('accuracy', False):
                stages.append(('accuracy', [sys.executable, 'benchmarks/accuracy.py', '--device', 'cuda',
                                           '--output', str(outputs / 'accuracy'), '--seeds', '7',
                                           '--iterations', '300', '--warmup', '100', '--chains', '32']))
            for name, command in stages:
                record = run_command(command, repo, outputs / (name + '.log'), deadline, environment)
                record['stage'] = name
                metadata['stages'].append(record)
                (outputs / 'metadata.json').write_text(json.dumps(metadata, indent=2))
                if record['returncode'] != 0 and name in ('dependencies', 'install-project'):
                    break
            notebook_results = outputs / 'notebooks' / 'execution-results.json'
            reports = json.loads(notebook_results.read_text()) if notebook_results.exists() else []
            stages_ok = len(metadata['stages']) == len(stages) and all(
                row['returncode'] == 0 for row in metadata['stages'])
            notebooks_ok = len(reports) == 6 and all(
                row.get('status') == 'passed' and row.get('device') == 'cuda' for row in reports)
            metadata['success'] = stages_ok and notebooks_ok
            if not metadata['success']:
                metadata['error'] = 'Validation failed; inspect stage logs and notebook execution results'
        except Exception as error:
            metadata['error'] = f'{type(error).__name__}: {error}'
            (outputs / 'exception.log').write_text(traceback.format_exc())
        metadata['seconds'] = time.monotonic() - started
        (outputs / 'metadata.json').write_text(json.dumps(metadata, indent=2))
        try:
            artifact = bundle_results(outputs)
        except Exception as error:
            metadata['success'] = False
            metadata['error'] = f'Artifact packaging failed: {error}'
            # Keep metadata/failure information transportable rather than report false success.
            small = work / 'failure-summary'
            small.mkdir()
            metadata['truncated_artifact'] = True
            metadata['omitted_files'] = [str(f.relative_to(outputs)) for f in outputs.rglob('*') if f.is_file() and f.suffix != '.log']
            (small / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            for log in outputs.glob('*.log'):
                with log.open('rb') as source:
                    source.seek(max(0, log.stat().st_size - 64000))
                    (small / log.name).write_bytes(source.read())
            artifact = bundle_results(small)
        result = {'run_id': run_id, 'success': metadata['success'], **artifact}
        if not metadata['success']:
            result['error'] = metadata.get('error', 'GPU validation failed')
        return result


if __name__ == '__main__':
    import runpod
    runpod.serverless.start({'handler': handler})
