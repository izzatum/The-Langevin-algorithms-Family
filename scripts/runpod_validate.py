#!/usr/bin/env python3
"""Prepare or explicitly run one bounded Runpod Serverless CUDA validation job."""
from __future__ import annotations

import argparse
import base64
from decimal import Decimal, ROUND_UP
import fcntl
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
import uuid

from runpod_worker import safe_extract

ROOT = Path(__file__).resolve().parents[1]
# Official Docker Hub tag2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04,
# checked2026-10-15. Python>=3.11 and torch>=2.6 are required by this project.
IMAGE = 'runpod/pytorch@sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853'
EXECUTION_MS = 900_000
TTL_MS = 1_200_000
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMED_OUT'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.new')
    with temporary.open('w') as file:
        os.chmod(temporary, 0o600)
        json.dump(value, file, indent=2)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    temporary.replace(path)


def build_payload(root):
    """Archive only explicit source file types; strip stale notebook outputs."""
    root = Path(root).resolve()
    files = set()
    for directory in ('src', 'tests', 'scripts', 'benchmarks'):
        files.update((root / directory).rglob('*.py'))
    files.update((root / 'notebooks').glob('*.ipynb'))
    for metadata_file in ('pyproject.toml', 'README.md', 'LICENSE'):
        if (root / metadata_file).is_file():
            files.add(root / metadata_file)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w') as archive:
            for path in sorted(files):
                if path.is_symlink() or not path.resolve().is_relative_to(root):
                    raise ValueError('Source payload may not contain symlinks')
                if any(part.startswith('.') or part in ('__pycache__',) for part in path.relative_to(root).parts):
                    continue
                data = path.read_bytes()
                if path.suffix == '.ipynb':
                    notebook = json.loads(data)
                    for cell in notebook.get('cells', []):
                        if cell.get('cell_type') == 'code':
                            cell['outputs'] = []
                            cell['execution_count'] = None
                    data = json.dumps(notebook).encode()
                info = tarfile.TarInfo(path.relative_to(root).as_posix())
                info.size = len(data)
                info.mode = 0o644
                archive.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    payload = {'payload_b64': base64.b64encode(data).decode(), 'payload_sha256': digest(data)}
    if len(json.dumps(payload).encode()) > 9_000_000:
        raise ValueError('Payload exceeds safe nine MB request envelope')
    return payload


def reserve_budget(path, run_id, reservation=1, run_budget=3):
    """Persist conservative charged/reserved cost; never silently reset or refund."""
    reservation, run_budget = Decimal(str(reservation)), Decimal(str(run_budget))
    if (not reservation.is_finite() or not run_budget.is_finite()
            or not 1 <= reservation <= run_budget <= 3):
        raise ValueError('Each attempt reserves >=$1 within a <=$3 run budget')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = json.loads(path.read_text()) if path.exists() else {
            'hard_cap_usd': 10, 'reserved_usd': 0, 'attempts': []}
        spent = Decimal(str(ledger['reserved_usd']))
        if not spent.is_finite() or spent < 0 or ledger.get('hard_cap_usd') != 10:
            raise ValueError('Invalid budget ledger; manual reconciliation required')
        recorded = sum((Decimal(str(row['reservation_usd'])) for row in ledger['attempts']), Decimal(0))
        if not recorded.is_finite() or recorded != spent:
            raise ValueError('Inconsistent budget ledger; manual reconciliation required')
        if any(row['run_id'] == run_id for row in ledger['attempts']):
            raise ValueError('Run identity already reserved; no automatic resubmission')
        if spent + reservation > 10:
            raise ValueError('Hard total $10 budget would be exceeded')
        ledger['attempts'].append({'run_id': run_id, 'reservation_usd': float(reservation),
                                   'reserved_at_unix': time.time()})
        ledger['reserved_usd'] = float(spent + reservation)
        write_json(path, ledger)
    return ledger


def normalize_output(response):
    output = response.get('output')
    if isinstance(output, str):
        output = json.loads(output)
    if isinstance(output, list) and len(output) == 1:
        output = output[0]
    if not isinstance(output, dict):
        raise ValueError('Provider response has no structured worker output')
    return output


def api(host, path, method='GET', body=None):
    key = os.environ.get('RUNPOD_API_KEY')
    if not key:
        raise RuntimeError('RUNPOD_API_KEY binding is missing; use secure environment settings')
    if host not in ('rest.runpod.io', 'api.runpod.ai'):
        raise ValueError('Unsupported credential destination')
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request('https://' + host + path, data=data, method=method,
        headers={'Authorization': 'Bearer ' + key, 'User-Agent': 'runpod-python/1.12.0',
                 'Accept': 'application/json', 'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            content = response.read(12_000_000)
            return json.loads(content) if content else None
    except urllib.error.HTTPError as error:
        text = error.read(2000).decode(errors='replace').replace(key, '[REDACTED]')
        raise RuntimeError(f'{method} {host}{path}: HTTP{error.code} {text}') from None


def management(path, method='GET', body=None):
    return api('rest.runpod.io', '/v1' + path, method, body)



def retry_management(path, method='GET', body=None, attempts=3):
    """Bounded retries for reads/idempotent cleanup only; never call with POST."""
    if method not in ('GET', 'PATCH', 'DELETE'):
        raise ValueError('Creation/submission must never be retried automatically')
    for attempt in range(attempts):
        try:
            return management(path, method, body)
        except (OSError, RuntimeError):
            if attempt + 1 == attempts:
                raise
            time.sleep(2 ** attempt)


def require_idle_health(health):
    """Accept only a new, idle endpoint with no current or historical jobs."""
    if not isinstance(health, dict):
        raise ValueError('Unexpected idle endpoint health response')
    for section, required in (('jobs', ('inQueue', 'inProgress')),
                              ('workers', ('idle', 'running'))):
        counts = health.get(section)
        if not isinstance(counts, dict) or not all(key in counts for key in required):
            raise ValueError('Missing idle endpoint job/worker counters')
        if any(type(value) is not int or value != 0 for value in counts.values()):
            raise ValueError('Require idle endpoint without prior or queued jobs/workers')


def require_storage(endpoint, template):
    for resource in (endpoint, template):
        if any(resource.get(key) for key in ('networkVolumeId', 'networkVolumeIds',
                                             'networkVolume', 'networkVolumes')):
            raise ValueError('Persistent network volumes are not permitted')
        if resource.get('volumeInGb') not in (None, 0):
            raise ValueError('Persistent local volume is not permitted')
        if 'containerDiskInGb' in resource and resource['containerDiskInGb'] != 20:
            raise ValueError('Require exact 20GB container disk')
    # Serverless GET omits Pod-only zero-volume fields. Serverless persistent
    # storage is an explicitly attached network volume, rejected above.
    volume = template.get('volumeInGb', 0 if template.get('isServerless') is True else None)
    if template.get('containerDiskInGb') != 20 or volume != 0:
        raise ValueError('Require verified 20GB container disk and zero volume')


def setup_resources(state_path):
    """Explicit management-only setup: min0/max1, private template, no jobs."""
    state_path = Path(state_path)
    if state_path.exists():
        raise ValueError('Receipt already exists; reconcile its resources before using a new path')
    name = 'lafa-validation-' + uuid.uuid4().hex
    state = {'name': name, 'endpoint_name': name, 'template_name': name,
             'created_at_unix': time.time(), 'job_submission_authorized': False,
             'status': 'setup-started', 'setup_verified': False}
    write_json(state_path, state)
    try:
        template_settings = {'name': name, 'imageName': IMAGE, 'isPublic': False,
            'isServerless': True, 'containerDiskInGb': 20, 'volumeInGb': 0,
            'ports': [], 'dockerEntrypoint': ['python', '-u', '-c'],
            'dockerStartCmd': ["import time;time.sleep(1)"]}
        # Persist intent before each POST. An uncertain response is reconciled by
        # this unique name; POST itself is never retried.
        state['status'] = 'template-creation-started'
        write_json(state_path, state)
        template = management('/templates', 'POST', template_settings)
        state['template_id'] = template['id']
        state['status'] = 'endpoint-creation-started'
        write_json(state_path, state)
        endpoint = management('/endpoints', 'POST', {
            'name': name, 'templateId': state['template_id'], 'computeType': 'GPU',
            'gpuCount': 1, 'gpuTypeIds': ['NVIDIA RTX A4000'], 'workersMin': 0,
            'workersMax': 1, 'idleTimeout': 5, 'executionTimeoutMs': EXECUTION_MS,
            'minCudaVersion': '12.8', 'scalerType': 'REQUEST_COUNT',
            'scalerValue': 1, 'flashboot': False})
        state['endpoint_id'] = endpoint['id']
        write_json(state_path, state)
        verified = verify_owned(state)
        template = retry_management('/templates/' + state['template_id'])
        require_storage(verified, template)
        for field, expected in (('workersMin', 0), ('workersMax', 1), ('gpuCount', 1),
                                ('gpuTypeIds', ['NVIDIA RTX A4000']), ('idleTimeout', 5),
                                ('executionTimeoutMs', EXECUTION_MS), ('minCudaVersion', '12.8')):
            if verified.get(field) != expected:
                raise ValueError('Provider changed setup field: ' + field)
        if template.get('isPublic', False) is not False or template.get('imageName') != IMAGE:
            raise ValueError('Provider changed private template image/settings')
        state.update(status='idle-no-jobs', setup_verified=True)
        write_json(state_path, state)
        return state
    except Exception as original:
        state['setup_error'] = str(original)
        unknown_field = ('template_id' if state['status'] == 'template-creation-started' else 'endpoint_id')
        unknown_post = unknown_field not in state
        state['uncertain_creation'] = unknown_field if unknown_post else None
        state['reconcile_required'] = unknown_post
        try:
            # Reconcile unknown POST outcomes by exact unique resource name,
            # then clean up only those resources. No job or retry POST occurs.
            for collection, field in (('endpoints', 'endpoint_id'), ('templates', 'template_id')):
                matches = [r for r in retry_management('/' + collection) if r.get('name') == name]
                if len(matches) > 1:
                    raise ValueError('Ambiguous setup resources; manual reconciliation required')
                if matches:
                    state[field] = matches[0]['id']
            write_json(state_path, state)
            cleanup(state)
            state['reconcile_required'] = unknown_post and unknown_field not in state
            state['cleanup_confirmed'] = not state['reconcile_required']
        except Exception as error:
            state['cleanup_confirmed'] = False
            state['cleanup_error'] = str(error)
        state['status'] = 'setup-failed'
        write_json(state_path, state)
        raise


def load_state(path, *, allow_partial=False):
    state = json.loads(Path(path).read_text())
    for field in ('endpoint_id', 'template_id'):
        if allow_partial and field not in state:
            continue
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', state.get(field, '')):
            raise ValueError('Invalid task resource ID in state')
    if not state.get('endpoint_name', '').startswith('lafa-validation-'):
        raise ValueError('State must identify a task-specific validation endpoint')
    if allow_partial and not all(field in state for field in ('endpoint_id', 'template_id')):
        name = state['endpoint_name']
        if not re.fullmatch(r'lafa-validation-[0-9a-f]{32}', name) or state.get('template_name') != name:
            raise ValueError('Partial receipt requires its exact unique setup names')
    return state


def cleanup_receipt(path):
    """Recover interrupted creation by its saved unique name, then stop/delete."""
    state = load_state(path, allow_partial=True)
    try:
        for collection, field, name_field in (('endpoints', 'endpoint_id', 'endpoint_name'),
                                              ('templates', 'template_id', 'template_name')):
            if field in state:
                continue
            matches = [resource for resource in retry_management('/' + collection)
                       if resource.get('name') == state[name_field]]
            if len(matches) > 1:
                raise ValueError('Ambiguous setup resources; manual reconciliation required')
            if matches:
                resource_id = matches[0].get('id', '')
                if not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', resource_id):
                    raise ValueError('Invalid reconciled resource ID')
                state[field] = resource_id
        write_json(path, state)
        cleanup(state)
        uncertain = state.get('uncertain_creation')
        if state.get('reconcile_required') and (not uncertain or uncertain not in state):
            raise RuntimeError('Creation outcome remains unresolved; repeat cleanup after provider reconciliation')
        state.update(cleanup_confirmed=True, reconcile_required=False)
    except Exception as error:
        state.update(cleanup_confirmed=False, cleanup_error=str(error))
        raise
    finally:
        write_json(path, state)


def verify_owned(state):
    endpoint = retry_management('/endpoints/' + state['endpoint_id'])
    if (endpoint.get('name') != state['endpoint_name']
            or endpoint.get('templateId') != state['template_id']):
        raise ValueError('Refusing operation on endpoint outside recorded task ownership')
    template = retry_management('/templates/' + state['template_id'])
    if not template.get('name', '').startswith('lafa-validation-'):
        raise ValueError('Refusing operation on template outside task ownership')
    return endpoint


def bootstrap_command(worker_source):
    encoded = base64.b64encode(gzip.compress(worker_source.encode(), mtime=0)).decode()
    # Only worker source is embedded, never RUNPOD_API_KEY or the local environment.
    return ("import base64,gzip,pathlib,subprocess,sys,threading,os; "
            "t=threading.Timer(1500,lambda:os._exit(124));t.daemon=True;t.start(); "
            "subprocess.run([sys.executable,'-m','pip','install','--disable-pip-version-check',"
            "'--no-cache-dir','runpod==1.12.0'],check=True,timeout=120); "
            f"p=pathlib.Path('/tmp/lafa-runpod-worker.py');p.write_bytes(gzip.decompress(base64.b64decode('{encoded}'))); "
            "subprocess.run([sys.executable,'-u',str(p)],check=True,timeout=1380)")


def collect(response, directory, run_id):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / 'provider-response.json', response)
    output = normalize_output(response)
    if output.get('run_id') != run_id:
        raise ValueError('Returned artifact belongs to a different run')
    data = base64.b64decode(output['artifact_b64'], validate=True)
    if len(data) > 6_000_000 or digest(data) != output.get('artifact_sha256'):
        raise ValueError('Result artifact size/checksum mismatch')
    (directory / 'results.tar.gz').write_bytes(data)
    safe_extract(data, directory / 'results')
    metadata = json.loads((directory / 'results' / 'metadata.json').read_text())
    if metadata.get('run_id') != run_id:
        raise ValueError('Artifact metadata belongs to a different run')
    return (response.get('status') == 'COMPLETED' and output.get('success') is True
            and metadata.get('success') is True and metadata.get('cuda_executed') is True)



def cleanup_once(state):
    """Reconcile a task's resources without requiring a template read to stop GPU."""
    endpoint_id, template_id = state.get('endpoint_id'), state.get('template_id')
    endpoints = retry_management('/endpoints')
    endpoint = next((e for e in endpoints if e['id'] == endpoint_id), None)
    if endpoint:
        # A matching endpoint list entry establishes ownership before any stop.
        # Template availability is deliberately not a prerequisite for stopping.
        if (endpoint.get('name') != state['endpoint_name']
                or endpoint.get('templateId') != template_id):
            raise ValueError('Endpoint ownership mismatch during cleanup')
        try:
            retry_management('/endpoints/' + endpoint_id, 'PATCH', {'workersMin': 0, 'workersMax': 0})
        except (RuntimeError, OSError):
            pass  # DELETE is still required if scaling fails or max0 is coerced.
        try:
            retry_management('/endpoints/' + endpoint_id, 'DELETE')
        except (RuntimeError, OSError):
            # A successful deletion may have lost its response; list reconciles it.
            pass
        endpoints = retry_management('/endpoints')
        if any(e['id'] == endpoint_id for e in endpoints):
            raise RuntimeError('Task endpoint deletion unconfirmed')
    # A template is free, but still remove it only after endpoint shutdown and
    # after establishing no other endpoint references it.
    if template_id and not any(e.get('templateId') == template_id for e in endpoints):
        template = next((t for t in retry_management('/templates') if t['id'] == template_id), None)
        if template:
            if (not template.get('name', '').startswith('lafa-validation-')
                    or state.get('template_name', template['name']) != template['name']):
                raise ValueError('Template ownership mismatch')
            try:
                retry_management('/templates/' + template_id, 'DELETE')
            except (RuntimeError, OSError):
                pass
        if any(t['id'] == template_id for t in retry_management('/templates')):
            raise RuntimeError('Task template deletion unconfirmed')


def cleanup(state, rounds=3):
    """Bounded reconciliation retries survive transient prerequisite/list errors."""
    for attempt in range(rounds):
        try:
            return cleanup_once(state)
        except (RuntimeError, OSError):
            if attempt + 1 == rounds:
                raise
            time.sleep(min(10, 2 ** attempt))


def watchdog(state_path, deadline):
    while time.time() < deadline:
        time.sleep(min(10, max(.01, deadline - time.time())))
    state = load_state(state_path)
    cleanup(state, rounds=5)


def run(args, state):
    # Scope must work before mutations, budget reservations, or any paid allocation.
    health = api('api.runpod.ai', '/v2/' + state['endpoint_id'] + '/health')
    require_idle_health(health)
    endpoint = verify_owned(state)
    require_storage(endpoint, retry_management('/templates/' + state['template_id']))
    if endpoint.get('workersMin') != 0 or endpoint.get('workersMax') != 1:
        raise ValueError('Require verified workersMin0/max1 before proceeding')
    payload = json.loads(args.payload.read_text())
    raw = base64.b64decode(payload['payload_b64'], validate=True)
    if digest(raw) != payload.get('payload_sha256') or digest(raw) != args.approved_payload_sha256:
        raise ValueError('Payload differs from explicitly reviewed checksum')
    worker = Path(__file__).with_name('runpod_worker.py').read_text()
    if payload.get('worker_sha256') != digest(worker.encode()):
        raise ValueError('Worker changed after payload approval; prepare/review again')
    if len(json.dumps(payload).encode()) > 9_000_000:
        raise ValueError('Payload request too large')
    run_id = 'lafa-' + uuid.uuid4().hex
    rate = Decimal(str(args.hourly_ceiling))
    if not rate.is_finite() or not 0 < rate <= Decimal('.70'):
        raise ValueError('Only reviewed Serverless ceiling up to $0.70/hour is permitted')
    reservation = max(Decimal(1), ((rate + Decimal('.003')) / 2 + 1).quantize(Decimal('.01'), rounding=ROUND_UP))
    reserve_budget(args.ledger, run_id, reservation, args.run_budget)
    output = args.output / run_id
    output.mkdir(parents=True, exist_ok=False)
    receipt = {'run_id': run_id, 'endpoint_id': state['endpoint_id'], 'template_id': state['template_id'],
               'payload_sha256': digest(raw), 'reservation_usd': float(reservation),
               'execution_timeout_ms': EXECUTION_MS, 'ttl_ms': TTL_MS, 'job_submitted': False}
    write_json(output / 'receipt.json', receipt)
    ownership_snapshot = output / 'ownership.json'
    write_json(ownership_snapshot, state)
    watcher = None
    try:
        # A detached cloud watchdog complements provider TTL; it cannot survive loss of this machine.
        watcher_log = (output / 'watchdog.log').open('ab')
        watcher = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '_watchdog',
                                    '--state', str(ownership_snapshot.resolve()), '--deadline', str(time.time() + 1800)],
                                   stdout=watcher_log, stderr=subprocess.STDOUT, start_new_session=True)
        watcher_log.close()
        template_settings = {
            'imageName': IMAGE, 'isServerless': True, 'isPublic': False,
            'containerDiskInGb': 20, 'volumeInGb': 0, 'ports': [],
            'dockerEntrypoint': ['python', '-u', '-c'],
            'dockerStartCmd': [bootstrap_command(worker)]}
        management('/templates/' + state['template_id'], 'PATCH', template_settings)
        updated_template = retry_management('/templates/' + state['template_id'])
        for field in ('imageName', 'dockerEntrypoint', 'dockerStartCmd', 'isServerless', 'isPublic', 'volumeInGb', 'containerDiskInGb'):
            # RunPod omits these Pod-only/default-false fields on Serverless GET.
            default = {'isPublic': False, 'volumeInGb': 0}.get(field)
            if updated_template.get(field, default) != template_settings[field]:
                raise ValueError('Provider changed required template field: ' + field)
        settings = {'workersMin': 0, 'workersMax': 1, 'gpuCount': 1,
                    'gpuTypeIds': ['NVIDIA RTX A4000'], 'idleTimeout': 5,
                    'executionTimeoutMs': EXECUTION_MS, 'flashboot': False, 'minCudaVersion': '12.8'}
        management('/endpoints/' + state['endpoint_id'], 'PATCH', settings)
        verified = verify_owned(state)
        require_storage(verified, updated_template)
        for field in ('workersMin', 'workersMax', 'gpuCount', 'gpuTypeIds', 'idleTimeout', 'executionTimeoutMs', 'minCudaVersion'):
            if verified.get(field) != settings[field]:
                raise ValueError('Provider changed required endpoint field: ' + field)
        body = {'input': {**payload, 'run_id': run_id, 'accuracy': args.accuracy},
                'policy': {'executionTimeout': EXECUTION_MS, 'ttl': TTL_MS}}
        require_idle_health(api('api.runpod.ai', '/v2/' + state['endpoint_id'] + '/health'))
        receipt['submission_started_unix'] = time.time()
        write_json(output / 'receipt.json', receipt)
        # Deliberately no automatic retry: an uncertain HTTP response may hide a real job.
        response = api('api.runpod.ai', '/v2/' + state['endpoint_id'] + '/run', 'POST', body)
        receipt.update(job_id=response['id'], job_submitted=True)
        write_json(output / 'receipt.json', receipt)
        deadline = time.monotonic() + TTL_MS / 1000 + 60
        while response.get('status') not in TERMINAL and time.monotonic() < deadline:
            time.sleep(5)
            response = api('api.runpod.ai', '/v2/' + state['endpoint_id'] + '/status/' + receipt['job_id'])
            write_json(output / 'provider-response.json', response)
        return 0 if collect(response, output, run_id) else 1
    except Exception as error:
        receipt['controller_error'] = str(error)
        raise
    finally:
        try:
            cleanup(state)
            receipt['cleanup_confirmed'] = True
        except Exception as error:
            receipt['cleanup_confirmed'] = False
            receipt['cleanup_error'] = str(error)
            raise
        finally:
            write_json(output / 'receipt.json', receipt)
            if watcher and receipt.get('cleanup_confirmed'):
                watcher.terminate()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'prepare', 'health', 'run', 'cleanup', '_watchdog'])
    parser.add_argument('--state', type=Path, help='Task ownership receipt from setup')
    parser.add_argument('--payload', type=Path)
    parser.add_argument('--output', type=Path, default=Path('/tmp/lafa-runpod-results'))
    parser.add_argument('--ledger', type=Path, help='Persistent ledger shared by all GPU attempts')
    parser.add_argument('--approved-payload-sha256', help='Exact checksum approved after payload review')
    parser.add_argument('--execute', action='store_true', help='Explicitly submit one billable job')
    parser.add_argument('--accuracy', action='store_true', help='Also run a small actual CUDA accuracy budget')
    parser.add_argument('--run-budget', type=float, default=3)
    parser.add_argument('--hourly-ceiling', type=float, default=.70)
    parser.add_argument('--deadline', type=float)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        if not args.payload:
            parser.error('prepare requires --payload path')
        payload = build_payload(ROOT)
        payload['worker_sha256'] = digest(Path(__file__).with_name('runpod_worker.py').read_bytes())
        write_json(args.payload, payload)
        print(json.dumps({'payload_sha256': payload['payload_sha256'], 'worker_sha256': payload['worker_sha256'],
                          'payload_bytes': args.payload.stat().st_size, 'path': str(args.payload)}))
        return 0
    if not args.state:
        parser.error('this action requires --state')
    if args.action == 'setup':
        print(json.dumps(setup_resources(args.state)))
        return 0
    if args.action == 'cleanup':
        cleanup_receipt(args.state)
        print('Task endpoint and unused task template deletion confirmed')
        return 0
    state = load_state(args.state)
    if args.action == 'health':
        verify_owned(state)
        print(json.dumps(api('api.runpod.ai', '/v2/' + state['endpoint_id'] + '/health')))
        return 0
    if args.action == '_watchdog':
        if not args.deadline or not math.isfinite(args.deadline):
            parser.error('watchdog requires an absolute deadline')
        watchdog(args.state, args.deadline)
        return 0
    if not args.execute or not args.approved_payload_sha256 or not args.payload or not args.ledger:
        parser.error('run requires --execute, --payload, --approved-payload-sha256 and --ledger')
    return run(args, state)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
