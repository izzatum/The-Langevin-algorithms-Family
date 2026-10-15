"""Local safety checks; these tests never call a cloud API or allocate a GPU."""
import base64
import importlib.util
import io
import json
from pathlib import Path
import sys
import tarfile
import time
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


def load_tool(name):
    path = ROOT / 'scripts' / (name + '.py')
    assert path.exists(), f'Missing validation tool: {name}'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def archive(name, content=b'ok', kind=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        info = tarfile.TarInfo(name)
        if kind is not None:
            info.type = kind
            info.linkname = '/tmp/escape'
        else:
            info.size = len(content)
        tar.addfile(info, io.BytesIO(content))
    return buffer.getvalue()


def test_payload_excludes_secrets_and_clears_old_notebook_outputs(tmp_path):
    tool = load_tool('runpod_validate')
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'sample.py').write_text('x = 1\n')
    (tmp_path / 'src' / '.env').write_text('API_KEY=do-not-upload')
    (tmp_path / 'notebooks').mkdir()
    (tmp_path / 'notebooks' / 'demo.ipynb').write_text(json.dumps({'cells': [
        {'cell_type': 'code', 'outputs': [{'text': 'old cpu output'}], 'execution_count': 7}]}))
    payload = tool.build_payload(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload['payload_b64']))) as tar:
        assert tar.getnames() == ['notebooks/demo.ipynb', 'src/sample.py']
        notebook = json.load(tar.extractfile('notebooks/demo.ipynb'))
        assert notebook['cells'][0]['outputs'] == []
        assert notebook['cells'][0]['execution_count'] is None
    assert payload['payload_sha256'] == tool.digest(base64.b64decode(payload['payload_b64']))


@pytest.mark.parametrize('name,kind', [('../escape', None), ('/tmp/escape', None),
                                      ('link', tarfile.SYMTYPE), ('link', tarfile.LNKTYPE)])
def test_archive_rejects_paths_and_links(tmp_path, name, kind):
    worker = load_tool('runpod_worker')
    with pytest.raises(ValueError):
        worker.safe_extract(archive(name, kind=kind), tmp_path / 'extract')
    assert not (tmp_path / 'escape').exists()


def test_archive_round_trip_and_byte_limit(tmp_path):
    worker = load_tool('runpod_worker')
    worker.safe_extract(archive('result.json', b'{"ok":true}'), tmp_path / 'ok')
    assert json.loads((tmp_path / 'ok' / 'result.json').read_text()) == {'ok': True}
    with pytest.raises(ValueError, match='size|limit'):
        worker.safe_extract(archive('big', b'abc'), tmp_path / 'limited', max_bytes=2)


def test_budget_reservations_never_exceed_total_or_reset_existing_ledger(tmp_path):
    tool = load_tool('runpod_validate')
    ledger = tmp_path / 'budget.json'
    for index in range(10):
        tool.reserve_budget(ledger, f'run-{index}', reservation=1, run_budget=3)
    with pytest.raises(ValueError, match='10|budget'):
        tool.reserve_budget(ledger, 'over-budget', reservation=1, run_budget=3)
    assert json.loads(ledger.read_text())['reserved_usd'] == 10
    with pytest.raises(ValueError):
        tool.reserve_budget(ledger, 'oversized-run', reservation=4, run_budget=3)


def test_failed_job_output_is_preserved_for_collection():
    tool = load_tool('runpod_validate')
    result = {'run_id': 'abc', 'artifact_b64': 'eA==', 'artifact_sha256': 'example'}
    response = {'status': 'FAILED', 'error': 'pytest failed', 'output': json.dumps(result)}
    assert tool.normalize_output(response) == result
    assert response['status'] == 'FAILED'


def test_command_failure_and_timeout_persist_real_status(tmp_path):
    worker = load_tool('runpod_worker')
    failed = worker.run_command([sys.executable, '-c', "print('failure details');raise SystemExit(7)"],
                                tmp_path, tmp_path / 'failure.log', time.monotonic() + 5)
    assert failed['returncode'] == 7
    assert 'failure details' in (tmp_path / 'failure.log').read_text()
    timed = worker.run_command([sys.executable, '-c', 'import time;time.sleep(5)'],
                               tmp_path, tmp_path / 'timeout.log', time.monotonic() + .1)
    assert timed['timed_out'] is True
    assert timed['returncode'] != 0


def test_malformed_payload_marks_sdk_job_failed_and_retains_artifact():
    worker = load_tool('runpod_worker')
    result = worker.handler({'input': {'run_id': 'bad-payload', 'payload_b64': 'invalid-base64'}})
    assert result['error']
    assert result['success'] is False
    assert result['artifact_b64']
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(result['artifact_b64']))) as tar:
        metadata = json.load(tar.extractfile('metadata.json'))
        assert metadata['success'] is False
        assert metadata['error']


def test_result_collection_keeps_failed_artifacts_and_rejects_wrong_identity(tmp_path):
    worker, tool = load_tool('runpod_worker'), load_tool('runpod_validate')
    results = tmp_path / 'remote'
    results.mkdir()
    (results / 'metadata.json').write_text(json.dumps({'run_id': 'expected', 'success': False,
                                                      'cuda_executed': True}))
    response = {'status': 'FAILED', 'error': 'pytest failed', 'output': {
        'run_id': 'expected', 'success': False, **worker.bundle_results(results)}}
    assert tool.collect(response, tmp_path / 'local', 'expected') is False
    assert (tmp_path / 'local' / 'results' / 'metadata.json').exists()
    with pytest.raises(ValueError, match='different run'):
        tool.collect(response, tmp_path / 'wrong', 'other')


def test_budget_rejects_nonfinite_values_and_duplicate_run(tmp_path):
    tool = load_tool('runpod_validate')
    ledger = tmp_path / 'ledger.json'
    for value in ('NaN', 'Infinity', '-1'):
        with pytest.raises(ValueError):
            tool.reserve_budget(ledger, 'invalid', reservation=value)
    tool.reserve_budget(ledger, 'one')
    with pytest.raises(ValueError, match='already'):
        tool.reserve_budget(ledger, 'one')


def test_bootstrap_contains_only_worker_source_and_is_valid_python():
    tool = load_tool('runpod_validate')
    command = tool.bootstrap_command('print("ready")')
    compile(command, '<bootstrap>', 'exec')
    assert 'runpod==1.12.0' in command
    assert 'timeout=120' in command
    assert 'RUNPOD_API_KEY' not in command


def test_cpu_runtime_cannot_be_reported_as_cuda(tmp_path, monkeypatch):
    worker, tool = load_tool('runpod_worker'), load_tool('runpod_validate')
    import torch
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    payload = tool.build_payload(tmp_path)
    payload['worker_sha256'] = tool.digest(Path(worker.__file__).read_bytes())
    result = worker.handler({'input': {'run_id': 'cpu-is-not-cuda', **payload}})
    assert result['success'] is False
    assert 'CUDA' in result['error']
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(result['artifact_b64']))) as tar:
        metadata = json.load(tar.extractfile('metadata.json'))
        assert metadata['cuda_executed'] is False
        assert metadata['stages'] == []


def test_corrupted_budget_total_is_not_trusted(tmp_path):
    tool = load_tool('runpod_validate')
    ledger = tmp_path / 'corrupt.json'
    ledger.write_text(json.dumps({'hard_cap_usd': 10, 'reserved_usd': 0,
                                  'attempts': [{'run_id': 'old', 'reservation_usd': 3}]}))
    with pytest.raises(ValueError, match='ledger|reconciliation'):
        tool.reserve_budget(ledger, 'new')


def idle_health():
    return {'jobs': {'completed': 0, 'failed': 0, 'inProgress': 0, 'inQueue': 0, 'retried': 0},
            'workers': {'idle': 0, 'running': 0}}


def acknowledge_watchdog(command, **kwargs):
    """Replace only the detached process; retain real receipt/budget handling."""
    ready = Path(command[command.index('--ready-file') + 1])
    ready.write_text(command[command.index('--ownership-sha256') + 1])
    return SimpleNamespace(pid=123, poll=lambda: None, terminate=lambda: None)


@pytest.mark.parametrize('section,field', [('jobs', 'inQueue'), ('jobs', 'inProgress'),
                                          ('jobs', 'completed'), ('workers', 'running'),
                                          ('workers', 'initializing')])
def test_preflight_rejects_active_or_previous_work(section, field):
    tool = load_tool('runpod_validate')
    health = idle_health()
    health[section][field] = 1
    with pytest.raises(ValueError, match='idle|work|job'):
        tool.require_idle_health(health)


@pytest.mark.parametrize('resource,changes', [('endpoint', {'networkVolumeId': 'existing'}),
    ('endpoint', {'networkVolumeIds': ['existing']}), ('template', {'containerDiskInGb': 10000}),
    ('template', {'volumeInGb': 20})])
def test_preflight_rejects_persistent_volumes_and_unexpected_disk(resource, changes):
    tool = load_tool('runpod_validate')
    endpoint, template = {}, {'containerDiskInGb': 20, 'volumeInGb': 0}
    (endpoint if resource == 'endpoint' else template).update(changes)
    with pytest.raises(ValueError, match='volume|disk'):
        tool.require_storage(endpoint, template)


def test_cleanup_survives_transient_reads_without_losing_stop_delete(monkeypatch):
    tool = load_tool('runpod_validate')
    state = {'endpoint_id': 'endpoint', 'template_id': 'template',
             'endpoint_name': 'lafa-validation-test'}
    endpoint = {'id': 'endpoint', 'name': state['endpoint_name'], 'templateId': 'template'}
    template = {'id': 'template', 'name': 'lafa-validation-test'}
    resources = {'endpoints': [endpoint], 'templates': [template]}
    calls, failures = [], {'/endpoints': 1, '/templates': 1}
    def fake(path, method='GET', body=None):
        calls.append((method, path))
        if method == 'GET' and failures.get(path, 0):
            failures[path] -= 1
            raise RuntimeError('HTTP503 temporary unavailable')
        if method == 'GET':
            return resources[path[1:]]
        if method == 'DELETE':
            resources[path.split('/')[1]] = []
        return {'workersMin': 0, 'workersMax': 3}  # PATCH coercion never substitutes for DELETE.
    monkeypatch.setattr(tool, 'management', fake)
    monkeypatch.setattr(tool.time, 'sleep', lambda _: None)
    tool.cleanup(state)
    assert ('PATCH', '/endpoints/endpoint') in calls
    assert ('DELETE', '/endpoints/endpoint') in calls
    assert ('DELETE', '/templates/template') in calls
    assert resources == {'endpoints': [], 'templates': []}


def test_setup_creates_only_private_idle_resources_and_durable_receipt(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    resources, mutations = {}, []
    ledger = tmp_path / 'ledger.json'
    def fake(path, method='GET', body=None):
        if method == 'POST':
            # Endpoint creation can allocate an idle GPU without a submitted job.
            assert json.loads(ledger.read_text())['reserved_usd'] == 1
            assert list(tmp_path.glob('*.ready'))
            mutations.append((path, body))
            resource = {**body, 'id': 'template' if path == '/templates' else 'endpoint'}
            resources[path + '/' + resource['id']] = resource
            return resource
        return resources[path]
    monkeypatch.setattr(tool, 'management', fake)
    monkeypatch.setattr(tool, 'api', lambda *a, **k: pytest.fail('setup must use management only'))
    monkeypatch.setattr(tool.subprocess, 'Popen', acknowledge_watchdog)
    receipt = tmp_path / 'state.json'
    state = tool.setup_resources(receipt, ledger)
    assert state['endpoint_id'] == 'endpoint'
    assert json.loads(receipt.read_text())['setup_verified'] is True
    assert mutations[0][1]['isPublic'] is False
    assert mutations[1][1]['workersMin'] == 0
    assert mutations[1][1]['workersMax'] == 1
    assert all('/run' not in path for path, _ in mutations)
    assert state['ledger_path'] == str(ledger.resolve())
    assert state['cleanup_deadline_unix'] - state['created_at_unix'] == 1800
    with pytest.raises(ValueError, match='exists'):
        tool.setup_resources(receipt, ledger)


def test_setup_budget_cap_blocks_every_remote_creation(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    ledger = tmp_path / 'ledger.json'
    for index in range(10):
        tool.reserve_budget(ledger, f'previous-{index}')
    monkeypatch.setattr(tool, 'management', lambda *a, **k: pytest.fail('No API call after exhausted budget'))
    monkeypatch.setattr(tool.subprocess, 'Popen', lambda *a, **k: pytest.fail('No watchdog needed'))
    with pytest.raises(ValueError, match='10|budget'):
        tool.setup_resources(tmp_path / 'state.json', ledger)
    assert json.loads(ledger.read_text())['reserved_usd'] == 10


def test_setup_cli_requires_shared_ledger_before_creation(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    monkeypatch.setattr(tool, 'management', lambda *a, **k: pytest.fail('No unbudgeted setup'))
    with pytest.raises(SystemExit):
        tool.main(['setup', '--state', str(tmp_path / 'state.json')])


def test_setup_does_not_create_when_watchdog_could_expire_during_post(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    now = [100.]
    monkeypatch.setattr(tool.time, 'time', lambda: now[0])
    def stalled_spawn(command, **kwargs):
        now[0] += 1770  # Only 30 seconds remain after a stalled parent resumes.
        return acknowledge_watchdog(command, **kwargs)
    monkeypatch.setattr(tool.subprocess, 'Popen', stalled_spawn)
    def management(path, method='GET', body=None):
        assert method == 'GET', 'No creation may race the cleanup deadline'
        return []
    monkeypatch.setattr(tool, 'management', management)
    with pytest.raises(ValueError, match='lifetime'):
        tool.setup_resources(tmp_path / 'state.json', tmp_path / 'ledger.json')


def test_setup_refuses_creation_if_watchdog_exits_before_ready(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    monkeypatch.setattr(tool.subprocess, 'Popen', lambda *a, **k: SimpleNamespace(poll=lambda: 1))
    def management(path, method='GET', body=None):
        assert method == 'GET', 'No allocation without an armed watchdog'
        return []
    monkeypatch.setattr(tool, 'management', management)
    with pytest.raises(RuntimeError, match='watchdog exited'):
        tool.setup_resources(tmp_path / 'state.json', tmp_path / 'ledger.json')


def test_unknown_setup_response_never_retries_post_and_preserves_reconciliation(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    posts = []
    def fake(path, method='GET', body=None):
        if method == 'POST':
            posts.append(path)
            raise OSError('response connection lost')
        return []
    monkeypatch.setattr(tool, 'management', fake)
    monkeypatch.setattr(tool.subprocess, 'Popen', acknowledge_watchdog)
    state_path = tmp_path / 'state.json'
    with pytest.raises(OSError):
        tool.setup_resources(state_path, tmp_path / 'ledger.json')
    assert posts == ['/templates']
    state = json.loads(state_path.read_text())
    assert state['reconcile_required'] is True
    assert state['cleanup_confirmed'] is False


def test_status_failure_cleans_up_after_single_submission_and_snapshots_ownership(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    state = {'endpoint_id': 'endpoint', 'template_id': 'template', 'endpoint_name': 'lafa-validation-run'}
    ledger = tmp_path / 'ledger.json'
    tool.reserve_budget(ledger, 'setup-test')
    state.update(ledger_path=str(ledger.resolve()), setup_reservation_id='setup-test',
                 cleanup_deadline_unix=time.time() + 1800)
    endpoint = {'id': 'endpoint', 'name': state['endpoint_name'], 'templateId': 'template',
                'workersMin': 0, 'workersMax': 1}
    template = {'id': 'template', 'name': 'lafa-validation-run', 'volumeInGb': 0, 'containerDiskInGb': 20}
    resources = {'endpoint': endpoint, 'template': template}
    def management(path, method='GET', body=None):
        if path in ('/endpoints', '/templates'):
            key = 'endpoint' if path == '/endpoints' else 'template'
            return [resources[key]] if resources.get(key) else []
        key = 'endpoint' if path.startswith('/endpoints/') else 'template'
        if method == 'PATCH':
            resources[key].update(body)
        if method == 'DELETE':
            resources[key] = None
        return resources.get(key)
    submissions = []
    def api(host, path, method='GET', body=None):
        if path.endswith('/health'):
            return idle_health()
        if path.endswith('/run'):
            submissions.append(body)
            return {'id': 'job', 'status': 'IN_QUEUE'}
        raise RuntimeError('temporary status failure')
    watchdog_args = []
    def spawn(command, **kwargs):
        watchdog_args.extend(command)
        return acknowledge_watchdog(command, **kwargs)
    monkeypatch.setattr(tool, 'management', management)
    monkeypatch.setattr(tool, 'api', api)
    monkeypatch.setattr(tool.subprocess, 'Popen', spawn)
    monkeypatch.setattr(tool.time, 'sleep', lambda _: None)
    payload = tool.build_payload(tmp_path)
    payload['worker_sha256'] = tool.digest((ROOT / 'scripts/runpod_worker.py').read_bytes())
    payload_path = tmp_path / 'payload.json'
    payload_path.write_text(json.dumps(payload))
    state_path = tmp_path / 'mutable-state.json'
    state_path.write_text(json.dumps(state))
    args = SimpleNamespace(payload=payload_path, approved_payload_sha256=payload['payload_sha256'],
        hourly_ceiling=.70, ledger=ledger, run_budget=3, output=tmp_path / 'out',
        state=state_path, accuracy=False)
    with pytest.raises(RuntimeError, match='status failure'):
        tool.run(args, state)
    assert len(submissions) == 1
    assert resources == {'endpoint': None, 'template': None}
    receipt = json.loads(next((tmp_path / 'out').glob('*/receipt.json')).read_text())
    assert receipt['cleanup_confirmed'] is True
    assert 'status failure' in receipt['controller_error']
    snapshot = Path(watchdog_args[watchdog_args.index('--state') + 1])
    assert snapshot != state_path
    assert json.loads(snapshot.read_text())['endpoint_id'] == 'endpoint'


@pytest.mark.parametrize('legacy', [False, True])
def test_run_rejects_different_or_missing_setup_ledger_before_api(tmp_path, monkeypatch, legacy):
    tool = load_tool('runpod_validate')
    state = {'endpoint_id': 'endpoint', 'template_id': 'template', 'endpoint_name': 'lafa-validation-run'}
    if not legacy:
        state.update(ledger_path=str(tmp_path / 'original-ledger.json'),
                     setup_reservation_id='setup-test', cleanup_deadline_unix=time.time() + 1800)
    monkeypatch.setattr(tool, 'api', lambda *a, **k: pytest.fail('Ledger check must precede API'))
    args = SimpleNamespace(ledger=tmp_path / 'new-ledger.json')
    with pytest.raises(ValueError, match='ledger|setup'):
        tool.run(args, state)


def test_watchdog_recovers_ids_from_immutable_partial_ownership(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    name = 'lafa-validation-' + 'c' * 32
    receipt = tmp_path / 'ownership.json'
    receipt.write_text(json.dumps({'endpoint_name': name, 'template_name': name}))
    checksum = tool.digest(receipt.read_bytes())
    resources = {'endpoints': [{'id': 'late-endpoint', 'name': name, 'templateId': 'late-template'}],
                 'templates': [{'id': 'late-template', 'name': name}]}
    def management(path, method='GET', body=None):
        collection = path.split('/')[1]
        if method == 'DELETE':
            resources[collection] = []
        return resources[collection]
    monkeypatch.setattr(tool, 'management', management)
    # The child reads and checks the snapshot before sleeping. Later edits cannot redirect it.
    now = [0.]
    monkeypatch.setattr(tool.time, 'time', lambda: now[0])
    def sleep(_):
        receipt.write_text(json.dumps({'endpoint_name': 'lafa-validation-unrelated'}))
        now[0] = 11.
    monkeypatch.setattr(tool.time, 'sleep', sleep)
    tool.watchdog(receipt, 10, checksum)
    assert resources == {'endpoints': [], 'templates': []}


def test_watchdog_stops_unique_endpoint_before_reading_its_template(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    name = 'lafa-validation-' + 'd' * 32
    receipt = tmp_path / 'ownership.json'
    receipt.write_text(json.dumps({'endpoint_name': name, 'template_name': name}))
    resources = {'endpoints': [{'id': 'endpoint', 'name': name, 'templateId': 'template'}],
                 'templates': [{'id': 'template', 'name': name}]}
    def management(path, method='GET', body=None):
        if path == '/templates':
            assert not resources['endpoints'], 'GPU shutdown must precede template reads'
        collection = path.split('/')[1]
        if method == 'DELETE':
            resources[collection] = []
        return resources[collection]
    monkeypatch.setattr(tool, 'management', management)
    tool.watchdog(receipt, time.time() - 1, tool.digest(receipt.read_bytes()))
    assert resources == {'endpoints': [], 'templates': []}


def test_watchdog_reconciles_late_creation_after_initial_empty_inventory(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    name = 'lafa-validation-' + 'e' * 32
    receipt = tmp_path / 'ownership.json'
    receipt.write_text(json.dumps({'endpoint_name': name, 'template_name': name}))
    resources = {'endpoints': [], 'templates': []}
    appeared = []
    def sleep(_):
        if not appeared:
            resources['endpoints'] = [{'id': 'endpoint', 'name': name, 'templateId': 'template'}]
            resources['templates'] = [{'id': 'template', 'name': name}]
            appeared.append(True)
    def management(path, method='GET', body=None):
        collection = path.split('/')[1]
        if method == 'DELETE':
            resources[collection] = []
        return resources[collection]
    monkeypatch.setattr(tool.time, 'sleep', sleep)
    monkeypatch.setattr(tool, 'management', management)
    tool.watchdog(receipt, time.time() - 1, tool.digest(receipt.read_bytes()))
    assert appeared
    assert resources == {'endpoints': [], 'templates': []}


def test_watchdog_rejects_changed_snapshot_before_acknowledgement(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    receipt = tmp_path / 'ownership.json'
    receipt.write_text('{}')
    monkeypatch.setattr(tool, 'management', lambda *a, **k: pytest.fail('No API with altered snapshot'))
    ready = tmp_path / 'ready'
    with pytest.raises(ValueError, match='checksum'):
        tool.watchdog(receipt, time.time() - 1, '0' * 64, ready)
    assert not ready.exists()


@pytest.mark.parametrize('reserved,remaining', [(False, 1800), (True, 1259)])
def test_run_requires_recorded_setup_reservation_and_full_job_lifetime(tmp_path, monkeypatch,
                                                                    reserved, remaining):
    tool = load_tool('runpod_validate')
    ledger = tmp_path / 'ledger.json'
    tool.reserve_budget(ledger, 'setup-test' if reserved else 'unrelated-attempt')
    state = {'ledger_path': str(ledger.resolve()), 'setup_reservation_id': 'setup-test',
             'cleanup_deadline_unix': time.time() + remaining}
    monkeypatch.setattr(tool, 'api', lambda *a, **k: pytest.fail('No API with invalid setup reservation/lifetime'))
    with pytest.raises(ValueError, match='reservation|lifetime'):
        tool.run(SimpleNamespace(ledger=ledger), state)


def test_cleanup_command_recovers_partial_setup_receipt(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    name = 'lafa-validation-' + 'a' * 32
    receipt = tmp_path / 'partial.json'
    receipt.write_text(json.dumps({'endpoint_name': name, 'template_name': name,
        'status': 'setup-failed', 'reconcile_required': True,
        'uncertain_creation': 'template_id'}))
    resources = {'endpoints': [], 'templates': [{'id': 'late-template', 'name': name}]}
    def management(path, method='GET', body=None):
        if method == 'DELETE':
            assert path == '/templates/late-template'
            resources['templates'] = []
            return None
        return resources[path[1:]]
    monkeypatch.setattr(tool, 'management', management)
    assert tool.main(['cleanup', '--state', str(receipt)]) == 0
    state = json.loads(receipt.read_text())
    assert state['template_id'] == 'late-template'
    assert state['cleanup_confirmed'] is True
    assert state['reconcile_required'] is False
    assert not resources['templates']


def test_partial_receipt_cannot_run_or_claim_unobserved_creation_cleaned(tmp_path, monkeypatch):
    tool = load_tool('runpod_validate')
    name = 'lafa-validation-' + 'b' * 32
    receipt = tmp_path / 'partial.json'
    receipt.write_text(json.dumps({'endpoint_name': name, 'template_name': name,
        'status': 'setup-failed', 'reconcile_required': True,
        'uncertain_creation': 'template_id'}))
    with pytest.raises(ValueError, match='resource ID'):
        tool.load_state(receipt)
    monkeypatch.setattr(tool, 'management', lambda *a, **k: [])
    with pytest.raises(RuntimeError, match='unresolved'):
        tool.main(['cleanup', '--state', str(receipt)])
    assert json.loads(receipt.read_text())['cleanup_confirmed'] is False


def test_serverless_omits_pod_only_zero_volume_fields():
    tool = load_tool('runpod_validate')
    # Observed authenticated Serverless template GET: disk present, volume omitted.
    tool.require_storage({}, {'isServerless': True, 'containerDiskInGb': 20})
    with pytest.raises(ValueError, match='volume'):
        tool.require_storage({}, {'isServerless': False, 'containerDiskInGb': 20})
    with pytest.raises(ValueError, match='volumes'):
        tool.require_storage({'networkVolumeId': 'attached'},
                             {'isServerless': True, 'containerDiskInGb': 20})


def test_payload_contains_required_package_metadata(tmp_path):
    tool = load_tool('runpod_validate')
    for name in ['pyproject.toml', 'README.md', 'LICENSE']:
        (tmp_path / name).write_text(name)
    (tmp_path / 'unrelated-private.md').write_text('must not be included')
    payload = tool.build_payload(tmp_path)
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload['payload_b64'])), mode='r:gz') as archive:
        assert set(archive.getnames()) == {'pyproject.toml', 'README.md', 'LICENSE'}
