import json
import threading
import time

import pytest
import torch


def api():
    from lafa.recording import Recorder, TraceReader, RingBuffer, OnlineMoments
    return Recorder, TraceReader, RingBuffer, OnlineMoments


def test_owned_detached_selected_thinned_lossless(tmp_path):
    Recorder, Reader, _, _ = api()
    x = torch.arange(3., dtype=torch.float64, requires_grad=True)
    with Recorder(tmp_path, fields=['position'], every=2, chunk_size=2) as r:
        r.record(0, position=x * 2, unwanted=object())
        r.record(1, position=object())
        r.record(2, position=x)
        with torch.no_grad():
            x.fill_(99)
        assert r.memory_bytes <= r.max_buffer_bytes
    reader = Reader(tmp_path)
    got = reader.read('position')
    assert got.dtype == torch.float64 and not got.requires_grad
    torch.testing.assert_close(got, torch.tensor([[0., 2., 4.], [0., 1., 2.]], dtype=torch.float64))
    assert reader.manifest['fields'].keys() == {'position'}
    assert reader.manifest['last_step'] == 2
    assert r.memory_bytes == 0
    r.close()


def test_explicit_downcast_does_not_change_state(tmp_path):
    Recorder, Reader, _, _ = api()
    x = torch.ones(4, dtype=torch.float64)
    with Recorder(tmp_path, storage_dtype=torch.float32) as r:
        r.record(1, position=x)
    assert x.dtype == torch.float64
    assert Reader(tmp_path).read('position').dtype == torch.float32


def test_schema_and_order(tmp_path):
    Recorder, Reader, _, _ = api()
    with Recorder(tmp_path) as r:
        r.record(2, position=torch.ones(2))
        for fields in [dict(position=torch.ones(3)), dict(position=torch.ones(2, dtype=torch.float64)), dict(other=torch.ones(2))]:
            with pytest.raises(ValueError):
                r.record(3, **fields)
        with pytest.raises(ValueError):
            r.record(2, position=torch.ones(2))
        r.record(4, position=torch.ones(2))
    assert Reader(tmp_path).manifest['last_step'] == 4


def test_resume_atomic_manifest_and_orphan_cleanup(tmp_path):
    Recorder, Reader, _, _ = api()
    with Recorder(tmp_path, chunk_size=1) as r:
        r.record(0, x=torch.tensor([0.]))
        r.record(2, x=torch.tensor([2.]))
    (tmp_path / 'chunk-99999999.pt').write_bytes(b'incomplete')
    (tmp_path / '.chunk-0123456789abcdef0123456789abcdef.tmp').write_bytes(b'partial')
    with pytest.raises(FileExistsError):
        Recorder(tmp_path)
    with Recorder(tmp_path, resume=True) as r:
        with pytest.raises(ValueError):
            r.record(2, x=torch.tensor([20.]))
        r.record(3, x=torch.tensor([3.]))
    assert not (tmp_path / 'chunk-99999999.pt').exists()
    assert not (tmp_path / '.chunk-0123456789abcdef0123456789abcdef.tmp').exists()
    assert Reader(tmp_path).read('x').flatten().tolist() == [0., 2., 3.]
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['schema_version'] == 1
    assert len(manifest['chunks']) == 3


def test_empty_idempotent_close_and_invalid_options(tmp_path):
    Recorder, Reader, _, _ = api()
    with Recorder(tmp_path / 'empty') as r:
        r.flush()
    r.close()
    assert list(Reader(tmp_path / 'empty')) == []
    for kw in [dict(every=0), dict(chunk_size=0), dict(max_buffer_bytes=0), dict(fields=[])]:
        with pytest.raises(ValueError):
            Recorder(tmp_path / 'invalid', **kw)


def test_single_record_rejected_before_copy(tmp_path):
    Recorder, _, _, _ = api()
    with Recorder(tmp_path, max_buffer_bytes=16) as r:
        with pytest.raises(ValueError, match='max_buffer_bytes'):
            r.record(0, x=torch.zeros(100))
        assert r.memory_bytes == 0


def test_reader_slices_only_needed_chunks(tmp_path, monkeypatch):
    Recorder, Reader, _, _ = api()
    with Recorder(tmp_path, chunk_size=2) as r:
        for i in range(10):
            r.record(i, x=torch.tensor([i], dtype=torch.int64))
    original = torch.load
    calls = []
    def load(*args, **kwargs):
        assert kwargs['weights_only'] is True and kwargs['map_location'] == 'cpu'
        calls.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(torch, 'load', load)
    assert Reader(tmp_path).read('x', start=3, stop=5).flatten().tolist() == [3, 4]
    assert len(calls) == 2


def test_async_backpressure_and_fixed_memory_plateau(tmp_path, monkeypatch):
    Recorder, Reader, _, _ = api()
    entered, release = threading.Event(), threading.Event()
    original = Recorder._write_chunk
    def slow(self, records):
        entered.set()
        assert release.wait(5)
        return original(self, records)
    monkeypatch.setattr(Recorder, '_write_chunk', slow)
    r = Recorder(tmp_path, async_write=True, chunk_size=1, max_buffer_bytes=48)
    r.record(0, x=torch.ones(4))  # 16 tensor bytes + 8 step bytes
    assert entered.wait(3)
    r.record(1, x=torch.ones(4))
    done = threading.Event()
    def third():
        r.record(2, x=torch.ones(4))
        done.set()
    thread = threading.Thread(target=third)
    thread.start()
    assert not done.wait(.1)
    assert r.memory_bytes <= 48
    release.set()
    thread.join(5)
    assert done.is_set()
    for i in range(3, 403):
        r.record(i, x=torch.ones(4))
        assert r.memory_bytes <= 48
    r.close()
    assert r.stats['peak_buffer_bytes'] <= 48
    assert r.memory_bytes == 0
    assert len(Reader(tmp_path).read('x')) == 403


@pytest.mark.parametrize('asynchronous', [False, True])
def test_write_failure_surfaces_and_committed_trace_survives(tmp_path, monkeypatch, asynchronous):
    Recorder, Reader, _, _ = api()
    r = Recorder(tmp_path, chunk_size=1, async_write=asynchronous)
    r.record(0, x=torch.tensor(1.))
    r.flush()
    def fail(*args, **kwargs):
        raise OSError('disk full')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises((OSError, RuntimeError), match='disk full'):
        r.record(1, x=torch.tensor(2.))
        r.flush()
    with pytest.raises((OSError, RuntimeError), match='disk full'):
        r.close()
    assert Reader(tmp_path).read('x').tolist() == [1.]


def test_manifest_failure_leaves_orphan_for_resume(tmp_path, monkeypatch):
    Recorder, Reader, _, _ = api()
    r = Recorder(tmp_path, chunk_size=1)
    import lafa.recording as module
    original = module.os.replace
    def fail(source, target):
        if str(target).endswith('manifest.json'):
            raise OSError('manifest failure')
        return original(source, target)
    monkeypatch.setattr(module.os, 'replace', fail)
    with pytest.raises(OSError, match='manifest failure'):
        r.record(0, x=torch.tensor(1.))
    with pytest.raises((OSError, RuntimeError)):
        r.close()
    monkeypatch.setattr(module.os, 'replace', original)
    with Recorder(tmp_path, resume=True) as r:
        r.record(0, x=torch.tensor(2.))
    assert Reader(tmp_path).read('x').tolist() == [2.]


def test_ring_buffer_and_online_moments():
    _, _, Ring, Moments = api()
    ring = Ring(2, fields=['x'])
    moments = Moments()
    for i in range(5):
        x = torch.tensor([float(i), float(i + 2)], requires_grad=True)
        ring.record(i, x=x, ignored=object())
        moments.update(x)
        with torch.no_grad():
            x.fill_(-100)
    snapshots = list(ring)
    assert [entry['step'] for entry in snapshots] == [3, 4]
    assert not snapshots[-1]['fields']['x'].requires_grad
    assert snapshots[-1]['fields']['x'].tolist() == [4., 6.]
    torch.testing.assert_close(moments.mean, torch.tensor([2., 4.], dtype=torch.float64))
    torch.testing.assert_close(moments.variance, torch.tensor([2.5, 2.5], dtype=torch.float64))
    restored = Moments()
    restored.load_state_dict(moments.state_dict())
    restored.update(torch.tensor([5., 7.]))
    assert restored.count == 6
    torch.testing.assert_close(restored.mean, torch.tensor([2.5, 4.5], dtype=torch.float64))


def test_step_result_records_accepted_position_and_current_gradient(tmp_path):
    Recorder, Reader, _, _ = api()
    from lafa.samplers import MALA
    sampler = MALA(lambda x: x.square().sum(-1) / 2, torch.ones(100, 2), step_size=2., generator=torch.Generator().manual_seed(10))
    result = sampler.step()
    assert (~result.accepted).any()
    with Recorder(tmp_path) as r:
        r.record(1, position=result.position, gradient=result.gradient, proposal_gradient=torch.full_like(result.gradient, -77.))
    reader = Reader(tmp_path)
    torch.testing.assert_close(reader.read('position'), reader.read('gradient'))
    assert (reader.read('proposal_gradient') == -77).all()


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64, torch.bool, torch.complex64])
def test_tensor_dtypes_and_storage_estimate(tmp_path, dtype):
    Recorder, Reader, _, _ = api()
    from lafa.recording import estimate_storage_bytes
    x = torch.ones(3, dtype=dtype)
    assert estimate_storage_bytes(10, {'x': x}) == 10 * (8 + 3*x.element_size())
    with Recorder(tmp_path) as r:
        r.record(0, x=x)
    got = Reader(tmp_path).read('x')
    assert got.dtype == dtype and torch.equal(got[0], x)


def test_async_failed_worker_releases_pending_buffer(tmp_path, monkeypatch):
    Recorder, _, _, _ = api()
    entered, release = threading.Event(), threading.Event()
    def fail(self, records):
        entered.set()
        assert release.wait(5)
        raise OSError('worker failure')
    monkeypatch.setattr(Recorder, '_write_chunk', fail)
    r = Recorder(tmp_path, async_write=True, chunk_size=2)
    r.record(0, x=torch.ones(2))
    r.record(1, x=torch.ones(2))
    assert entered.wait(3)
    r.record(2, x=torch.ones(2))
    release.set()
    r._worker.join(5)
    with pytest.raises(RuntimeError, match='worker failure'):
        r.record(3, x=torch.ones(2))
    with pytest.raises(RuntimeError, match='worker failure'):
        r.close()
    assert r.memory_bytes == 0


def test_uncommitted_partial_chunk_does_not_affect_reader(tmp_path):
    Recorder, Reader, _, _ = api()
    with Recorder(tmp_path) as r:
        r.record(0, x=torch.ones(2))
    (tmp_path / 'chunk-00000001.pt').write_bytes(b'partial')
    assert Reader(tmp_path).read('x').tolist() == [[1., 1.]]
    with Recorder(tmp_path, resume=True) as r:
        r.record(1, x=torch.ones(2)*2)
    assert Reader(tmp_path).read('x').tolist() == [[1., 1.], [2., 2.]]


@pytest.mark.parametrize('chunk_size', [1, 2])
def test_worker_failure_during_producer_copy_cleans_reservations(tmp_path, monkeypatch, chunk_size):
    Recorder, _, _, _ = api()
    entered, release = threading.Event(), threading.Event()
    def fail(self, records):
        entered.set()
        assert release.wait(5)
        raise OSError('failure during staging')
    monkeypatch.setattr(Recorder, '_write_chunk', fail)
    r = Recorder(tmp_path, async_write=True, chunk_size=chunk_size)
    for step in range(chunk_size):
        r.record(step, x=torch.ones(2))
    assert entered.wait(3)
    original = torch.Tensor.to
    def staging(tensor, *args, **kwargs):
        release.set()
        r._worker.join(5)
        return original(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, 'to', staging)
    with pytest.raises(RuntimeError, match='failure during staging'):
        r.record(chunk_size, x=torch.ones(2))
    assert r.memory_bytes == 0
    with pytest.raises(RuntimeError, match='failure during staging'):
        r.flush()
    with pytest.raises(RuntimeError, match='failure during staging'):
        r.close()
    assert r.memory_bytes == 0


def test_resume_preserves_unrecognized_files(tmp_path):
    Recorder, _, _, _ = api()
    with Recorder(tmp_path) as r:
        r.record(0, x=torch.ones(2))
    unrelated = tmp_path / 'chunk-user-notes.pt'
    unrelated.write_bytes(b'not a managed tensor chunk')
    with Recorder(tmp_path, resume=True):
        pass
    assert unrelated.read_bytes() == b'not a managed tensor chunk'


@pytest.mark.parametrize('filename', [
    '.manifest-user-notes.tmp', '.chunk-user-notes.tmp',
    '.manifest-0123456789abcdef0123456789abcde.tmp',
    '.chunk-0123456789abcdef0123456789abcdef0.tmp',
    '.chunk-0123456789abcdef0123456789abcdeg.tmp',
])
def test_resume_preserves_unrecognized_temporary_files(tmp_path, filename):
    Recorder, _, _, _ = api()
    with Recorder(tmp_path) as r:
        r.record(0, x=torch.ones(2))
    unrelated = tmp_path / filename
    unrelated.write_bytes(b'user file')
    with Recorder(tmp_path, resume=True):
        pass
    assert unrelated.read_bytes() == b'user file'


@pytest.mark.parametrize('prefix', ['chunk', 'manifest'])
def test_resume_cleans_recognized_uuid_staging_files(tmp_path, prefix):
    Recorder, _, _, _ = api()
    with Recorder(tmp_path):
        pass
    orphan = tmp_path / f'.{prefix}-0123456789abcdef0123456789abcdef.tmp'
    orphan.write_bytes(b'incomplete owned staging file')
    with Recorder(tmp_path, resume=True):
        pass
    assert not orphan.exists()
