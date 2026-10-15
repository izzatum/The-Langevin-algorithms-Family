"""Bounded, optional tensor traces independent of sampler implementation.

``async_write`` overlaps disk writes only. Device staging is deliberately synchronous:
record() returns after it owns detached CPU snapshots, including on a CUDA stream.
There are no pinned buffers or asynchronous device transfers to reuse incorrectly.
The byte cap covers tensor payload plus eight bytes per step in active, queued and
in-flight chunks. Python objects, serialization workspace and manifest/index metadata
are not included. Chunk references in the JSON manifest grow with the trace length.
Only one producer / writer may use a run directory at a time.
"""
from collections import deque
import copy
import json
import os
import re
from pathlib import Path
import threading
import uuid

import torch

__all__ = ['Recorder', 'TraceReader', 'RingBuffer', 'OnlineMoments', 'estimate_storage_bytes']


def _field_names(fields):
    if fields is None:
        return None
    names = tuple(fields)
    if not names or len(set(names)) != len(names) or any(not isinstance(k, str) or not k for k in names):
        raise ValueError('fields must be distinct nonempty names')
    return names


def _dtype_name(dtype):
    return str(dtype).removeprefix('torch.')


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def estimate_storage_bytes(num_records, fields, *, storage_dtype=None):
    """Estimate raw tensor bytes plus int64 steps, excluding file format overhead."""
    if num_records < 0:
        raise ValueError('num_records must be nonnegative')
    return num_records * (8 + sum(t.numel() * (torch.empty((), dtype=storage_dtype).element_size()
                                if storage_dtype is not None else t.element_size()) for t in fields.values()))


class Recorder:
    """Write detached owned snapshots with bounded payload buffering and backpressure.

    Steps must increase strictly and steps divisible by ``every`` are selected. Fields
    are tensor labels supplied by the caller; ``gradient`` has no inferred semantics.
    Resume starts strictly after the manifest's last committed step. To resume a sampler,
    its checkpoint iteration must equal that step; automatic trace truncation is absent.
    A successful ``flush`` is the persistence boundary. Files are atomically replaced,
    but no guarantee is made for sudden power loss (no fsync protocol).
    """
    def __init__(self, path, *, fields=None, every=1, chunk_size=64,
                 max_buffer_bytes=64 * 1024 * 1024, async_write=False,
                 storage_dtype=None, resume=False):
        self.fields = _field_names(fields)
        self.every = _positive_integer(every, 'every')
        self.chunk_size = _positive_integer(chunk_size, 'chunk_size')
        self.max_buffer_bytes = _positive_integer(max_buffer_bytes, 'max_buffer_bytes')
        if storage_dtype is not None and not isinstance(storage_dtype, torch.dtype):
            raise ValueError('storage_dtype must be a torch.dtype')
        self.storage_dtype = storage_dtype
        self.async_write = bool(async_write)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        manifest_path = self.path / 'manifest.json'
        if resume:
            if not manifest_path.exists():
                raise FileNotFoundError('resume requires a committed manifest')
            self._manifest = _load_manifest(self.path)
            if self._manifest['every'] != self.every:
                raise ValueError('resume every must match the existing trace')
            if self._manifest['storage_dtype'] != (_dtype_name(storage_dtype) if storage_dtype is not None else None):
                raise ValueError('resume storage_dtype must match the existing trace')
            saved_names = tuple(self._manifest['fields'])
            if saved_names:
                if self.fields is not None and set(self.fields) != set(saved_names):
                    raise ValueError('resume fields must match the existing trace')
                self.fields = saved_names
            committed = {c['file'] for c in self._manifest['chunks']}
            for pattern in ('chunk-*.pt', '.chunk-*.tmp', '.manifest-*.tmp'):
                for candidate in self.path.glob(pattern):
                    managed = (re.fullmatch(r'chunk-[0-9]{8,}\.pt', candidate.name) is not None
                               or re.fullmatch(r'\.(?:chunk|manifest)-[0-9a-f]{32}\.tmp', candidate.name) is not None)
                    if managed and candidate.name not in committed and candidate.is_file():
                        candidate.unlink()
        else:
            if any(self.path.iterdir()):
                raise FileExistsError('run directory is not empty; use resume=True explicitly')
            self._manifest = dict(schema_version=1, fields={}, every=self.every,
                                  storage_dtype=_dtype_name(storage_dtype) if storage_dtype is not None else None,
                                  chunks=[], count=0, last_step=None)
            self._save_manifest(self._manifest)
        self._schema = copy.deepcopy(self._manifest['fields'])
        self._last_step = self._manifest['last_step']
        self._buffer = []
        self._buffer_bytes = 0
        self._memory_bytes = 0
        self._peak = 0
        self._queue = deque()
        self._busy = False
        self._error = None
        self._closed = False
        self._stop = False
        self._condition = threading.Condition()
        self._worker = None
        if self.async_write:
            self._worker = threading.Thread(target=self._work, name='lafa-trace-writer', daemon=True)
            self._worker.start()

    @property
    def memory_bytes(self):
        with self._condition:
            return self._memory_bytes

    @property
    def stats(self):
        with self._condition:
            return {'buffer_bytes': self._memory_bytes, 'peak_buffer_bytes': self._peak,
                    'max_buffer_bytes': self.max_buffer_bytes,
                    'committed_records': self._manifest['count']}

    @property
    def last_committed_step(self):
        return self._manifest['last_step']

    def _check(self):
        if self._error is not None:
            raise RuntimeError(f'recording writer failed: {self._error}')
        if self._closed:
            raise RuntimeError('recorder is closed')

    def record(self, step, **fields):
        self._check()
        if isinstance(step, bool) or not isinstance(step, int) or step < 0 or step > torch.iinfo(torch.int64).max:
            raise ValueError('step must be a nonnegative int64 integer')
        if step % self.every:
            return
        if self._last_step is not None and step <= self._last_step:
            raise ValueError('recorded steps must increase strictly beyond the last committed/reserved step')
        names = self.fields if self.fields is not None else tuple(fields)
        if not names:
            raise ValueError('at least one tensor field is required')
        if self.fields is None and self._schema and set(fields) != set(self._schema):
            raise ValueError('field names must remain consistent')
        if self.fields is not None and any(k not in fields for k in names):
            raise ValueError('missing selected field')
        selected = {k: fields[k] for k in names}
        schema = {}
        for key, value in selected.items():
            if not isinstance(value, torch.Tensor) or value.layout != torch.strided or value.device.type == 'meta':
                raise ValueError('recording requires dense, materialized tensors')
            dtype = self.storage_dtype if self.storage_dtype is not None else value.dtype
            schema[key] = dict(shape=list(value.shape), dtype=_dtype_name(dtype), source_dtype=_dtype_name(value.dtype))
        if self._schema and schema != self._schema:
            raise ValueError('field names, shapes and dtypes must remain consistent')
        size = estimate_storage_bytes(1, selected, storage_dtype=self.storage_dtype)
        if size > self.max_buffer_bytes:
            raise ValueError('single record exceeds max_buffer_bytes before copying')
        if self._buffer and self.memory_bytes + size > self.max_buffer_bytes:
            self._submit()
        with self._condition:
            while self._memory_bytes + size > self.max_buffer_bytes:
                self._check()
                self._condition.wait()
            self._check()
            self._memory_bytes += size  # reserve before copying / transferring
            self._peak = max(self._peak, self._memory_bytes)
        owned = None
        try:
            owned = {k: t.detach().to(device='cpu', dtype=self.storage_dtype or t.dtype, copy=True)
                     for k, t in selected.items()}
            with self._condition:
                # A worker can fail while staging. Accept only while its status is
                # healthy, even when this record leaves a partially filled chunk.
                self._check()
                self._schema = schema
                self._last_step = step
                self._buffer.append((step, owned))
                self._buffer_bytes += size
        except BaseException:
            # Staging has not transferred ownership to the buffer on failure.
            del owned
            with self._condition:
                self._memory_bytes -= size
                self._condition.notify_all()
            raise
        if len(self._buffer) >= self.chunk_size:
            self._submit()

    def _submit(self):
        if not self._buffer:
            return
        if self.async_write:
            with self._condition:
                # Keep the producer buffer owned until submission is guaranteed.
                self._check()
                self._queue.append((self._buffer, self._buffer_bytes))
                self._buffer, self._buffer_bytes = [], 0
                self._condition.notify_all()
        else:
            records, size = self._buffer, self._buffer_bytes
            self._buffer, self._buffer_bytes = [], 0
            try:
                self._write_chunk(records)
            except BaseException as exc:
                self._error = str(exc)
                raise
            finally:
                with self._condition:
                    self._memory_bytes -= size
                    self._condition.notify_all()

    def _work(self):
        while True:
            with self._condition:
                while not self._queue and not self._stop:
                    self._condition.wait()
                if self._stop and not self._queue:
                    return
                records, size = self._queue.popleft()
                self._busy = True
            try:
                self._write_chunk(records)
            except BaseException as exc:
                with self._condition:
                    self._error = str(exc)
                    while self._queue:
                        _, queued_size = self._queue.popleft()
                        self._memory_bytes -= queued_size
                    self._stop = True
            finally:
                # Drop owned tensors before releasing their byte reservation.
                del records
                with self._condition:
                    self._memory_bytes -= size
                    self._busy = False
                    self._condition.notify_all()
            if self._error is not None:
                return

    def _save_manifest(self, manifest):
        temporary = self.path / f'.manifest-{uuid.uuid4().hex}.tmp'
        try:
            temporary.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            os.replace(temporary, self.path / 'manifest.json')
        finally:
            temporary.unlink(missing_ok=True)

    def _write_chunk(self, records):
        index = len(self._manifest['chunks'])
        filename = f'chunk-{index:08d}.pt'
        target = self.path / filename
        if target.exists():
            raise FileExistsError(f'uncommitted chunk already exists: {filename}; reopen with resume=True')
        temporary = self.path / f'.chunk-{uuid.uuid4().hex}.tmp'
        payload = {'steps': torch.tensor([r[0] for r in records], dtype=torch.int64),
                   'fields': {k: [r[1][k] for r in records] for k in records[0][1]}}
        try:
            torch.save(payload, temporary)
            os.replace(temporary, target)
            manifest = dict(self._manifest)
            manifest['fields'] = self._schema
            manifest['chunks'] = self._manifest['chunks'] + [dict(file=filename, count=len(records),
                first_step=records[0][0], last_step=records[-1][0], start=self._manifest['count'])]
            manifest['count'] += len(records)
            manifest['last_step'] = records[-1][0]
            self._save_manifest(manifest)
            self._manifest = manifest
        finally:
            temporary.unlink(missing_ok=True)

    def flush(self):
        self._check()
        self._submit()
        with self._condition:
            while self._queue or self._busy:
                self._condition.wait()
            self._check()

    def close(self):
        if self._closed:
            if self._error is not None:
                raise RuntimeError(f'recording writer failed: {self._error}')
            return
        try:
            self.flush()
        finally:
            with self._condition:
                self._stop = True
                self._memory_bytes -= self._buffer_bytes
                self._buffer.clear()
                self._buffer_bytes = 0
                self._condition.notify_all()
            if self._worker is not None:
                self._worker.join()
            self._closed = True

    def __enter__(self):
        self._check()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def _load_manifest(path):
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('unsupported trace schema_version')
    for chunk in manifest['chunks']:
        filename = chunk['file']
        if Path(filename).name != filename or not filename.startswith('chunk-') or not filename.endswith('.pt'):
            raise ValueError('invalid chunk reference')
        if not (path / filename).is_file():
            raise FileNotFoundError(f'committed chunk is missing: {filename}')
    return manifest


class TraceReader:
    """Iterate one bounded chunk at a time as {'steps': tensor, 'fields': dict}.

    ``read`` uses recorded-row indices (Python slicing, including negative indices),
    loads only intersecting chunks, and allocates the entire selected output. For a
    full trace use iteration when full-output allocation is undesirable. Iteration's
    peak tensor allocation is at most twice one stored chunk while stacking records.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = _load_manifest(self.path)

    def _load(self, chunk):
        return torch.load(self.path / chunk['file'], weights_only=True, map_location='cpu')

    def __iter__(self):
        for chunk in self.manifest['chunks']:
            payload = self._load(chunk)
            yield {'steps': payload['steps'], 'fields': {k: torch.stack(v) for k, v in payload['fields'].items()}}

    def read(self, field, start=None, stop=None):
        if field not in self.manifest['fields']:
            raise KeyError(field)
        spec = self.manifest['fields'][field]
        first, last, _ = slice(start, stop).indices(self.manifest['count'])
        count = max(0, last - first)
        output = torch.empty((count, *spec['shape']), dtype=getattr(torch, spec['dtype']))
        for chunk in self.manifest['chunks']:
            begin, end = chunk['start'], chunk['start'] + chunk['count']
            lo, hi = max(first, begin), min(last, end)
            if lo >= hi:
                continue
            payload = self._load(chunk)
            for row in range(lo, hi):
                output[row - first].copy_(payload['fields'][field][row - begin])
        return output


class RingBuffer:
    """Retain ``capacity`` owned detached CPU snapshots with a fixed tensor schema."""
    def __init__(self, capacity, fields):
        self.capacity = _positive_integer(capacity, 'capacity')
        self.fields = _field_names(fields)
        if self.fields is None:
            raise ValueError('RingBuffer requires explicit fields')
        self._records = deque(maxlen=self.capacity)
        self._schema = None

    def record(self, step, **fields):
        selected = {k: fields[k] for k in self.fields}
        schema = {k: (t.shape, t.dtype) for k, t in selected.items()}
        if self._schema is not None and schema != self._schema:
            raise ValueError('ring field shapes and dtypes must remain consistent')
        owned = {k: t.detach().to(device='cpu', copy=True) for k, t in selected.items()}
        self._schema = schema
        self._records.append({'step': step, 'fields': owned})

    @property
    def memory_bytes(self):
        return sum(estimate_storage_bytes(1, r['fields']) for r in self._records)

    def __iter__(self):
        return iter(self._records)

    def __len__(self):
        return len(self._records)


class OnlineMoments:
    """Welford elementwise mean and sample variance; one update is one vector draw.

    State uses CPU float64, O(number of elements) storage, and no covariance matrix.
    """
    def __init__(self):
        self.count = 0
        self._mean = None
        self._m2 = None

    def update(self, value):
        if not isinstance(value, torch.Tensor) or value.is_complex():
            raise ValueError('moments require a real tensor')
        x = value.detach().to(device='cpu', dtype=torch.float64)
        if self._mean is None:
            self._mean, self._m2 = torch.zeros_like(x), torch.zeros_like(x)
        if x.shape != self._mean.shape:
            raise ValueError('moment shape must remain consistent')
        self.count += 1
        delta = x - self._mean
        self._mean.add_(delta / self.count)
        self._m2.add_(delta * (x - self._mean))

    @property
    def mean(self):
        return None if self._mean is None else self._mean.clone()

    @property
    def variance(self):
        if self._m2 is None:
            return None
        return self._m2 / (self.count - 1) if self.count > 1 else torch.full_like(self._m2, float('nan'))

    def state_dict(self):
        return {'count': self.count, 'mean': None if self._mean is None else self._mean.clone(),
                'm2': None if self._m2 is None else self._m2.clone()}

    def load_state_dict(self, state):
        count = state['count']
        mean, m2 = state['mean'], state['m2']
        if not isinstance(count, int) or count < 0 or (count == 0) != (mean is None and m2 is None):
            raise ValueError('invalid moment state')
        if count and (not isinstance(mean, torch.Tensor) or not isinstance(m2, torch.Tensor) or mean.shape != m2.shape):
            raise ValueError('invalid moment tensors')
        self.count = count
        self._mean = None if mean is None else mean.detach().to(device='cpu', dtype=torch.float64, copy=True)
        self._m2 = None if m2 is None else m2.detach().to(device='cpu', dtype=torch.float64, copy=True)
