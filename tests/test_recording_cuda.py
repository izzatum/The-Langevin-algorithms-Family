import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable: staging validation requires a CUDA host')


def test_cuda_stream_staging_owns_snapshot_before_source_overwrite(tmp_path):
    from lafa.recording import Recorder, TraceReader
    stream = torch.cuda.Stream()
    x = torch.zeros(128, device='cuda')
    with Recorder(tmp_path, async_write=True, chunk_size=1, max_buffer_bytes=4096) as r:
        with torch.cuda.stream(stream):
            x.fill_(7)
            r.record(0, x=x)
            # record() uses synchronous CPU staging on the current stream.
            x.fill_(-1)
    torch.testing.assert_close(TraceReader(tmp_path).read('x'), torch.full((1, 128), 7.))
