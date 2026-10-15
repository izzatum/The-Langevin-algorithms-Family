#!/usr/bin/env python3
"""Execute tutorial notebooks in clean kernels using this Python environment."""
import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--output', type=Path, help='Output directory; default: update notebooks in place')
    parser.add_argument('--full', action='store_true', help='Run longer comparison budgets')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout per code cell in seconds')
    parser.add_argument('notebooks', nargs='*', type=Path)
    args = parser.parse_args()
    import torch
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but torch.cuda.is_available() is false')
    import nbformat
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager
    root = Path(__file__).resolve().parents[1]
    paths = args.notebooks or sorted((root / 'notebooks').glob('*.ipynb'))
    if not paths:
        parser.error('No notebooks found')
    output = args.output.resolve() if args.output else root / 'notebooks'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix='lafa-notebooks-') as temporary:
        temporary = Path(temporary)
        spec = temporary / 'kernels' / 'lafa-run'
        spec.mkdir(parents=True)
        (spec / 'kernel.json').write_text(json.dumps(dict(
            argv=[sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            display_name='LAFA execution', language='python')))
        os.environ.update(LAFA_DEVICE=args.device, LAFA_QUICK='0' if args.full else '1',
                          JUPYTER_RUNTIME_DIR=str(temporary / 'runtime'),
                          IPYTHONDIR=str(temporary / 'ipython'),
                          MPLCONFIGDIR=str(temporary / 'matplotlib'),
                          XDG_CACHE_HOME=str(temporary / 'cache'),
                          PYTHONPATH=str(root / 'src') + os.pathsep + os.environ.get('PYTHONPATH', ''))
        for path in paths:
            path = path.resolve()
            notebook = nbformat.read(path, as_version=4)
            for cell in notebook.cells:
                if cell.cell_type == 'code':
                    cell.outputs = []
                    cell.execution_count = None
            manager = KernelManager(kernel_name='lafa-run', transport='ipc',
                                    kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(temporary / 'kernels')]))
            client = NotebookClient(notebook, km=manager, timeout=args.timeout,
                                    resources={'metadata': {'path': str(root)}}, allow_errors=False)
            start = time.perf_counter()
            result = dict(notebook=path.name, device=args.device, status='passed')
            print(f'Executing {path.name} on {args.device}', flush=True)
            try:
                client.execute()
            except Exception as error:
                result.update(status='failed', exception=type(error).__name__, detail=str(error))
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
                result['seconds'] = time.perf_counter() - start
                nbformat.write(notebook, output / path.name)
                results.append(result)
                (output / 'execution-results.json').write_text(json.dumps(results, indent=2))
            print(json.dumps(result), flush=True)
    return int(any(result['status'] != 'passed' for result in results))


if __name__ == '__main__':
    sys.exit(main())
