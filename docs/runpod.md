# Runpod CUDA validation

The local controller prepares a source-only payload, submits **one** Serverless job,
collects real CUDA results, and deletes only its recorded task endpoint/template.
It does not create Pods, build images, install another PyTorch, or send the cloud
API credential to a worker. The current readiness work has **not run a GPU job**.

## Authentication prerequisite

`RUNPOD_API_KEY` is a proxy-backed environment binding. Its allowed destinations
must include **`rest.runpod.io`** for management and **`api.runpod.ai`** for job
submission, health and results. `api.runpod.io` is a different hostname and does
not enable `api.runpod.ai`. Add the latter to the existing binding in secure
cloud environment settings, preserve the existing destinations, and save.
Do not paste a key into chat, source files, templates or shell command arguments.

The latest management-only lifecycle check created and removed an idle task
endpoint/template without submitting a job. The `.ai` health route returned
`401 invalid api key` under the current destination binding. No GPU was allocated.

## Set up an idle endpoint

Choose a persistent workspace directory outside the source checkout. Use a new
ownership receipt for each attempt, but **reuse the same budget ledger**.

```bash
python scripts/runpod_validate.py setup --state /workspace/lafa-runpod/ownership-01.json
python scripts/runpod_validate.py health --state /workspace/lafa-runpod/ownership-01.json
```

`setup` performs management requests only: a private pinned-image template,
20 GB container disk, no persistent volume, and an endpoint with minimum 0 /
maximum 1 worker. It submits no job. Returned settings are checked. Creation
POSTs are never retried; unique names and a durable receipt support reconciliation
if a response is lost. Existing receipts are preserved rather than overwritten.
`health` is read-only. Authentication to `api.runpod.ai`, zero queued/running or
historical work, and verified storage limits are required before a billable run.

## Prepare and review without allocating a GPU

```bash
python scripts/runpod_validate.py prepare \
  --payload /workspace/lafa-runpod/payload.json
```

This prints the payload and worker SHA256 checksums. The archive includes Python
sources under `src`, `tests`, `scripts`, `benchmarks`, package metadata
(`pyproject.toml`, `README.md`, `LICENSE`), and six
notebooks with old outputs removed. It excludes `.git`, environment files,
caches, dependencies and existing figures. Inspect the source and intended run
before approving its printed checksum. Reprepare after source changes.

The controller uses Python's standard library. The worker SDK is pinned to
`runpod==1.12.0`; its installation has a 120-second timeout. The compatible official
image is `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04`, pinned to
Docker Hub digest:

```text
sha256:cb154fcca15d1d6ce858cfa672b76505e30861ef981d28ec94bd44168767d853
```

Setup selects the compatible image; the run step verifies it again and installs
the reviewed worker bootstrap before submission. The worker installs build/test
tooling but does not replace PyTorch.

## Submit only after authentication and payload approval

The following command **submits one billable job**. Replace the checksum with
the exact reviewed `payload_sha256`; neither a missing checksum nor a missing
`--execute` flag can submit a job.

```bash
python scripts/runpod_validate.py run --execute \
  --state /workspace/lafa-runpod/ownership-01.json \
  --payload /workspace/lafa-runpod/payload.json \
  --approved-payload-sha256 REVIEWED_PAYLOAD_SHA256 \
  --ledger /workspace/lafa-runpod/budget.json \
  --output /workspace/lafa-runpod/results
```

Add `--accuracy` to also run a small actual CUDA accuracy experiment: one seed,
300 iterations, 100 warmup steps and 32 chains. The notebook default is its documented
quick budget. Larger experiment budgets need a separately reviewed payload.

The worker allocates a CUDA tensor, executes a kernel and synchronizes before it
records `cuda_executed:true`. It installs notebook/test tooling, installs the
project with `--no-deps --no-build-isolation`, runs pytest, then executes all six
notebooks in clean kernels with `--device cuda`. Optional accuracy also receives
`--device cuda`. PyTorch is never resolved/reinstalled. A missing GPU or incompatible
image fails explicitly.

Logs, pytest JUnit XML, executed notebooks, notebook outcomes, hardware/runtime
metadata, and optional accuracy JSON/figures return in a checked tar archive.
Failed commands retain their actual status and output. The pinned SDK converts
the handler's `error` field to provider `FAILED` while preserving its artifact
fields; the controller downloads those artifacts and exits nonzero. Provider
errors/timeouts without output remain failures and their responses are retained.
Archives reject absolute paths, traversal, links, duplicates and excessive sizes.
Input is below the 10 MB `/run` limit; result transport is capped at 6 MB compressed.
An oversized result fails explicitly and returns a failure metadata summary.

## Time and budget bounds

- Exactly one A4000 GPU; minimum 0 / maximum 1 worker; no network volume.
- Provider execution timeout 900 seconds; per-job absolute TTL 1200 seconds, including
  queueing and retries; idle timeout 5 seconds. No automatic job resubmission.
- Worker deadline 840 seconds leaves packaging time; bootstrap wall timer 25 minutes.
- A detached local controller watchdog attempts cleanup at 30 minutes. Provider
  TTL and scale-to-zero remain necessary because a local watchdog cannot survive
  loss of the cloud machine.
- Advertised Serverless AMPERE_16 flex price is $0.58/hour. The conservative configured
  ceiling is $0.70/hour. The 20 GB ephemeral container disk is included in Serverless
  running cost; the estimate retains an additional small allowance. This is not the
  much cheaper Community Pod rate.
- A 20-minute lifetime plus 10-minute reserve is estimated below $0.36. The persistent
  ledger reserves that estimate **plus $1**, rounded upward (normally $1.36).
  Every attempt reserves at least $1, cannot exceed the default $3 per-run budget,
  and all reservations together cannot exceed the authorized **$10 total**.
  Reservations are not automatically refunded. Keep the same ledger across runs;
  manual invoice reconciliation must account for earlier charges before editing it.

These are conservative estimates using provider duration controls, not a
provider-enforced dollar cap. Startup/shutdown anomalies and price changes still
require monitoring. The controller verifies returned worker limits and GPU choice,
uses no silent POST retries, and finally scales down/deletes its endpoint and
unused private template. Inspect the receipt's `cleanup_confirmed` field.

Run completion removes the recorded endpoint/template. For another attempt, use
`setup` with a new ownership filename and the same cumulative budget ledger.
REST creation was observed to change `workersMax:0` to 3, despite the schema
permitting 0; setup requests and verifies minimum 0 / maximum 1 explicitly.

If an interrupted run leaves task resources behind, use:

```bash
python scripts/runpod_validate.py cleanup \
  --state /workspace/lafa-runpod/ownership-01.json
```

Cleanup checks the recorded task identity and template ownership, preserves
unrelated account resources, and confirms deletion through fresh list requests.
Reads and idempotent deletion are retried within bounded limits. Partial setup
receipts can reconcile exact unique resource names; uncertain creation that is
still unobserved remains explicitly unresolved. Repeat cleanup after the provider
resolves that request; do not submit another job while cleanup is uncertain.

## Local checks

No cloud API calls or GPU allocation are needed:

```bash
python -m pytest tests/test_runpod_tools.py -q
python scripts/runpod_validate.py prepare --payload /tmp/lafa-payload.json
```

See the setup evidence in
`/workspace/lafa-planning/execution/runpod-readiness.md`. Relevant official docs:
[endpoint configuration](https://docs.runpod.io/serverless/endpoints/endpoint-configurations),
[job execution policies](https://docs.runpod.io/serverless/endpoints/send-requests),
[Serverless pricing](https://docs.runpod.io/serverless/pricing),
[Serverless storage](https://docs.runpod.io/serverless/storage/overview), and
[template API](https://docs.runpod.io/api-reference/templates/POST/templates).
