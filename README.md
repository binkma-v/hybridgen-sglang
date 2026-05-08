# HybridGen SGLang

This fork integrates a HybridGen-style KV cache offloading backend into
SGLang. The backend keeps a recent KV window on GPU, offloads older KV to host
memory, selects CPU-resident heavy-hitter tokens with CPU top-k, and computes a
single merged softmax over CPU top-k scores plus the GPU dense segment.

The implementation is focused on decode-time KV cache offloading for MHA
models. It is not a replacement for the full upstream SGLang project; the
original SGLang documentation remains the best reference for installation,
serving APIs, model support, and deployment:

- SGLang docs: https://docs.sglang.io/
- Upstream repository: https://github.com/sgl-project/sglang

## Implemented Work

- `hybrid_kvcache` attention backend for MHA models.
- Host KV storage through SGLang's `MHATokenToKVPoolHost`.
- GPU KV shadow release for request-owned offloaded tokens, with cleanup
  protections in chunk/radix cache paths.
- Adaptive `topk_ratio` / `cpu_k_cap` feedback policy.
- Minimum GPU recent window for short-prompt long-generation workloads:
  `gpu_cap = max(prompt_len * gpu_cache_factor, min_gpu_recent_tokens, 1)`.
- Decode hot-path optimizations:
  - grouped CPU bmm for GQA-aware top-k scoring
  - reusable CPU top-k workspace
  - per-head V gather without `torch.unique`
  - Triton fused merged-softmax attention
  - guarded H2D side-stream copy overlap
  - guarded CPU/GPU partial-attention overlap for larger cap settings
- Decode profiling with per-layer attribution for CPU top-k, host V gather,
  H2D, GPU attention/merge, and eviction copy time.
- Unit tests for release semantics, feedback cap behavior, residency cap,
  workspace reuse, and Triton attention equivalence.

## Environment

The commands below assume the repository root is:

```bash
cd /home/binkma/bm_ds/hybridgen-sglang/sglang
```

Use the prepared Python environment:

```bash
PYTHONPATH=python .venv/bin/python -c "import torch; print(torch.__version__)"
```

The benchmark commands below use this local model snapshot:

```bash
MODEL=/grand/hp-ptycho/binkma/HFmodel/hub/models--Qwen--Qwen2.5-Coder-3B/snapshots/09d9bc5d376b0cfa0100a0694ea7de7232525803
```

## Unit Tests

Run the HybridGen-specific tests:

```bash
PYTHONPATH=python .venv/bin/python -m py_compile \
  python/sglang/srt/server_args.py \
  python/sglang/srt/layers/attention/hybrid_kvcache_backend.py \
  python/sglang/srt/layers/attention/hybridgen_topk.py

PYTHONPATH=python .venv/bin/python test/registered/unit/layers/test_hybridgen_topk.py
PYTHONPATH=python .venv/bin/python test/registered/unit/layers/test_hybridgen_feedback.py
PYTHONPATH=python .venv/bin/python test/registered/unit/layers/test_hybridgen_residency.py
PYTHONPATH=python .venv/bin/python test/registered/unit/mem_cache/test_hybridgen_release.py
```

## Launch Hybrid Backend

Example launch for forced offload with a 512-token minimum GPU recent window:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  HOME=/home/binkma \
  PYTHONPATH=python \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  .venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 \
  --port 31000 \
  --attention-backend hybrid_kvcache \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --max-running-requests 1 \
  --max-total-tokens 12288 \
  --mem-fraction-static 0.70 \
  --hybridgen-gpu-cache-factor 0.1 \
  --hybridgen-min-gpu-recent-tokens 512 \
  --hybridgen-topk-ratio 0.05 \
  --hybridgen-cpu-k-cap 2048 \
  --hybridgen-feedback-interval 4 \
  --hybridgen-host-ratio 2.0 \
  --hybridgen-host-layout layer_first
```

Key HybridGen options:

- `--hybridgen-gpu-cache-factor`: fraction of the prompt-sized window kept on
  GPU.
- `--hybridgen-min-gpu-recent-tokens`: minimum recent KV window kept on GPU;
  default is `512`, and `0` restores the old factor-only behavior.
- `--hybridgen-topk-ratio`: host top-k ratio.
- `--hybridgen-cpu-k-cap`: cap on host K length scanned by CPU top-k.
- `--hybridgen-feedback-interval`: decode steps between feedback updates;
  `0` disables feedback.

## Launch Torch Native Baseline

Use this for pure GPU baseline comparison:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  HOME=/home/binkma \
  PYTHONPATH=python \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  .venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 \
  --port 31000 \
  --attention-backend torch_native \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --max-running-requests 1 \
  --max-total-tokens 12288 \
  --mem-fraction-static 0.70
```

## Benchmark Requests

Run prompt-length and generation-length sweeps against a running server:

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  HOME=/home/binkma \
  PYTHONPATH=python \
  .venv/bin/python - <<'PY'
import time

import requests
from transformers import AutoTokenizer

model = "/grand/hp-ptycho/binkma/HFmodel/hub/models--Qwen--Qwen2.5-Coder-3B/snapshots/09d9bc5d376b0cfa0100a0694ea7de7232525803"
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=False)
token_id = tok.encode(" hello", add_special_tokens=False)[0]
url = "http://127.0.0.1:31000/generate"

params = {
    "max_new_tokens": 1024,
    "temperature": 0,
    "ignore_eos": True,
}

for prompt_len in (128, 512):
    payload = {
        "input_ids": [token_id] * prompt_len,
        "sampling_params": params,
    }
    t0 = time.perf_counter()
    response = requests.post(url, json=payload, timeout=600)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    meta = response.json().get("meta_info", {})
    completion_tokens = meta.get("completion_tokens", 0)
    tok_per_s = completion_tokens / (elapsed_ms / 1000)
    print(
        "RESULT",
        prompt_len,
        "gen=1024",
        f"latency_ms={elapsed_ms:.1f}",
        f"tok_per_s={tok_per_s:.2f}",
        "status=" + str(response.status_code),
        "completion_tokens=" + str(completion_tokens),
    )
PY
```

For a short smoke test, change `max_new_tokens` to `32` and prompt lengths to
`2048, 4096, 8192`.

## Decode Profiling

Profiling is disabled by default. Enable it when you need time attribution:

```bash
SGLANG_HYBRIDGEN_PROFILE=1 \
SGLANG_HYBRIDGEN_PROFILE_INTERVAL=64 \
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  -u all_proxy -u ALL_PROXY \
  HOME=/home/binkma \
  PYTHONPATH=python \
  NO_PROXY=127.0.0.1,localhost \
  no_proxy=127.0.0.1,localhost \
  .venv/bin/python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 127.0.0.1 \
  --port 31000 \
  --attention-backend hybrid_kvcache \
  --disable-radix-cache \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --max-running-requests 1 \
  --max-total-tokens 12288 \
  --mem-fraction-static 0.70 \
  --hybridgen-gpu-cache-factor 0.1 \
  --hybridgen-min-gpu-recent-tokens 512 \
  --hybridgen-topk-ratio 0.05 \
  --hybridgen-cpu-k-cap 2048 \
  --hybridgen-feedback-interval 4 \
  --hybridgen-host-ratio 2.0 \
  --hybridgen-host-layout layer_first
```

The profile log reports per-layer averages for:

- `decode_total/layer`
- `q_to_cpu/layer`
- `cpu_topk/layer`
- `host_v_gather/layer`
- `h2d/layer`
- `gpu_attn_or_merge/layer`
- `gpu_partial/layer`
- `evict_total/batch`, `evict_backup/batch`, `evict_release/batch`
- average `cpu_len`, `effective_cpu_len`, `gpu_len`, and `topk`

Profiling inserts GPU synchronization around measured regions, so use it for
attribution rather than throughput numbers.

## Current Results

Measured on A100-40GB with Qwen2.5-Coder-3B, batch size 1, CUDA graph disabled,
radix cache disabled, and `max_new_tokens=1024`.

| Prompt tokens | Backend | Latency | Throughput |
|---:|---|---:|---:|
| 128 | `torch_native` | 23.16 s | 44.22 tok/s |
| 128 | `hybrid_kvcache`, `min_gpu_recent=512` | 35.40 s | 28.92 tok/s |
| 512 | `torch_native` | 23.03 s | 44.46 tok/s |
| 512 | `hybrid_kvcache`, `min_gpu_recent=512` | 43.19 s | 23.71 tok/s |

The minimum GPU recent window fixes the short-prompt long-generation policy
issue: prompt 128 and prompt 512 both keep a 512-token GPU dense segment after
offload starts, instead of only 12 or 51 tokens with the old factor-only rule.
The remaining gap versus pure GPU is mostly the host path: CPU top-k, host V
gather, H2D transfer, and feedback increasing top-k ratio as the host segment
grows.

## Repository Notes

`INTEGRATION.md` is treated as a local scratch/design note and is ignored by
git. Keep user-facing instructions in this README.
