# llms — local LLM serving stack

Serve any GGUF model on your own hardware with one command, plus tooling to run
local-LLM sub-agents under Claude Code. Chosen over Ollama/LM Studio for
fine-grained control of GPU/CPU memory placement (MoE expert offload).

Works on **NVIDIA (CUDA)**, **AMD (ROCm or Vulkan)**, and **macOS (Metal or MLX)** —
the backend is autodetected.

## Layout

| path | what |
|---|---|
| `run-llm.sh` | launch `llama-server` for any GGUF — backend autodetect, smart MoE offload, OOM auto-retry |
| `gguf-estimate.py` | reads a GGUF header and estimates `--n-cpu-moe` for the free VRAM + context (SWA-aware) |
| `localagent/` | run headless `claude` sub-agents on the local llama-server, validated by Opus — see [its README](localagent/README.md) |
| `llama.cpp/` | upstream clone + build (not tracked; `./run-llm.sh build` creates it) |

## Quick start

```bash
./run-llm.sh deps     # install build requirements (pacman/apt/dnf/zypper/brew)
./run-llm.sh build    # clone + cmake-build llama.cpp for your backend
./run-llm.sh          # pick a local model interactively, serve on :8089
```

Then point any OpenAI-compatible client at `http://localhost:8089/v1`.

## run-llm.sh

```bash
./run-llm.sh -m unsloth/gpt-oss-20b-GGUF:Q8_0   # any HF repo:quant (cached)
./run-llm.sh -m qwen3-coder -c 128k             # fuzzy-match a local GGUF
./run-llm.sh -m <model> --draft unsloth/Qwen3-0.6B-GGUF:Q8_0   # spec decoding
./run-llm.sh --backend mlx -m mlx-community/Qwen3-8B-4bit      # MLX on macOS
./run-llm.sh list | stop | status
```

- **Backends:** `cuda` / `rocm` / `vulkan` / `metal` / `mlx` / `cpu`, autodetected;
  override with `--backend` or `BACKEND=`.
- **Model resolution:** exact path → substring match against local models
  (LM Studio dirs + HF/llama.cpp caches) → HF download. Never re-downloads
  what you already have.
- **Smart offload:** free VRAM is measured (`nvidia-smi` / `rocm-smi`), then
  `gguf-estimate.py` sizes `--n-cpu-moe` so weights + KV cache + compute
  buffers fit. On OOM the launcher retries with more CPU offload automatically.
  Unified-memory backends (Metal) skip offload and use mmap.
- **Defaults:** 64k context, q4_0 KV cache, flash attention, threads = physical
  cores (never SMT — it collapses generation speed).
- Everything is a flag or env var: `MODEL CONTEXT NCMOE THREADS KVTYPE DRAFT
  HOST PORT BACKEND`; extra args after `--` go straight to `llama-server`.

## localagent

CLI + Anthropic-API router proxy to run `claude` sub-agents whose brain is the
local llama-server (zero Anthropic tokens), isolated in git worktrees so the
orchestrating Claude validates results before they land. Install the CLIs:

```bash
localagent/install.sh        # symlinks localagent + claude-local into ~/.local/bin
```

## License

MIT — see [LICENSE](LICENSE). This project builds on
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT, The ggml authors) and
optionally [mlx-lm](https://github.com/ml-explore/mlx-lm) (MIT, Apple Inc.);
see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
