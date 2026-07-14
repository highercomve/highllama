# highllama — local LLM serving stack

Serve any GGUF model on your own hardware with one command, plus tooling to run
local-LLM sub-agents under Claude Code. Chosen over Ollama/LM Studio for
fine-grained control of GPU/CPU memory placement (MoE expert offload).

Works on **NVIDIA (CUDA)**, **AMD (ROCm or Vulkan)**, and **macOS (Metal or MLX)** —
the backend is autodetected.

## Layout

| path | what |
|---|---|
| `highllama` | launch `llama-server` for any GGUF — backend autodetect, smart MoE offload, OOM auto-retry |
| `install.sh` | symlink `highllama` into `~/.local/bin` (`--systemd` adds a user service) |
| `gguf-estimate.py` | reads a GGUF header and estimates `--n-cpu-moe` for the free VRAM + context (SWA-aware) |
| `localagent/` | run headless `claude` sub-agents on the local llama-server, validated by Opus — see [its README](localagent/README.md) |
| `code_benchmark/` | private code-quality benchmark for highllama models — see [its README](code_benchmark/README.md) |
| `speed_benchmark/` | token-generation speed benchmark — see [its README](speed_benchmark/README.md) |
| `llama.cpp/` | upstream clone + build (not tracked; `highllama build` / `update` creates/updates it) |

## Quick start

```bash
./install.sh          # symlink highllama into ~/.local/bin
highllama deps        # install build requirements (pacman/apt/dnf/zypper/brew)
highllama build       # clone + cmake-build llama.cpp for your backend
highllama update      # git pull llama.cpp + rebuild for your backend
highllama             # pick a local model interactively, serve on :8089
```

Then point any OpenAI-compatible client at `http://localhost:8089/v1`.

## Run as a service (Linux)

`./install.sh --systemd` installs an optional **systemd user unit**.
Configure it in `~/.config/highllama/highllama.env` — `MODEL=` is required, and
any highllama env var (`CONTEXT`, `PORT`, `KVTYPE`, ...) works there too:

```bash
systemctl --user start highllama       # launch the server
systemctl --user enable highllama      # start on login
loginctl enable-linger $USER           # ...or even without logging in
journalctl --user -u highllama -f      # follow the server logs
highllama status | stop                # stop detects the unit and uses systemd
```

## highllama

```bash
highllama pull                               # browse/search HF, pick + download
highllama pull qwen3 coder                   # search HF by term
highllama pull unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF:Q5_K_XL   # direct
highllama -m unsloth/gpt-oss-20b-GGUF:Q8_0   # any HF repo:quant (cached)
highllama -m qwen3-coder -c 128k             # fuzzy-match a local GGUF
highllama -m <model> --draft unsloth/Qwen3-0.6B-GGUF:Q8_0   # spec decoding
highllama -m unsloth/gemma-4-12B-it-GGUF:Q4_K_XL --mtp       # multi-token prediction
highllama -m unsloth/gemma-4-12B-it-GGUF:Q4_K_XL -c 160k    # head_dim=512: symmetric q8_0 KV, stays on-GPU
highllama -m <model> --kv q4_0 --dry 0                      # leaner symmetric KV; disable the DRY sampler
highllama -m <model> --chat-template ./fixed.jinja          # override a broken GGUF template
highllama --backend mlx -m mlx-community/Qwen3-8B-4bit      # MLX on macOS
highllama ls                                 # list local models (picker view)
highllama list | stop | status | logs        # full paths | kill | status | view logs
highllama update                             # git pull llama.cpp + rebuild for the backend
```

- **Downloads:** `pull` with no args asks for a search term and lists matching
  HF repos (most downloaded first), then the repo's quants with sizes — pick
  one and it downloads (resumable, multi-part shards handled) into
  `~/.lmstudio/models/<publisher>/<repo>/` — so LM Studio sees it too — or
  `~/.cache/llama.cpp` if LM Studio isn't installed; `list`/`-m` find both.
  Set `HF_TOKEN` for gated models, `PULL_DIR` to download elsewhere.
- **Build / Update:** `deps` installs package requirements; `build` clones
  and compiles `llama.cpp` for the active backend; `update` pulls the latest
  upstream `llama.cpp` changes and rebuilds.

- **Backends:** `cuda` / `rocm` / `vulkan` / `metal` / `mlx` / `cpu`, autodetected;
  override with `--backend` or `BACKEND=`.
- **Model resolution:** exact path → substring match against local models
  (LM Studio dirs + HF/llama.cpp caches + Unsloth Studio cache/exports) →
  HF download. Never re-downloads what you already have.
- **Smart offload:** free VRAM is measured (`nvidia-smi` / `rocm-smi`), then
  `gguf-estimate.py` sizes `--n-cpu-moe` so weights + KV cache + compute
  buffers fit. On OOM the launcher retries with more CPU offload automatically.
  Unified-memory backends (Metal) skip offload and use mmap.
- **Defaults:** 64k context, symmetric `q8_0` KV cache, flash attention (auto),
  DRY anti-loop sampler (0.8), threads = physical cores (never SMT — it collapses
  generation speed).
- **Flash-attn & KV precision:** `--fa auto|on|off` (default `auto`, FA on). The KV
  cache defaults to **symmetric `q8_0`** because the CUDA flash-attn kernels are
  only built for a matching k/v type (`f16/f16`, `q4_0/q4_0`, `q8_0/q8_0`, …)
  unless `GGML_CUDA_FA_ALL_QUANTS` is set — a *mismatched* cache returns no GPU
  kernel and attention falls back to CPU, collapsing prompt processing ~14× (GPU
  sits idle). Symmetric `q8_0` is near-lossless, half the VRAM of `f16`, and runs
  on-GPU even for large `head_dim` (Gemma 4 = 512, via the f16 mma kernel that
  dequantizes on read) — so more layers/experts stay resident at long context.
  Override with `--kv q4_0` (leaner) or `--kv f16` (max precision); `--fa off`
  uses the non-FA path and forces `f16`. Keep any per-side `--kv-k`/`--kv-v`
  symmetric.
- **Anti-loop insurance (DRY):** `--dry <mult>` / `DRY=` (default 0.8; `0`
  disables) sets a server-side DRY sampler that penalizes repeated token
  *sequences* — the backstop against degenerate `<think></think>` / repeat loops.
  It's a default any client request can override, and DRY (not `repeat-penalty`)
  so legitimate code repetition isn't punished.
- **MTP (multi-token prediction):** `--mtp` enables llama.cpp's `draft-mtp`
  speculative decoding (~1.4–2.2× faster) for models that ship an MTP head
  (e.g. Gemma 4). With `-hf <repo>` the bundled `mtp-` head is auto-fetched; with
  a local file, drop the `mtp-*.gguf` next to the model and it's picked up. Tune
  drafted tokens with `--mtp-nmax N` / `MTP_NMAX=` (default 2; try 1–6). MTP
  reserves ~2 GB extra VRAM, which the offload estimate accounts for. Mutually
  exclusive with `--draft`.
- **MTP + embeddings at once:** speculative decoding (causal) and embeddings
  (pooled) can't share one model context, so when `--mtp`/`--draft` is active and
  embeddings are on, highllama switches to llama.cpp **router mode** — the chat
  model (with MTP) and a small embedding model run as separate child processes
  behind one endpoint (`/v1/chat/completions` → chat, `/v1/embeddings` → embed).
  The embedding model is auto-discovered (embeddinggemma / bge / nomic / e5 / …)
  or set explicitly with `EMBED_MODEL=`; `--no-embeddings` opts out. Needs a
  local-file chat model.
- **Chat template fixes:** some GGUFs ship a Jinja template llama.cpp's minja
  engine can't render — Gemma 4's tool-use template uses `map('upper')`, so any
  request carrying tools 500s with `NotImplemented: map: filter-mapping`, which
  breaks agent clients (opencode, etc.). highllama auto-applies a bundled fix
  (`templates/gemma-4-fixed.jinja`) for Gemma 4; override any model's template
  with `--chat-template <file>` / `TEMPLATE=`, or disable the auto-fix with
  `IGNORE_TEMPLATE=1`.
- Everything is a flag or env var: `MODEL CONTEXT NCMOE THREADS KVTYPE KVTYPE_K
  KVTYPE_V FA DRY DRAFT MTP MTP_NMAX TEMPLATE EMBED_MODEL HOST PORT BACKEND`;
  extra args after `--` go straight to `llama-server`.

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
