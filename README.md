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
highllama -m <mtp-model> --mtp 2             # multi-token prediction (self-draft)
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
  `gguf-estimate.py` sizes the offload so weights + KV cache + compute buffers
  fit. MoE models peel experts to the CPU (`--n-cpu-moe`); **dense models**
  (Gemma, Llama, etc.) keep whole layers on the GPU (`-ngl`) instead, since
  they have no experts to offload. On OOM the launcher retries with more CPU
  offload automatically. Unified-memory backends (Metal) skip offload and use mmap.
- **MTP (multi-token prediction):** `--mtp <n>` (or `MTP=<n>`) turns on
  self-speculative decoding using the model's own MTP head baked into the GGUF —
  no second draft model, <10% extra VRAM, ~1.5-2× faster generation. Needs a
  recent `llama.cpp` (`highllama update`) and an MTP-converted GGUF (e.g. Unsloth
  `*-MTP-*` repos). Maps to `--spec-type draft-mtp --spec-draft-n-max <n>
  --spec-draft-p-min 0.75`; forces a single slot (`PARALLEL=1`) and is mutually
  exclusive with `--draft`.
- **Defaults:** 64k context, q4_0 KV cache, flash attention, threads = physical
  cores (never SMT — it collapses generation speed).
- Everything is a flag or env var: `MODEL CONTEXT NCMOE THREADS KVTYPE DRAFT
  MTP HOST PORT BACKEND`; extra args after `--` go straight to `llama-server`.

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
