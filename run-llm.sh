#!/usr/bin/env bash
#
# run-llm.sh — launch ANY GGUF model on llama.cpp server (NVIDIA / AMD / Apple)
#
# Backends (autodetected, override with --backend or BACKEND=):
#   cuda    NVIDIA (nvidia-smi present)
#   rocm    AMD via HIP/ROCm (rocm-smi / amd-smi present)
#   vulkan  AMD/Intel/anything with a Vulkan driver
#   metal   macOS / Apple silicon (default on Darwin; unified memory)
#   mlx     macOS MLX runtime (mlx_lm.server, not llama.cpp; -m = HF repo)
#   cpu     no GPU found
#
# One-shot setup on any platform:
#   ./run-llm.sh deps [backend]    # install build requirements (pacman/apt/dnf/zypper/brew)
#   ./run-llm.sh build [backend]   # clone (if needed) + cmake-build llama.cpp for the backend
#
# Manual cmake equivalents (inside llama.cpp/):
#   NVIDIA: cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
#   AMD:    cmake -B build -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1100   # your gfx
#       or: cmake -B build -DGGML_VULKAN=ON                         # easier
#   macOS:  cmake -B build                                          # Metal default
#   MLX:    pip install mlx-lm                                      # no build
#
# Usage:
#   ./run-llm.sh                                   # interactive picker, 64k ctx
#   ./run-llm.sh -m unsloth/gpt-oss-20b-GGUF:Q8_0  # any HF repo:quant
#   ./run-llm.sh -m /path/to/model.gguf -c 32k     # any local GGUF
#   ./run-llm.sh -m <model> -c 128k --ncmoe 44     # 128k + manual offload
#   ./run-llm.sh --backend mlx -m mlx-community/Qwen3-8B-4bit
#   ./run-llm.sh -m <model> --draft unsloth/Qwen3-0.6B-GGUF:Q8_0   # spec-decoding
#
# Everything is overridable by flag OR env var (flag wins):
#   MODEL CONTEXT NCMOE THREADS KVTYPE DRAFT HOST PORT BACKEND
#
set -euo pipefail

# ---- defaults -------------------------------------------------------------
LLAMA_DIR="${LLAMA_DIR:-$HOME/Code/llms/llama.cpp}"
MODEL="${MODEL:-}"          # empty => interactive picker (no-arg dynamic mode)
CONTEXT="${CONTEXT:-64k}"   # accepts 32k / 64k / 128k or a raw token count
NCMOE="${NCMOE:-}"          # MoE layers pinned to CPU; empty = auto by free VRAM
THREADS="${THREADS:-}"      # empty = physical cores. NEVER SMT count (gen collapses)
KVTYPE="${KVTYPE:-q4_0}"    # KV cache quant: q4_0 (lean) | q8_0 (higher quality) | f16
PARALLEL="${PARALLEL:-1}"   # server slots. 1 = single-user (least VRAM). >1 multiplies buffers
DRAFT="${DRAFT:-}"          # optional draft model (HF tag or path) for speculative decoding
HOST="${HOST:-0.0.0.0}"   # 0.0.0.0 = reachable from other machines on the LAN
PORT="${PORT:-8089}"        # 8080 is a very common conflict; auto-advances if busy
BACKEND="${BACKEND:-}"      # empty = autodetect (see detect_backend)
EXTRA=()                    # any extra args after `--` are passed straight to llama-server

# Where local GGUFs live: LM Studio models + llama.cpp/HF download caches
MODEL_ROOTS=("$HOME/.lmstudio/models" "$HOME/.cache/huggingface/hub" \
             "$HOME/.cache/llama.cpp" "$HOME/Library/Caches/llama.cpp")

# ---- portability helpers (GNU/Linux + BSD/macOS) ---------------------------
OS="$(uname -s)"

file_size() {  # bytes, following symlinks
  stat -Lc%s "$1" 2>/dev/null || stat -Lf%z "$1" 2>/dev/null
}

port_busy() {
  if command -v ss >/dev/null 2>&1; then
    ss -ltn 2>/dev/null | command grep -q ":$1 "
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1
  else
    netstat -an 2>/dev/null | command grep -q "[.:]$1 .*LISTEN"
  fi
}

physical_cores() {
  if [ "$OS" = "Darwin" ]; then
    sysctl -n hw.physicalcpu 2>/dev/null || echo 8
  else
    local n
    n="$(lscpu -b -p=Core,Socket 2>/dev/null | command grep -v '^#' | sort -u | wc -l)"
    [ "${n:-0}" -gt 0 ] && echo "$n" || echo "$(( $(nproc 2>/dev/null || echo 16) / 2 ))"
  fi
}

detect_backend() {
  if [ "$OS" = "Darwin" ]; then echo metal; return; fi
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then echo cuda; return; fi
  if command -v rocm-smi >/dev/null 2>&1 || command -v amd-smi >/dev/null 2>&1; then echo rocm; return; fi
  if command -v vulkaninfo >/dev/null 2>&1; then echo vulkan; return; fi
  echo cpu
}

# free VRAM in MiB; prints nothing if unknown (caller falls back to buckets)
free_vram_mib() {
  case "$1" in
    cuda)
      nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -dc '0-9' ;;
    rocm)
      # rocm-smi CSV: device,VRAM Total Memory (B),VRAM Total Used Memory (B)
      rocm-smi --showmeminfo vram --csv 2>/dev/null | awk -F, '
        NR==2 { printf "%d\n", ($2-$3)/1048576 }' ;;
    *) : ;;   # vulkan/metal/cpu: unknown / unified memory
  esac
}

gpu_status() {
  case "$1" in
    cuda) nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader 2>/dev/null || true ;;
    rocm) rocm-smi --showmeminfo vram 2>/dev/null | command grep -i vram || true ;;
    metal) echo "unified memory: $(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1048576 )) MiB total" ;;
    *) echo "(no VRAM telemetry for backend '$1')" ;;
  esac
}

# list every local .gguf (excluding multimodal projectors) with size + path
list_local_models() {
  local d
  printf '%-10s  %s\n' "SIZE" "PATH"
  for d in "${MODEL_ROOTS[@]}"; do
    [ -d "$d" ] || continue
    command find -L "$d" -type f -iname '*.gguf' 2>/dev/null   # -L: follow HF-cache symlinks
  done | command grep -iv 'mmproj' | command grep -v '/blobs/' | sort -u | while read -r f; do
    printf '%-10s  %s\n' "$(command du -hL "$f" 2>/dev/null | cut -f1)" "$f"
  done
}

# resolve a search term to the best local .gguf (largest non-mmproj match); empty if none
resolve_local_gguf() {
  local term="$1" d
  for d in "${MODEL_ROOTS[@]}"; do
    [ -d "$d" ] || continue
    command find -L "$d" -type f -iname '*.gguf' 2>/dev/null   # -L: follow HF-cache symlinks
  done | command grep -iv 'mmproj' | command grep -v '/blobs/' | command grep -i -- "$term" \
    | while read -r f; do printf '%s\t%s\n' "$(file_size "$f")" "$f"; done \
    | sort -rn | head -1 | cut -f2- || true
}

# all local model paths, one per line (for the interactive picker)
collect_local_models() {
  local d
  for d in "${MODEL_ROOTS[@]}"; do
    [ -d "$d" ] || continue
    command find -L "$d" -type f -iname '*.gguf' 2>/dev/null
  done | command grep -iv 'mmproj' | command grep -v '/blobs/' | sort -u || true
}

# normalize a context size: accepts 32k / 64K / 128k / 131072 -> token count
to_tokens() {
  local v="$1"
  case "$v" in
    *[kK]) echo $(( ${v%[kK]} * 1024 )) ;;
    *[mM]) echo $(( ${v%[mM]} * 1024 * 1024 )) ;;
    *)     echo "$v" ;;
  esac
}

# is a server already running? prints its cmdline if so.
# Detect by exact process NAME so it matches whatever `stop` (pkill -x) kills,
# regardless of how/where the server was launched (avoids stacking a 2nd server
# on top of one already holding VRAM). Also catches an MLX server.
server_running() {
  local pid
  pid="$(pgrep -x llama-server 2>/dev/null | head -1 || true)"
  [ -n "$pid" ] || pid="$(pgrep -f 'mlx_lm.server|mlx_lm server' 2>/dev/null | head -1 || true)"
  [ -n "$pid" ] || return 0
  ps -o args= -p "$pid" 2>/dev/null || true
}

# ---- build requirements per platform (deps subcommand) ---------------------
install_deps() {
  local backend="$1"
  echo ">> installing build requirements for backend=$backend ..."
  if [ "$OS" = "Darwin" ]; then
    xcode-select -p >/dev/null 2>&1 || xcode-select --install || true
    if command -v brew >/dev/null 2>&1; then
      brew install cmake git curl
    else
      echo "!! Homebrew not found (https://brew.sh) — need cmake + git on top of Xcode CLT" >&2
    fi
    [ "$backend" = "mlx" ] && python3 -m pip install --upgrade mlx-lm
    echo ">> deps done."
    return 0
  fi

  local SUDO=""
  [ "$(id -u)" -ne 0 ] && SUDO="sudo"

  if command -v pacman >/dev/null 2>&1; then
    $SUDO pacman -S --needed --noconfirm base-devel cmake git curl python
    case "$backend" in
      cuda)   $SUDO pacman -S --needed --noconfirm cuda ;;
      rocm)   $SUDO pacman -S --needed --noconfirm rocm-hip-sdk rocminfo ;;
      vulkan) $SUDO pacman -S --needed --noconfirm vulkan-headers vulkan-icd-loader vulkan-tools shaderc ;;
    esac
  elif command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y build-essential cmake git libcurl4-openssl-dev python3
    case "$backend" in
      cuda)   $SUDO apt-get install -y nvidia-cuda-toolkit \
                || echo "!! install the CUDA toolkit from NVIDIA's repo: https://developer.nvidia.com/cuda-downloads" >&2 ;;
      rocm)   $SUDO apt-get install -y rocm-hip-sdk rocminfo \
                || echo "!! ROCm needs AMD's repo first: https://rocm.docs.amd.com/projects/install-on-linux/" >&2 ;;
      vulkan) $SUDO apt-get install -y libvulkan-dev glslc vulkan-tools \
                || $SUDO apt-get install -y libvulkan-dev glslang-tools vulkan-tools ;;
    esac
  elif command -v dnf >/dev/null 2>&1; then
    $SUDO dnf install -y gcc-c++ cmake git libcurl-devel python3
    case "$backend" in
      cuda)   echo "!! CUDA on Fedora: enable NVIDIA's repo, then: dnf install cuda-toolkit" >&2 ;;
      rocm)   $SUDO dnf install -y rocm-hip-devel hipblas-devel rocminfo \
                || echo "!! see https://rocm.docs.amd.com/projects/install-on-linux/" >&2 ;;
      vulkan) $SUDO dnf install -y vulkan-headers vulkan-loader-devel glslc vulkan-tools ;;
    esac
  elif command -v zypper >/dev/null 2>&1; then
    $SUDO zypper install -y gcc-c++ cmake git libcurl-devel python3
    case "$backend" in
      cuda)   echo "!! CUDA on openSUSE: add NVIDIA's repo, then: zypper install cuda-toolkit" >&2 ;;
      rocm)   echo "!! ROCm on openSUSE: see https://rocm.docs.amd.com/projects/install-on-linux/" >&2 ;;
      vulkan) $SUDO zypper install -y vulkan-headers vulkan-loader-devel shaderc vulkan-tools ;;
    esac
  else
    echo "!! unknown package manager — install manually: C++ toolchain, cmake, git, libcurl dev" >&2
    return 1
  fi
  echo ">> deps done."
}

# ---- clone (if needed) + configure + build llama.cpp (build subcommand) ----
build_llama() {
  local backend="$1" flags=() jobs g cap gfx
  if [ "$backend" = "mlx" ]; then
    echo ">> mlx uses mlx-lm, not llama.cpp — nothing to build (pip install mlx-lm)"
    return 0
  fi
  if [ ! -d "$LLAMA_DIR" ]; then
    echo ">> cloning llama.cpp into $LLAMA_DIR ..."
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_DIR"
  fi
  case "$backend" in
    cuda)
      [ -d /opt/cuda ] && export CUDAToolkit_ROOT=/opt/cuda
      flags=(-DGGML_CUDA=ON)
      cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
      [ -n "$cap" ] && flags+=(-DCMAKE_CUDA_ARCHITECTURES="$cap")
      # nvcc often rejects the newest system gcc; prefer an older versioned one
      for g in gcc-15 gcc-14 gcc-13; do
        if command -v "$g" >/dev/null 2>&1; then
          flags+=(-DCMAKE_CUDA_HOST_COMPILER="$(command -v "$g")"); break
        fi
      done ;;
    rocm)
      gfx="$(rocminfo 2>/dev/null | command grep -o 'gfx[0-9a-f]*' | command grep -v '^gfx\(000\)\?$' | head -1 || true)"
      flags=(-DGGML_HIP=ON)
      [ -n "$gfx" ] && flags+=(-DAMDGPU_TARGETS="$gfx" -DGPU_TARGETS="$gfx") ;;
    vulkan) flags=(-DGGML_VULKAN=ON) ;;
    metal|cpu) ;;   # cmake defaults are right (Metal is on by default on Apple silicon)
    *) echo "!! unknown backend '$backend' (cuda|rocm|vulkan|metal|cpu|mlx)" >&2; return 1 ;;
  esac
  if [ "$OS" = "Darwin" ]; then jobs="$(sysctl -n hw.ncpu)"; else jobs="$(nproc)"; fi
  echo ">> cmake configure: ${flags[*]:-'(defaults)'}"
  cmake -S "$LLAMA_DIR" -B "$LLAMA_DIR/build" -DCMAKE_BUILD_TYPE=Release ${flags[@]+"${flags[@]}"}
  cmake --build "$LLAMA_DIR/build" --config Release -j "$jobs"
  echo ">> built: $LLAMA_DIR/build/bin/llama-server"
}

# ---- subcommands: list / stop / status / deps / build ----------------------
case "${1:-}" in
  list|--list)
    echo "Local GGUF models (usable with -m <substring>, no re-download):"
    list_local_models
    exit 0
    ;;
  stop)
    if [ -n "$(server_running)" ]; then
      echo ">> stopping server..."
      pkill -x llama-server 2>/dev/null || true
      pkill -f 'mlx_lm.server|mlx_lm server' 2>/dev/null || true
      sleep 1
      echo ">> stopped."
    else
      echo ">> no server running."
    fi
    exit 0
    ;;
  status)
    running="$(server_running)"
    if [ -n "$running" ]; then
      echo ">> RUNNING: $running"
    else
      echo ">> no server running."
    fi
    exit 0
    ;;
  deps|install-deps)
    install_deps "${2:-$(detect_backend)}"
    exit 0
    ;;
  build)
    build_llama "${2:-$(detect_backend)}"
    exit 0
    ;;
esac

# ---- arg parsing ----------------------------------------------------------
while [ $# -gt 0 ]; do
  case "$1" in
    -m|--model)    MODEL="$2"; shift 2;;
    -c|--ctx)      CONTEXT="$2"; shift 2;;
    --ncmoe)       NCMOE="$2"; shift 2;;
    -t|--threads)  THREADS="$2"; shift 2;;
    --kv)          KVTYPE="$2"; shift 2;;
    --draft|-md)   DRAFT="$2"; shift 2;;
    --host)        HOST="$2"; shift 2;;
    --port)        PORT="$2"; shift 2;;
    --backend)     BACKEND="$2"; shift 2;;
    --)            shift; EXTRA+=("$@"); break;;
    -h|--help)     sed -n '2,33p' "$0"; exit 0;;
    *)             EXTRA+=("$1"); shift;;
  esac
done

[ -n "$BACKEND" ] || BACKEND="$(detect_backend)"
[ -n "$THREADS" ] || THREADS="$(physical_cores)"
echo ">> backend=$BACKEND"

# ---- if a server is already up, just report it and exit -------------------
running="$(server_running)"
if [ -n "$running" ]; then
  rport="$(printf '%s\n' "$running" | command grep -oE -- '--port [0-9]+' | command grep -oE '[0-9]+' | head -1)"
  echo ">> A server is already running:"
  echo "   $running"
  [ -n "$rport" ] && echo ">> Endpoint: http://${HOST}:${rport}/v1"
  echo ">> To switch models: ./run-llm.sh stop   (then run this again)"
  exit 0
fi

# ---- pick a free port (auto-advance if the requested one is busy) ----------
orig_port="$PORT"
while port_busy "$PORT"; do
  PORT=$((PORT + 1))
  if [ "$PORT" -gt $((orig_port + 20)) ]; then
    echo "!! no free port found near $orig_port" >&2
    exit 1
  fi
done
[ "$PORT" != "$orig_port" ] && echo ">> port $orig_port busy -> using $PORT"

# ---- MLX backend: hand off to mlx_lm.server (no GGUF machinery) ------------
if [ "$BACKEND" = "mlx" ]; then
  if [ "$OS" != "Darwin" ]; then
    echo "!! mlx backend only works on macOS (Apple silicon)" >&2; exit 1
  fi
  if [ -z "$MODEL" ]; then
    echo "!! mlx backend needs -m <hf-repo> (e.g. mlx-community/Qwen3-8B-4bit)" >&2; exit 1
  fi
  if command -v lms >/dev/null 2>&1; then
    echo ">> Unloading LM Studio models to free memory..."
    lms unload --all || true
  fi
  echo ">> model=$MODEL"
  echo ">> serving http://${HOST}:${PORT}/v1  (Ctrl-C to stop)"
  if command -v mlx_lm.server >/dev/null 2>&1; then
    exec mlx_lm.server --model "$MODEL" --host "$HOST" --port "$PORT" ${EXTRA[@]+"${EXTRA[@]}"}
  else
    exec python3 -m mlx_lm server --model "$MODEL" --host "$HOST" --port "$PORT" ${EXTRA[@]+"${EXTRA[@]}"}
  fi
fi

# ---- no model specified -> interactive picker (dynamic mode) ---------------
if [ -z "$MODEL" ]; then
  MODELS=()
  while IFS= read -r m; do [ -n "$m" ] && MODELS+=("$m"); done < <(collect_local_models)
  if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "No local GGUF models found. Specify one to download: $0 -m <hf-repo:quant>"
    exit 1
  fi
  echo "Available local models:"
  idx=1
  for m in "${MODELS[@]}"; do
    printf "  %2d) %-7s %s\n" "$idx" "$(command du -hL "$m" 2>/dev/null | cut -f1)" "$(basename "$m")"
    idx=$((idx + 1))
  done
  printf "Select model [1-%d] (q to quit): " "${#MODELS[@]}"
  read -r choice
  [ "$choice" = "q" ] && exit 0
  if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#MODELS[@]}" ]; then
    echo "Invalid selection."
    exit 1
  fi
  MODEL="${MODELS[$((choice - 1))]}"
  echo ">> selected: $(basename "$MODEL")"
fi

# ---- normalize context (accepts 32k / 64k / 128k or a raw token count) -----
CONTEXT="$(to_tokens "$CONTEXT")"
if ! [[ "$CONTEXT" =~ ^[0-9]+$ ]]; then
  echo "!! invalid context size: '$CONTEXT' (use e.g. 32k, 64k, 128k, or a number)" >&2
  exit 1
fi

# NCMOE (expert offload) is computed after `lms unload`, so the estimate sees
# the real free VRAM — see the smart-offload block below.

# runtime library paths (only where the toolkit lives outside the loader path)
case "$BACKEND" in
  cuda) [ -d /opt/cuda/lib64 ] && export LD_LIBRARY_PATH="/opt/cuda/lib64:${LD_LIBRARY_PATH:-}" ;;
  rocm) [ -d /opt/rocm/lib ]   && export LD_LIBRARY_PATH="/opt/rocm/lib:${LD_LIBRARY_PATH:-}" ;;
esac

# ---- model source resolution (avoid re-downloading what you already have) ----
#   1) exact local file path -> use it
#   2) substring match against local GGUFs (LM Studio + caches) -> use it
#   3) otherwise treat as HF repo tag -> -hf (downloads, or reuses HF cache)
MODEL_ARGS=()
if [ -f "$MODEL" ]; then
  MODEL_ARGS=(-m "$MODEL")
  echo ">> using local file: $MODEL"
else
  LOCAL_HIT="$(resolve_local_gguf "$MODEL")"
  if [ -n "$LOCAL_HIT" ]; then
    MODEL_ARGS=(-m "$LOCAL_HIT")
    echo ">> matched local model (no download): $LOCAL_HIT"
  else
    MODEL_ARGS=(-hf "$MODEL")
    echo ">> no local match; fetching from Hugging Face (reuses cache if present): $MODEL"
  fi
fi

# ---- friendly alias (clean name reported by /v1/models -> opencode picker) -
case "${MODEL_ARGS[0]}" in
  -m)  ALIAS="$(basename "${MODEL_ARGS[1]}")"; ALIAS="${ALIAS%.gguf}" ;;
  -hf) ALIAS="${MODEL##*/}"; ALIAS="${ALIAS%%:*}" ;;
  *)   ALIAS="model" ;;
esac

# ---- optional speculative decoding ---------------------------------------
SPEC_ARGS=()
if [ -n "$DRAFT" ]; then
  if [ -f "$DRAFT" ]; then SPEC_ARGS=(-md "$DRAFT"); else SPEC_ARGS=(--hf-repo-draft "$DRAFT"); fi
  SPEC_ARGS+=(-ngld 99 --spec-draft-n-max 16 --spec-draft-n-min 0)
fi

# ---- free the GPU ---------------------------------------------------------
if command -v lms >/dev/null 2>&1; then
  echo ">> Unloading LM Studio models to free VRAM..."
  lms unload --all || true
  sleep 1
fi
echo ">> GPU before launch:"
gpu_status "$BACKEND"

# ---- smart expert-offload estimate (model + free-VRAM + context aware) ------
# KV bits-per-element for the chosen cache type (used to size the KV reserve).
case "$KVTYPE" in
  q4_0|q4_1|iq4_nl) KVBITS=4.5 ;;
  q5_0|q5_1)        KVBITS=5.5 ;;
  q8_0)             KVBITS=8.5 ;;
  f16|bf16)         KVBITS=16  ;;
  *)                KVBITS=16  ;;
esac

NCMOE_CAP=64        # upper bound for auto-retry; refined to layer count if known
if [ -z "$NCMOE" ]; then
  case "$BACKEND" in
    metal|cpu)
      # unified memory / no discrete VRAM: expert offload buys nothing by default
      NCMOE=0
      ;;
    *)
      FREE_MIB="$(free_vram_mib "$BACKEND")"
      est=""
      if [ "${MODEL_ARGS[0]}" = "-m" ] && [ -n "${FREE_MIB:-}" ] && command -v python3 >/dev/null 2>&1; then
        est="$(python3 "$(dirname "$0")/gguf-estimate.py" "${MODEL_ARGS[1]}" "$FREE_MIB" "$CONTEXT" "$PARALLEL" "$KVBITS" 2>/dev/null || true)"
      fi
      if [ -n "$est" ]; then
        NCMOE="${est%% *}"                 # "<ncmoe> <layers>"
        NCMOE_CAP="${est##* }"
        echo ">> estimated n-cpu-moe=$NCMOE (of $NCMOE_CAP layers) for ${FREE_MIB}MiB free, ctx=$CONTEXT"
      else                                 # fallback: context buckets
        if   [ "$CONTEXT" -le 16384 ]; then NCMOE=28
        elif [ "$CONTEXT" -le 65536 ]; then NCMOE=32
        else                                NCMOE=40
        fi
        echo ">> estimate unavailable; using fallback n-cpu-moe=$NCMOE"
      fi
      ;;
  esac
fi

# mmap is the right default on unified memory (Metal); --no-mmap elsewhere so
# CPU-side expert weights land in real RAM instead of thrashing page cache.
MMAP_ARGS=(--no-mmap)
[ "$BACKEND" = "metal" ] && MMAP_ARGS=()

# ---- launch ---------------------------------------------------------------
echo ">> model=$MODEL"
echo ">> context=$CONTEXT  threads=$THREADS  kv=$KVTYPE  parallel=$PARALLEL  draft=${DRAFT:-none}"
echo ">> serving http://${HOST}:${PORT}/v1  (Ctrl-C to stop)"
cd "$LLAMA_DIR"

LLAMA_BIN="${LLAMA_BIN:-./build/bin/llama-server}"
if [ ! -x "$LLAMA_BIN" ]; then
  echo "!! $LLAMA_BIN not found. Build llama.cpp for backend '$BACKEND' (see header: $0 -h)" >&2
  exit 1
fi

# ---- launch with OOM auto-retry: bump expert-offload until it fits ----------
LOG="$(mktemp "${TMPDIR:-/tmp}/run-llm.XXXXXX")"
SRV=""; TAILER=""
cleanup() { [ -n "$TAILER" ] && kill "$TAILER" 2>/dev/null; [ -n "$SRV" ] && kill "$SRV" 2>/dev/null; rm -f "$LOG"; }
trap cleanup EXIT
trap 'exit 130' INT TERM

# device OOM signatures across backends (CUDA / HIP / Vulkan / Metal)
OOM_RE='out of memory|failed to allocate|failed to create context|cudaMalloc failed|hipMalloc failed|hipErrorOutOfMemory|VK_ERROR_OUT_OF_DEVICE_MEMORY|ErrorOutOfDeviceMemory|Insufficient Memory|kIOGPUCommandBufferCallbackErrorOutOfMemory'

STEP=3
while :; do
  : > "$LOG"
  echo ">> launching with n-cpu-moe=$NCMOE ..."
  "$LLAMA_BIN" \
    "${MODEL_ARGS[@]}" \
    --alias "$ALIAS" \
    -ngl 99 \
    --n-cpu-moe "$NCMOE" \
    -t "$THREADS" \
    -fa 1 \
    -ctk "$KVTYPE" -ctv "$KVTYPE" \
    -c "$CONTEXT" \
    --parallel "$PARALLEL" \
    ${MMAP_ARGS[@]+"${MMAP_ARGS[@]}"} \
    --jinja \
    ${SPEC_ARGS[@]+"${SPEC_ARGS[@]}"} \
    --host "$HOST" --port "$PORT" \
    ${EXTRA[@]+"${EXTRA[@]}"} >"$LOG" 2>&1 &
  SRV=$!
  tail -f -n +1 "$LOG" &   # stream server logs live (killed in cleanup; no GNU --pid)
  TAILER=$!

  # watch for readiness vs. out-of-memory while the server is starting
  outcome=""
  while kill -0 "$SRV" 2>/dev/null; do
    if command grep -q "server is listening" "$LOG"; then outcome="ok"; break; fi
    if command grep -qiE "$OOM_RE" "$LOG"; then
      outcome="oom"; break
    fi
    sleep 0.5
  done

  if [ "$outcome" = "ok" ]; then
    wait "$SRV"; exit $?               # serve until stopped
  fi

  # not ok: stop this attempt, then either retry (OOM) or give up
  kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null || true
  kill "$TAILER" 2>/dev/null; TAILER=""
  if [ "$outcome" != "oom" ]; then
    echo "!! server exited without becoming ready (not an OOM). Last log lines:" >&2
    tail -n 15 "$LOG" >&2
    exit 1
  fi
  NCMOE=$((NCMOE + STEP))
  if [ "$NCMOE" -gt "$NCMOE_CAP" ]; then
    echo "!! still OOM at max offload (n-cpu-moe=$NCMOE_CAP). Try smaller -c (context) or a smaller model." >&2
    exit 1
  fi
  echo ">> OOM — retrying with more CPU offload (n-cpu-moe=$NCMOE)"
done
