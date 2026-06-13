#!/usr/bin/env bash
# localagent — run a headless `claude` sub-agent powered by the LOCAL llama-server,
# isolated in a git worktree, so the orchestrating Claude (Opus) can VALIDATE the
# result (diff + transcript) before anything lands on the real branch.
#
#   pipeline:  opus  ->  localagent run  ->  claude --bare -p (LOCAL brain)
#                                              -> anthropic_proxy.py -> llama-server
#
# Subcommands:
#   localagent proxy start|stop|status|restart
#   localagent run   --task "..."  [--repo DIR] [--in-place] [--readonly]
#                    [--tools "Bash,Read,..."] [--timeout SECS] [--name LABEL]
#   localagent list                         # show pending worktrees awaiting validation
#   localagent diff   <branch>              # re-print the diff for a run
#   localagent apply  <branch> [--into REF] # fast-forward/merge the branch into the repo
#   localagent discard <branch>             # delete the worktree + branch
#   localagent export [--format FMT] [-o OUT] # export logged dataset calls
#   localagent info                         # show proxy and database status/size
#
# All sub-agent output (transcript, diff, meta) is written under:
#   <repo>/.git/localagent/<branch>/
set -euo pipefail

SELF="$(readlink -f "${BASH_SOURCE[0]}")"   # follow the ~/.local/bin symlink
HERE="$(cd "$(dirname "$SELF")" && pwd)"
PROXY_PY="$HERE/anthropic_proxy.py"
PROXY_HOST="${LLAMA_PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${LLAMA_PROXY_PORT:-8090}"
PROXY_URL="http://$PROXY_HOST:$PROXY_PORT"
LLAMA_BASE="${LLAMA_BASE:-http://127.0.0.1:8089}"
STATE_DIR="${LOCALAGENT_STATE:-$HOME/.local/state/localagent}"
PIDFILE="$STATE_DIR/proxy.pid"
PROXY_LOG="$STATE_DIR/proxy.log"

# default tool sets (which tool SCHEMAS are sent to the weak model — smaller = better)
TOOLS_FULL="Bash,Read,Write,Edit,Grep,Glob"
TOOLS_READONLY="Read,Grep,Glob,Bash"

mkdir -p "$STATE_DIR"

c_red()  { printf '\033[31m%s\033[0m' "$*"; }
c_grn()  { printf '\033[32m%s\033[0m' "$*"; }
c_dim()  { printf '\033[2m%s\033[0m' "$*"; }
die()    { echo "$(c_red error:) $*" >&2; exit 1; }

# ---------------------------------------------------------------- proxy mgmt
proxy_healthy() { curl -s --max-time 3 "$PROXY_URL/health" 2>/dev/null | grep -q '"ok": true'; }

proxy_start() {
  if proxy_healthy; then echo "proxy already up at $PROXY_URL"; return 0; fi
  command -v python3 >/dev/null || die "python3 not found"
  curl -s --max-time 3 "$LLAMA_BASE/v1/models" >/dev/null 2>&1 \
    || echo "$(c_red warn:) llama-server not reachable at $LLAMA_BASE (start it with highllama)" >&2
  echo "starting proxy: $PROXY_URL -> $LLAMA_BASE"
  LLAMA_PROXY_HOST="$PROXY_HOST" LLAMA_PROXY_PORT="$PROXY_PORT" \
  LLAMA_BASE="$LLAMA_BASE" LLAMA_PROXY_LOG="$PROXY_LOG" \
    setsid nohup python3 "$PROXY_PY" >>"$PROXY_LOG" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 30); do proxy_healthy && { echo "proxy ready (pid $(cat "$PIDFILE"))"; return 0; }; sleep 0.3; done
  die "proxy failed to become healthy; see $PROXY_LOG"
}

proxy_stop() {
  [[ -f "$PIDFILE" ]] && kill "$(cat "$PIDFILE")" 2>/dev/null || true
  pkill -f "$PROXY_PY" 2>/dev/null || true
  rm -f "$PIDFILE"
  echo "proxy stopped"
}

proxy_status() {
  if proxy_healthy; then
    local model; model=$(curl -s "$PROXY_URL/health" | python3 -c 'import sys,json;print(json.load(sys.stdin)["model"])')
    echo "$(c_grn up)   $PROXY_URL  model=$model  upstream=$LLAMA_BASE"
  else
    echo "$(c_red down) $PROXY_URL"
  fi
}

resolve_model() {
  curl -s --max-time 3 "$PROXY_URL/health" 2>/dev/null \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["model"])' 2>/dev/null \
    || echo "local-model"
}

# ---------------------------------------------------------------- run
cmd_run() {
  local repo task="" tools="$TOOLS_FULL" timeout=600 in_place=0 readonly=0 name=""
  repo="$(pwd)"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --task)      task="$2"; shift 2;;
      --task-file) task="$(cat "$2")"; shift 2;;
      --repo)      repo="$(cd "$2" && pwd)"; shift 2;;
      --tools)     tools="$2"; shift 2;;
      --timeout)   timeout="$2"; shift 2;;
      --name)      name="$2"; shift 2;;
      --in-place)  in_place=1; shift;;
      --readonly)  readonly=1; tools="$TOOLS_READONLY"; shift;;
      *) die "unknown flag: $1";;
    esac
  done
  [[ -n "$task" ]] || { task="$(cat)"; }   # fall back to stdin
  [[ -n "$task" ]] || die "no --task given"

  proxy_start >&2
  local model; model="$(resolve_model)"

  local is_git=0; git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 && is_git=1
  local workdir="$repo" branch="" outdir base=""

  if [[ $readonly -eq 1 || $in_place -eq 1 || $is_git -eq 0 ]]; then
    # no isolation: investigate in place (read-only) or caller opted in
    workdir="$repo"
    outdir="$STATE_DIR/runs/$(date +%Y%m%d-%H%M%S)-${name:-run}"
    mkdir -p "$outdir"
    [[ $is_git -eq 1 ]] && base="$(git -C "$repo" rev-parse HEAD 2>/dev/null || true)"
  else
    # isolated worktree on a fresh branch. NB: artifacts (outdir) must live OUTSIDE
    # the worktree, or `git add -A` would sweep them into the diff.
    base="$(git -C "$repo" rev-parse HEAD)"
    branch="la/${name:+$name-}$(date +%Y%m%d-%H%M%S)"
    local slug="${branch//\//_}"
    workdir="$repo/.git/localagent/wt/$slug"
    outdir="$repo/.git/localagent/meta/$slug"
    mkdir -p "$(dirname "$workdir")" "$outdir"
    git -C "$repo" worktree add -q -b "$branch" "$workdir" "$base"
  fi

  echo "$(c_dim "task     :") $task" >&2
  echo "$(c_dim "repo     :") $repo" >&2
  echo "$(c_dim "workdir  :") $workdir" >&2
  echo "$(c_dim "branch   :") ${branch:-<in-place>}" >&2
  echo "$(c_dim "tools    :") $tools   $(c_dim "model:")$model   $(c_dim "readonly:")$readonly" >&2
  echo "$(c_dim "running local sub-agent... (timeout ${timeout}s)")" >&2

  local perm="bypassPermissions"
  [[ $readonly -eq 1 ]] && perm="bypassPermissions"   # still bypass to avoid prompts; tools are read-only set
  local transcript="$outdir/transcript.json"
  local rc=0
  ( cd "$workdir"
    ANTHROPIC_BASE_URL="$PROXY_URL" \
    ANTHROPIC_API_KEY="local" \
    ANTHROPIC_MODEL="$model" \
    DISABLE_AUTOUPDATER=1 DISABLE_TELEMETRY=1 DO_NOT_TRACK=1 \
    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 DISABLE_PROMPT_CACHING=1 \
    timeout "$timeout" claude --bare -p "$task" \
      --model "$model" \
      --tools "$tools" \
      --permission-mode "$perm" \
      --no-session-persistence \
      --output-format json
  ) > "$transcript" 2>"$outdir/stderr.log" || rc=$?

  # summarize
  local result_txt
  result_txt="$(python3 -c '
import sys,json
try:
    d=json.load(open(sys.argv[1]))
    print(d.get("result",""))
except Exception as e:
    print("[no parseable result] "+str(e))
' "$transcript" 2>/dev/null || echo "[transcript unreadable]")"

  echo >&2
  echo "================ local sub-agent result ================" >&2
  if [[ $rc -ne 0 ]]; then echo "$(c_red "exit code: $rc (timeout/error — see $outdir/stderr.log)")" >&2; fi
  echo "$result_txt"
  echo "========================================================" >&2

  # changes report
  if [[ $is_git -eq 1 ]]; then
    local diff_file="$outdir/changes.diff"
    local nfiles
    if [[ -n "$branch" ]]; then
      # isolated worktree: safe to `add -A` to capture untracked files in the diff
      git -C "$workdir" add -A >/dev/null 2>&1 || true
      git -C "$workdir" diff --cached ${base:+"$base"} > "$diff_file" 2>/dev/null || git -C "$workdir" diff --cached > "$diff_file" 2>/dev/null || true
      nfiles=$(git -C "$workdir" diff --cached --name-only ${base:+"$base"} 2>/dev/null | wc -l | tr -d ' ')
    else
      # in-place: never touch the real index; report unstaged + untracked read-only
      { git -C "$workdir" diff; git -C "$workdir" ls-files --others --exclude-standard; } > "$diff_file" 2>/dev/null || true
      nfiles=$(git -C "$workdir" status --porcelain 2>/dev/null | wc -l | tr -d ' ')
    fi
    {
      echo
      echo "$(c_dim "changed files: $nfiles")"
      if [[ -n "$branch" ]]; then
        git -C "$workdir" diff --cached --stat ${base:+"$base"} 2>/dev/null || true
      else
        git -C "$workdir" status --short 2>/dev/null || true
      fi
      echo
      echo "$(c_dim "full diff : $diff_file")"
      echo "$(c_dim "transcript: $transcript")"
      [[ -n "$branch" ]] && echo "$(c_dim "validate then:") localagent apply $branch   $(c_dim "|") localagent discard $branch"
    } >&2
    # also emit machine-readable footer on stdout for the orchestrator
    echo "::localagent:: repo=$repo branch=${branch:-} workdir=$workdir diff=$diff_file transcript=$transcript files=$nfiles exit=$rc"
  fi
}

# ---------------------------------------------------------------- list/diff/apply/discard
find_repo_for_branch() { git -C "${2:-$(pwd)}" rev-parse --git-dir >/dev/null 2>&1 && pwd; }

cmd_list() {
  local repo="${1:-$(pwd)}"
  local d="$repo/.git/localagent"
  [[ -d "$d" ]] || { echo "no pending runs in $repo"; return 0; }
  echo "pending localagent worktrees in $repo:"
  git -C "$repo" worktree list 2>/dev/null | grep localagent || echo "  (none active)"
}

cmd_diff() {
  local branch="$1"; local repo="${2:-$(pwd)}"
  local slug="${branch//\//_}"
  local f="$repo/.git/localagent/meta/$slug/changes.diff"
  [[ -f "$f" ]] || die "no diff at $f"
  cat "$f"
}

cmd_apply() {
  local branch="$1"; shift || true
  local repo="$(pwd)" into=""
  while [[ $# -gt 0 ]]; do case "$1" in --repo) repo="$2"; shift 2;; --into) into="$2"; shift 2;; *) shift;; esac; done
  git -C "$repo" rev-parse --git-dir >/dev/null 2>&1 || die "not a git repo: $repo"
  local slug="${branch//\//_}"
  local wt="$repo/.git/localagent/wt/$slug"
  # if the sub-agent left uncommitted changes, commit them on the branch first
  if [[ -d "$wt" ]] && ! git -C "$wt" diff --quiet HEAD 2>/dev/null; then
    git -C "$wt" add -A && git -C "$wt" commit -q -m "localagent: $branch (validated)" || true
  fi
  echo "merging $branch into current branch of $repo"
  git -C "$repo" merge --no-ff "$branch" -m "Merge validated localagent run: $branch"
  echo "$(c_grn done). cleaning worktree."
  git -C "$repo" worktree remove --force "$wt" 2>/dev/null || true
}

cmd_discard() {
  local branch="$1"; local repo="${2:-$(pwd)}"
  local slug="${branch//\//_}"
  local wt="$repo/.git/localagent/wt/$slug"
  git -C "$repo" worktree remove --force "$wt" 2>/dev/null || true
  git -C "$repo" branch -D "$branch" 2>/dev/null || true
  rm -rf "$wt" "$repo/.git/localagent/meta/$slug"
  echo "discarded $branch"
}

cmd_info() {
  # Mirror the proxy's resolve_dataset_paths() exactly so we always report
  # the same DB the proxy is actually writing to.
  local dataset_path jsonl_file db_file
  if [[ -n "${LLAMA_PROXY_DATASET:-}" ]]; then
    dataset_path="$LLAMA_PROXY_DATASET"
  elif [[ -n "${LLAMA_PROXY_LOG:-}" ]]; then
    dataset_path="$(dirname "$LLAMA_PROXY_LOG")/dataset.jsonl"
  else
    dataset_path="$STATE_DIR/dataset.jsonl"
  fi
  db_file="${dataset_path%.jsonl}.db"
  if [[ "$db_file" == "$dataset_path" ]]; then
    # dataset_path had no .jsonl extension to strip — append .db
    db_file="${dataset_path}.db"
  fi
  
  echo "$(c_grn "LocalAgent Session Information:")"
  echo "----------------------------------------"
  
  # Proxy Status
  if proxy_healthy; then
    local model; model=$(curl -s "$PROXY_URL/health" | python3 -c 'import sys,json;print(json.load(sys.stdin)["model"])' 2>/dev/null || echo "unknown")
    echo "Proxy Status : $(c_grn "UP") ($PROXY_URL)"
    echo "Active Model : $model"
    echo "Upstream API : $LLAMA_BASE"
  else
    echo "Proxy Status : $(c_red "DOWN") ($PROXY_URL)"
  fi
  
  echo "Proxy Log    : $PROXY_LOG"
  if [[ -f "$PROXY_LOG" ]]; then
    echo "  Log Size   : $(du -sh "$PROXY_LOG" | awk '{print $1}')"
  fi
  
  echo "Database     : $db_file"
  if [[ -f "$db_file" ]]; then
    echo "  DB Size    : $(du -sh "$db_file" | awk '{print $1}')"
    # Row count using sqlite3
    if command -v sqlite3 >/dev/null; then
      local count; count=$(sqlite3 "$db_file" "SELECT COUNT(*) FROM dataset_calls" 2>/dev/null || echo "0")
      echo "  Log Count  : $count entries"
    fi
  else
    echo "  DB Size    : 0B (not created yet)"
  fi
  
  # Raw jsonl if it exists
  local jsonl_file="${db_file%.db}.jsonl"
  if [[ -f "$jsonl_file" ]]; then
    echo "Legacy JSONL : $jsonl_file"
    echo "  File Size  : $(du -sh "$jsonl_file" | awk '{print $1}')"
  fi
}

# ---------------------------------------------------------------- dispatch
usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; }

cmd="${1:-}"; shift || true
case "$cmd" in
  proxy)
    sub="${1:-status}"; shift || true
    case "$sub" in
      start) proxy_start;; stop) proxy_stop;; status) proxy_status;;
      restart) proxy_stop; proxy_start;; *) die "proxy: start|stop|status|restart";;
    esac;;
  run)     cmd_run "$@";;
  list)    cmd_list "$@";;
  diff)    cmd_diff "$@";;
  apply)   cmd_apply "$@";;
  discard) cmd_discard "$@";;
  export)  python3 "$HERE/export_dataset.py" "$@";;
  info)    cmd_info;;
  ""|-h|--help|help) usage;;
  *) die "unknown command: $cmd (try: proxy|run|list|diff|apply|discard|export|info)";;
esac
