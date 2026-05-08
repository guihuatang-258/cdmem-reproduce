#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CC_CONNECT_CONFIG:-$HOME/.cc-connect/config.toml}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/.git" ]]; then
  WORK_DIR_DEFAULT="$SCRIPT_DIR"
else
  WORK_DIR_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
DOCKER_HOST_DEFAULT="unix:///mnt/wsl/docker-desktop-bind-mounts/Ubuntu/docker.sock"

usage() {
  cat <<'EOF'
cc-connect-switch.sh - quick cc-connect project/agent/platform presets

Usage:
  cc-connect-switch.sh show
  cc-connect-switch.sh status
  cc-connect-switch.sh restart
  cc-connect-switch.sh logs

  cc-connect-switch.sh single --agent codex|claude --platform weixin|qq|both [--work-dir PATH] [--name NAME] [--restart]
  cc-connect-switch.sh split [--work-dir PATH] [--restart]
  cc-connect-switch.sh model MODEL [EFFORT] [--restart]

  cc-connect-switch.sh qq start|stop|restart|logs|status
  cc-connect-switch.sh napcat start|stop|restart|logs|status

Commands:
  show       Print the current ~/.cc-connect/config.toml.
  status     Show cc-connect daemon status.
  restart    Restart cc-connect daemon with the current config.
  logs       Follow cc-connect daemon logs.
  single     Write one cc-connect project: one agent connected to selected platform(s).
  split      Write two cc-connect projects: Codex+Weixin and Claude+QQ.
  model      Set model/reasoning_effort for all Codex projects in config.toml.
  qq         Manage the full QQ path: cc-connect daemon + local NapCat container.
  napcat     Manage only the local NapCat Docker container used by QQ.

Options:
  --agent codex|claude
             Select the coding agent for single mode.
             codex  -> type = "codex", mode = "auto-edit"
             claude -> type = "claudecode", mode = "auto"

  --platform weixin|qq|both
             Select chat entrance(s) for single mode.
             weixin -> only Weixin connects to the selected agent.
             qq     -> only QQ connects to the selected agent.
             both   -> both Weixin and QQ connect to the same selected agent.

  --work-dir PATH
             Project folder where the coding agent should run.
             Use an absolute path when switching to another repo.
             Default: the folder containing this script.

  --name NAME
             cc-connect project name written into config.toml.
             Optional; generated from work-dir, agent, and platform when omitted.

  --restart
             Restart cc-connect daemon immediately after writing config.
             Without this flag, config is written but not applied until restart.

Model command:
  MODEL      Codex model id. Common aliases:
             gpt54 -> gpt-5.4
             gpt55 -> gpt-5.5
             gpt52 -> gpt-5.2

  EFFORT     Optional Codex reasoning effort.
             Aliases: min -> minimal, mid -> medium, med -> medium,
             hi -> high, xhi -> xhigh.

QQ actions:
  start      Start napcat, then start cc-connect daemon.
  stop       Stop cc-connect daemon, then stop napcat.
  restart    Restart napcat, then restart cc-connect daemon.
  logs       Follow cc-connect daemon logs. Use "napcat logs" for QR login logs.
  status     Show both cc-connect daemon status and napcat container status.

NapCat actions:
  start      Start only the existing napcat container.
  stop       Stop only the napcat container.
  restart    Restart only the napcat container.
  logs       Follow only napcat logs; useful for QR login and OneBot status.
  status     Show only the napcat container status.

Examples:
  # Weixin -> Codex, QQ disabled
  cc-connect-switch.sh single --agent codex --platform weixin --restart

  # QQ -> Claude Code, Weixin disabled
  cc-connect-switch.sh single --agent claude --platform qq --restart

  # One project, both Weixin and QQ use Claude Code
  cc-connect-switch.sh single --agent claude --platform both --restart

  # Two projects at once: Codex+Weixin and Claude+QQ
  cc-connect-switch.sh split --restart

  # Switch current Codex projects to GPT-5.4 medium
  cc-connect-switch.sh model gpt54 mid --restart

  # Start or stop the whole QQ path
  cc-connect-switch.sh qq start
  cc-connect-switch.sh qq stop

Notes:
  - Config file: ~/.cc-connect/config.toml
  - Existing weixin/qq platform blocks are reused from the current config.
  - Every write creates a timestamped backup next to config.toml.
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "missing command: $1"
}

agent_type() {
  case "${1:-}" in
    codex) echo "codex" ;;
    claude|claudecode) echo "claudecode" ;;
    *) die "--agent must be codex or claude" ;;
  esac
}

agent_mode() {
  case "${1:-}" in
    codex) echo "auto-edit" ;;
    claude|claudecode) echo "auto" ;;
    *) die "--agent must be codex or claude" ;;
  esac
}

extract_platform() {
  local want="$1"
  [[ -f "$CONFIG" ]] || return 1

  awk -v want="$want" '
    function flush() {
      if (in_block && type == want) {
        printf "%s", block
        found = 1
        exit
      }
    }

    /^[[:space:]]*\[\[projects\.platforms\]\]/ {
      flush()
      in_block = 1
      type = ""
      block = $0 ORS
      next
    }

    /^[[:space:]]*\[\[projects\]\]/ {
      flush()
      in_block = 0
      block = ""
      type = ""
      next
    }

    in_block {
      block = block $0 ORS
      if ($0 ~ /^[[:space:]]*type[[:space:]]*=/) {
        line = $0
        sub(/^[^=]*=[[:space:]]*/, "", line)
        gsub(/[" ]/, "", line)
        type = line
      }
    }

    END {
      if (!found && in_block && type == want) {
        printf "%s", block
      }
    }
  ' "$CONFIG"
}

qq_default_block() {
  cat <<'EOF'
[[projects.platforms]]
type = "qq"

[projects.platforms.options]
ws_url = "ws://127.0.0.1:3001"
token = ""
allow_from = "*"
share_session_in_channel = false
EOF
}

platform_block() {
  local platform="$1"
  local block
  block="$(extract_platform "$platform" || true)"

  if [[ -n "$block" ]]; then
    printf "%s\n" "$block"
    return
  fi

  case "$platform" in
    qq) qq_default_block ;;
    weixin) die "no existing weixin platform block found in $CONFIG; keep a config with weixin once, then rerun" ;;
    *) die "unknown platform: $platform" ;;
  esac
}

backup_config() {
  mkdir -p "$(dirname "$CONFIG")"
  if [[ -f "$CONFIG" ]]; then
    local backup="${CONFIG}.bak-$(date +%Y%m%d-%H%M%S)"
    cp "$CONFIG" "$backup"
    echo "backup: $backup"
  fi
}

write_header() {
  cat <<'EOF'
# cc-connect configuration
# Generated by cc-connect-switch.sh

language = "zh"

[log]
level = "info"
EOF
}

write_project() {
  local name="$1"
  local agent="$2"
  local work_dir="$3"
  shift 3

  local type mode
  type="$(agent_type "$agent")"
  mode="$(agent_mode "$agent")"

  cat <<EOF

[[projects]]
name = "$name"

[projects.agent]
type = "$type"

[projects.agent.options]
work_dir = "$work_dir"
mode = "$mode"
EOF

  local block
  for block in "$@"; do
    echo
    printf "%s\n" "$block"
  done
}

format_config() {
  if command -v cc-connect >/dev/null 2>&1; then
    cc-connect config format >/dev/null || true
  fi
}

restart_daemon() {
  cc-connect daemon restart
}

normalize_model() {
  case "${1:-}" in
    gpt54|gpt-54|gpt5.4|gpt-5.4) echo "gpt-5.4" ;;
    gpt55|gpt-55|gpt5.5|gpt-5.5) echo "gpt-5.5" ;;
    gpt52|gpt-52|gpt5.2|gpt-5.2) echo "gpt-5.2" ;;
    "") die "model requires MODEL, e.g. gpt54" ;;
    *) echo "$1" ;;
  esac
}

normalize_effort() {
  case "${1:-}" in
    "") echo "" ;;
    min|minimal) echo "minimal" ;;
    low) echo "low" ;;
    mid|med|medium) echo "medium" ;;
    hi|high) echo "high" ;;
    xhi|xhigh) echo "xhigh" ;;
    *) die "unknown reasoning effort: $1" ;;
  esac
}

set_codex_model() {
  local model="" effort="" restart=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --restart) restart=1; shift ;;
      --model) model="${2:-}"; shift 2 ;;
      --effort|--reasoning-effort) effort="${2:-}"; shift 2 ;;
      -* ) die "unknown model arg: $1" ;;
      *)
        if [[ -z "$model" ]]; then
          model="$1"
        elif [[ -z "$effort" ]]; then
          effort="$1"
        else
          die "unexpected model arg: $1"
        fi
        shift
        ;;
    esac
  done

  model="$(normalize_model "$model")"
  effort="$(normalize_effort "$effort")"

  backup_config
  awk -v model="$model" -v effort="$effort" '
    function emit_model_fields(indent) {
      print indent "model = \"" model "\""
      if (effort != "") {
        print indent "reasoning_effort = \"" effort "\""
      }
      wrote = 1
    }

    function leaving_options() {
      if (in_codex_options && !wrote) {
        emit_model_fields(option_indent)
      }
      in_codex_options = 0
      wrote = 0
    }

    /^[[:space:]]*\[\[projects\]\]/ {
      leaving_options()
      current_agent = ""
      in_agent = 0
    }

    /^[[:space:]]*\[projects\.agent\]/ {
      leaving_options()
      in_agent = 1
    }

    /^[[:space:]]*\[projects\.agent\.options\]/ {
      leaving_options()
      if (current_agent == "codex") {
        in_codex_options = 1
        option_indent = ""
        match($0, /^[[:space:]]*/)
        option_indent = substr($0, RSTART, RLENGTH) "  "
      }
    }

    /^[[:space:]]*\[/ && $0 !~ /^[[:space:]]*\[projects\.agent\.options\]/ {
      if (in_codex_options) {
        leaving_options()
      }
    }

    in_agent && /^[[:space:]]*type[[:space:]]*=/ {
      line = $0
      sub(/^[^=]*=[[:space:]]*/, "", line)
      gsub(/[" ]/, "", line)
      current_agent = line
    }

    in_codex_options && /^[[:space:]]*model[[:space:]]*=/ {
      if (!wrote) {
        emit_model_fields(option_indent)
      }
      next
    }

    in_codex_options && /^[[:space:]]*reasoning_effort[[:space:]]*=/ {
      next
    }

    { print }

    END {
      leaving_options()
    }
  ' "$CONFIG" >"${CONFIG}.tmp"
  mv "${CONFIG}.tmp" "$CONFIG"
  format_config
  echo "set Codex model: $model${effort:+ ($effort)}"

  if [[ "$restart" -eq 1 ]]; then
    restart_daemon
  else
    echo "next: cc-connect daemon restart"
  fi
}

write_single() {
  local agent="" platform="" work_dir="$WORK_DIR_DEFAULT" name="" restart=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --agent) agent="${2:-}"; shift 2 ;;
      --platform) platform="${2:-}"; shift 2 ;;
      --work-dir) work_dir="${2:-}"; shift 2 ;;
      --name) name="${2:-}"; shift 2 ;;
      --restart) restart=1; shift ;;
      *) die "unknown single arg: $1" ;;
    esac
  done

  [[ -n "$agent" ]] || die "single requires --agent codex|claude"
  [[ -n "$platform" ]] || die "single requires --platform weixin|qq|both"
  agent_type "$agent" >/dev/null

  local platforms=()
  case "$platform" in
    weixin) platforms=(weixin) ;;
    qq) platforms=(qq) ;;
    both) platforms=(weixin qq) ;;
    *) die "--platform must be weixin, qq, or both" ;;
  esac

  [[ -n "$name" ]] || name="$(basename "$work_dir")-${agent}-${platform}"

  local blocks=()
  local platform_name
  for platform_name in "${platforms[@]}"; do
    blocks+=("$(platform_block "$platform_name")")
  done

  backup_config
  {
    write_header
    write_project "$name" "$agent" "$work_dir" "${blocks[@]}"
  } >"$CONFIG"
  format_config
  echo "wrote: $CONFIG"

  if [[ "$restart" -eq 1 ]]; then
    restart_daemon
  else
    echo "next: cc-connect daemon restart"
  fi
}

write_split() {
  local work_dir="$WORK_DIR_DEFAULT" restart=0

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --work-dir) work_dir="${2:-}"; shift 2 ;;
      --restart) restart=1; shift ;;
      *) die "unknown split arg: $1" ;;
    esac
  done

  backup_config
  local weixin_block qq_block
  weixin_block="$(platform_block weixin)"
  qq_block="$(platform_block qq)"
  {
    write_header
    write_project "$(basename "$work_dir")-codex-weixin" codex "$work_dir" "$weixin_block"
    write_project "$(basename "$work_dir")-claude-qq" claude "$work_dir" "$qq_block"
  } >"$CONFIG"
  format_config
  echo "wrote: $CONFIG"

  if [[ "$restart" -eq 1 ]]; then
    restart_daemon
  else
    echo "next: cc-connect daemon restart"
  fi
}

napcat() {
  local action="${1:-}"
  [[ -n "$action" ]] || die "napcat requires start|stop|restart|logs|status"

  case "$action" in
    start|stop|restart)
      docker --host "$DOCKER_HOST_DEFAULT" "$action" napcat
      ;;
    logs)
      docker --host "$DOCKER_HOST_DEFAULT" logs -f napcat
      ;;
    status)
      docker --host "$DOCKER_HOST_DEFAULT" ps -a --filter name=napcat
      ;;
    *)
      die "napcat requires start|stop|restart|logs|status"
      ;;
  esac
}

qq() {
  local action="${1:-}"
  [[ -n "$action" ]] || die "qq requires start|stop|restart|logs|status"

  case "$action" in
    start)
      docker --host "$DOCKER_HOST_DEFAULT" start napcat
      cc-connect daemon start
      ;;
    stop)
      cc-connect daemon stop
      docker --host "$DOCKER_HOST_DEFAULT" stop napcat
      ;;
    restart)
      docker --host "$DOCKER_HOST_DEFAULT" restart napcat
      cc-connect daemon restart
      ;;
    logs)
      cc-connect daemon logs -f
      ;;
    status)
      echo "cc-connect:"
      cc-connect daemon status
      echo
      echo "napcat:"
      docker --host "$DOCKER_HOST_DEFAULT" ps -a --filter name=napcat
      ;;
    *)
      die "qq requires start|stop|restart|logs|status"
      ;;
  esac
}

main() {
  local cmd="${1:-}"
  [[ -n "$cmd" ]] || { usage; exit 0; }
  shift || true

  case "$cmd" in
    help|-h|--help) usage ;;
    show) sed -n '1,240p' "$CONFIG" ;;
    status) cc-connect daemon status ;;
    restart) cc-connect daemon restart ;;
    logs) cc-connect daemon logs -f ;;
    single) write_single "$@" ;;
    split) write_split "$@" ;;
    model) set_codex_model "$@" ;;
    qq) qq "$@" ;;
    napcat) napcat "$@" ;;
    *) die "unknown command: $cmd" ;;
  esac
}

main "$@"
