#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_DIR="$ROOT_DIR/.runtime"

QDRANT_CONTAINER_DEFAULT="${QDRANT_CONTAINER:-chatbot-qdrant}"
PURGE_QDRANT="${PURGE_QDRANT:-false}"

read_runtime_file() {
  local file_path="$1"
  if [[ -f "$file_path" ]]; then
    cat "$file_path"
  fi
}

stop_pid_file_process() {
  local label="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    return
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"

  if [[ -z "$pid" ]]; then
    rm -f "$pid_file"
    return
  fi

  if kill -0 "$pid" 2>/dev/null; then
    echo "[INFO] Stopping $label (PID $pid)"
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      echo "[WARN] Force killing $label (PID $pid)"
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  rm -f "$pid_file"
}

stop_project_rasa_processes() {
  local matches
  matches="$(ps -eo pid=,args= | grep -E "rasa (run actions|shell|run)" | grep "$ROOT_DIR" | grep -v grep || true)"

  if [[ -z "$matches" ]]; then
    return
  fi

  echo "[INFO] Stopping leftover Rasa processes"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local pid
    pid="$(echo "$line" | awk '{print $1}')"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done <<< "$matches"

  sleep 1

  matches="$(ps -eo pid=,args= | grep -E "rasa (run actions|shell|run)" | grep "$ROOT_DIR" | grep -v grep || true)"
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    local pid
    pid="$(echo "$line" | awk '{print $1}')"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
  done <<< "$matches"
}

stop_qdrant_container() {
  local qdrant_container
  qdrant_container="$(read_runtime_file "$RUNTIME_DIR/qdrant_container")"
  qdrant_container="${qdrant_container:-$QDRANT_CONTAINER_DEFAULT}"

  if docker ps --format '{{.Names}}' | grep -q "^${qdrant_container}$"; then
    echo "[INFO] Stopping Qdrant container: $qdrant_container"
    docker stop "$qdrant_container" >/dev/null || true
  else
    echo "[INFO] Qdrant container is not running: $qdrant_container"
  fi

  if [[ "$PURGE_QDRANT" == "true" ]] && docker ps -a --format '{{.Names}}' | grep -q "^${qdrant_container}$"; then
    echo "[INFO] Removing Qdrant container: $qdrant_container"
    docker rm "$qdrant_container" >/dev/null || true
  fi
}

cleanup_runtime_files() {
  rm -f "$RUNTIME_DIR/action_server.pid"
  rm -f "$RUNTIME_DIR/qdrant_container"
  rmdir "$RUNTIME_DIR" 2>/dev/null || true
}

main() {
  echo "[INFO] Stopping Chatbot stack"

  stop_pid_file_process "action server" "$RUNTIME_DIR/action_server.pid"
  stop_project_rasa_processes
  stop_qdrant_container
  cleanup_runtime_files

  echo "[INFO] Cleanup completed"
}

main "$@"
