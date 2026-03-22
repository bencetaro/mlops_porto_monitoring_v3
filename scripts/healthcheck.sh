#!/usr/bin/env bash
set -euo pipefail

NAMESPACE="porto"
OUTFILE=""
KUBECTL="minikube kubectl --"

usage() {
  cat <<'USAGE'
Usage: healthcheck.sh [-n namespace] [-o output_file] [-k kubectl_cmd]

Options:
  -n   Kubernetes namespace (default: porto)
  -o   Write output to a file (in addition to stdout)
  -k   kubectl command wrapper (default: "minikube kubectl --")
USAGE
}

while getopts ":n:o:k:h" opt; do
  case "$opt" in
    n) NAMESPACE="$OPTARG" ;;
    o) OUTFILE="$OPTARG" ;;
    k) KUBECTL="$OPTARG" ;;
    h) usage; exit 0 ;;
    \?) echo "Unknown option: -$OPTARG"; usage; exit 1 ;;
    :) echo "Option -$OPTARG requires an argument."; usage; exit 1 ;;
  esac
done

if ! command -v minikube >/dev/null 2>&1; then
  echo "minikube not found in PATH." >&2
  exit 1
fi

exec_out() {
  if [[ -n "$OUTFILE" ]]; then
    # Tee to file while keeping stdout
    "$@" | tee -a "$OUTFILE"
  else
    "$@"
  fi
}

header() {
  local title="$1"
  if [[ -n "$OUTFILE" ]]; then
    printf "\n===== %s =====\n" "$title" | tee -a "$OUTFILE"
  else
    printf "\n===== %s =====\n" "$title"
  fi
}

if [[ -n "$OUTFILE" ]]; then
  : > "$OUTFILE"
fi

header "Cluster Info"
exec_out $KUBECTL version

header "Namespace Overview ($NAMESPACE)"
exec_out $KUBECTL -n "$NAMESPACE" get pods -o wide
exec_out $KUBECTL -n "$NAMESPACE" get jobs
exec_out $KUBECTL -n "$NAMESPACE" get svc -o wide

header "Pods With Restarts"
exec_out $KUBECTL -n "$NAMESPACE" get pods --sort-by=.status.containerStatuses[0].restartCount
header "Recent Events (last 50)"
exec_out $KUBECTL -n "$NAMESPACE" get events --sort-by=.lastTimestamp | tail -n 50

header "Key App Logs (last 200 lines)"
LABELS=(
  "app=streamlit"
  "app=grafana"
  "app=mlflow-server"
  "app=inference-api"
  "app=prometheus"
  "app=node-exporter"
  "app=training"
)

for label in "${LABELS[@]}"; do
  header "Logs for $label"
  # Avoid failure if no pods match
  if $KUBECTL -n "$NAMESPACE" get pods -l "$label" -o name | grep -q .; then
    exec_out $KUBECTL -n "$NAMESPACE" logs -l "$label" --tail=200
  else
    if [[ -n "$OUTFILE" ]]; then
      echo "No pods found for label $label" | tee -a "$OUTFILE"
    else
      echo "No pods found for label $label"
    fi
  fi

done

header "Done"
