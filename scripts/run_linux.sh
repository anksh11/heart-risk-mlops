#!/usr/bin/env bash
# Interactive runner for the whole pipeline on a Linux host (Rocky/RHEL/Amazon Linux or Ubuntu/Debian).
#
#   ./scripts/run_linux.sh            # menu
#   ./scripts/run_linux.sh all        # every step in order, asking before installs
#   ./scripts/run_linux.sh <step>     # one step: prereqs python pipeline docker k8s access smoke load status stop
#
# Services are exposed on localhost with `kubectl port-forward` (works the same on every distro,
# unlike `minikube tunnel`, which on Linux serves each LoadBalancer on its own cluster IP).
set -uo pipefail
cd "$(dirname "$0")/.."

NS=heart-risk
IMAGE=${IMAGE:-heart-risk-api:1.0.0}
BIND=${BIND:-127.0.0.1}          # BIND=0.0.0.0 exposes the UIs on the host's network – only on a trusted network
RUN_DIR=.run
VENV=.venv
PY="$VENV/bin/python"
ME=${USER:-$(id -un)}
mkdir -p "$RUN_DIR"

c_info=$'\e[36m'; c_ok=$'\e[32m'; c_warn=$'\e[33m'; c_err=$'\e[31m'; c_off=$'\e[0m'
info() { echo "${c_info}==>${c_off} $*"; }
ok()   { echo "${c_ok}✔${c_off} $*"; }
warn() { echo "${c_warn}!${c_off} $*"; }
err()  { echo "${c_err}✘${c_off} $*" >&2; }
has()  { command -v "$1" >/dev/null 2>&1; }
ask()  { local a; read -r -p "${c_warn}?${c_off} $1 [y/N] " a; [[ $a =~ ^[Yy] ]]; }
run()  { echo "${c_info}\$${c_off} $*"; "$@"; }

arch() { case "$(uname -m)" in aarch64|arm64) echo arm64 ;; *) echo amd64 ;; esac; }
pkg_install() {
  if has dnf; then run sudo dnf install -y "$@"
  elif has yum; then run sudo yum install -y "$@"
  elif has apt-get; then run sudo apt-get update -qq && run sudo apt-get install -y "$@"
  else err "No dnf/yum/apt-get – install $* manually."; return 1; fi
}

# ---------------------------------------------------------------------------
step_prereqs() {
  info "Checking tools"
  for t in git curl make; do
    if has "$t"; then ok "$t"; else
      warn "$t missing"; ask "Install $t with the package manager?" && pkg_install "$t"
    fi
  done

  if has kubectl; then ok "kubectl $(kubectl version --client 2>/dev/null | head -1)"; else
    warn "kubectl missing"
    if ask "Download kubectl to /usr/local/bin?"; then
      local v; v=$(curl -Ls https://dl.k8s.io/release/stable.txt)
      run curl -Lo /tmp/kubectl "https://dl.k8s.io/release/$v/bin/linux/$(arch)/kubectl" &&
        run sudo install /tmp/kubectl /usr/local/bin/kubectl && rm -f /tmp/kubectl
    fi
  fi

  if has minikube; then ok "minikube $(minikube version --short 2>/dev/null)"; else
    warn "minikube missing"
    if ask "Download minikube to /usr/local/bin?"; then
      run curl -Lo /tmp/minikube "https://storage.googleapis.com/minikube/releases/latest/minikube-linux-$(arch)" &&
        run sudo install /tmp/minikube /usr/local/bin/minikube && rm -f /tmp/minikube
    fi
  fi

  if ! has docker; then
    warn "docker missing"
    ask "Install Docker Engine with the official get.docker.com script?" &&
      curl -fsSL https://get.docker.com | sudo sh && run sudo systemctl enable --now docker
  fi
  if has docker; then
    if docker version 2>/dev/null | grep -qi podman; then
      warn "'docker' is the Podman shim – use DRIVER=podman for minikube, or install Docker Engine."
    elif docker ps >/dev/null 2>&1; then
      ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null)"
    else
      warn "docker installed but not usable by $ME"
      if ask "Start the daemon and add $ME to the docker group?"; then
        run sudo systemctl enable --now docker
        run sudo usermod -aG docker "$ME"
        warn "Log out and back in (or run 'newgrp docker'), then re-run this script."
      fi
    fi
  fi
  echo; info "Resources: $(nproc) CPUs, $(free -m | awk '/Mem:/{print $2}') MB RAM (minikube wants >= 2 CPUs, ~4 GB)"
}

# ---------------------------------------------------------------------------
step_python() {
  [[ -n ${CONDA_DEFAULT_ENV:-} ]] && warn "Conda env '$CONDA_DEFAULT_ENV' is active – the venv below is used explicitly, so this is fine."
  if [[ ! -x $PY ]]; then
    local base=""
    for p in python3.11 python3.12 python3.10 python3; do has "$p" && base=$p && break; done
    [[ -z $base ]] && { err "No python3 found."; return 1; }
    run "$base" -m venv "$VENV" || return 1
  fi
  "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' || { err "$PY is older than 3.10."; return 1; }
  ok "venv: $("$PY" --version)"
  run "$PY" -m pip install -q --upgrade pip && run "$PY" -m pip install -q -r requirements.txt && ok "requirements installed"
}

# ---------------------------------------------------------------------------
step_pipeline() {
  [[ -x $PY ]] || { err "Run the 'python' step first."; return 1; }
  export PATH="$PWD/$VENV/bin:$PATH"
  for t in data eda train test lint; do
    info "make $t"; run make "$t" PY="$PY" || { err "make $t failed"; return 1; }
  done
  ok "Data, EDA, training, tests and lint all passed"
}

# ---------------------------------------------------------------------------
step_docker() {
  run docker build -t "$IMAGE" . || return 1
  docker rm -f heart-risk-api >/dev/null 2>&1
  run docker run -d --rm -p 8000:8000 --name heart-risk-api "$IMAGE" || return 1
  for _ in $(seq 30); do curl -sf localhost:8000/health >/dev/null && break; sleep 1; done
  run docker ps --filter name=heart-risk-api
  run curl -s localhost:8000/health; echo
  run curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @samples/patient_high_risk.json; echo
  run curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @samples/patient_low_risk.json; echo
  ok "Container serves /predict – screenshot this terminal for docker_build_run.png"
  if ask "Stop the container now?"; then run docker stop heart-risk-api; fi
}

# ---------------------------------------------------------------------------
step_k8s() {
  local mem cpus driver=${DRIVER:-docker}
  mem=$(free -m | awk '/Mem:/{m=$2-1536; print (m>4096?4096:(m<2200?2200:m))}')
  cpus=$(( $(nproc) >= 2 ? 2 : 1 ))
  if minikube status >/dev/null 2>&1; then ok "minikube already running"; else
    run minikube start --driver="$driver" --cpus="$cpus" --memory="$mem" --addons=ingress,metrics-server || return 1
  fi
  run minikube addons enable ingress >/dev/null
  run minikube addons enable metrics-server >/dev/null
  run kubectl config use-context minikube
  run bash scripts/deploy_k8s.sh || return 1
  ok "Deployed – next: the 'access' step"
}

# ---------------------------------------------------------------------------
pf_start() {  # name service port
  local pidf="$RUN_DIR/pf-$1.pid"
  [[ -f $pidf ]] && kill "$(cat "$pidf")" 2>/dev/null
  $(has setsid && echo setsid) nohup kubectl -n "$NS" port-forward --address "$BIND" "svc/$2" "$3:$3" >"$RUN_DIR/pf-$1.log" 2>&1 &
  echo $! >"$pidf"
}
step_access() {
  kubectl -n "$NS" get svc heart-risk-api >/dev/null 2>&1 || { err "Nothing deployed – run the 'k8s' step first."; return 1; }
  [[ $BIND != 127.0.0.1 ]] && warn "Binding to $BIND: Grafana (admin/admin) becomes reachable from the network."
  pf_start api heart-risk-api 8080
  pf_start prometheus prometheus 9090
  pf_start grafana grafana 3000
  sleep 3
  local host=localhost; [[ $BIND == 0.0.0.0 ]] && host=$(hostname -I | awk '{print $1}')
  for u in "8080/health" "9090/-/ready" "3000/api/health"; do
    if curl -sf -m 5 "localhost:$u" >/dev/null; then ok "localhost:$u"; else warn "localhost:$u not answering yet (see $RUN_DIR/pf-*.log)"; fi
  done
  cat <<EOF

  Open in a browser on this machine (port-forwards run in the background):
    API docs     http://$host:8080/docs
    Prometheus   http://$host:9090/targets
    Grafana      http://$host:3000   -> Dashboards -> Heart Risk API

  From your own PC instead, keep this SSH tunnel open and use the same localhost URLs:
    ssh -L 8080:localhost:8080 -L 9090:localhost:9090 -L 3000:localhost:3000 $ME@<server-address>

  Port-forwards stop with the 'stop' step, or if their pod restarts (just re-run 'access').
EOF
}

# ---------------------------------------------------------------------------
api_url() { curl -sf -m 3 localhost:8080/health >/dev/null && echo http://localhost:8080 && return
            local ip; ip=$(kubectl -n "$NS" get svc heart-risk-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null)
            [[ -n $ip ]] && echo "http://$ip:8080"; }
step_smoke() {
  local url; url=$(api_url) || true
  [[ -z $url ]] && { err "API not reachable – run the 'access' step first."; return 1; }
  run kubectl -n "$NS" get pods,svc,ingress,hpa
  echo; info "Ingress (nginx, Host: heart-risk.local)"
  run curl -s -m 5 -H "Host: heart-risk.local" "http://$(minikube ip)/health"; echo; echo
  run env BASE_URL="$url" bash scripts/smoke_test.sh && ok "Smoke test passed against $url – screenshot for k8s_smoke_test.png"
}

step_load() {
  local url secs; url=$(api_url) || true
  [[ -z $url ]] && { err "API not reachable – run the 'access' step first."; return 1; }
  local conc
  read -r -p "Seconds of traffic [180]: " secs; secs=${secs:-180}
  read -r -p "Parallel workers (8+ makes the HPA scale out) [10]: " conc; conc=${conc:-10}
  info "Watch scaling in another shell:  kubectl -n $NS get hpa -w"
  info "Scale-up needs ~1-2 min of sustained CPU above 70% of the 100m request."
  run "$PY" scripts/load_test.py --url "$url" --seconds "$secs" --rps 20 --concurrency "$conc"
  run kubectl -n "$NS" get hpa,pods
}

step_status() {
  has minikube && run minikube status
  run kubectl -n "$NS" get pods,svc,ingress,hpa
  for f in "$RUN_DIR"/pf-*.pid; do
    [[ -f $f ]] || continue
    if kill -0 "$(cat "$f")" 2>/dev/null; then ok "port-forward $(basename "$f" .pid) running"; else warn "port-forward $(basename "$f" .pid) stopped"; fi
  done
}

step_stop() {
  for f in "$RUN_DIR"/pf-*.pid; do [[ -f $f ]] && kill "$(cat "$f")" 2>/dev/null; rm -f "$f"; done
  ok "Port-forwards stopped"
  if docker ps -aq --filter name=^heart-risk-api$ 2>/dev/null | grep -q .; then
    docker rm -f heart-risk-api >/dev/null && ok "Local container removed"
  fi
  if ask "Delete the Kubernetes deployment (namespace $NS)?"; then run kubectl delete -k k8s --ignore-not-found; fi
  if ask "Stop minikube (keeps the cluster for next time)?"; then run minikube stop; fi
}

step_all() {
  for s in prereqs python pipeline docker k8s access smoke load; do
    echo; info "──── $s ────"
    "step_$s" || { err "Step '$s' failed – fix it and re-run: $0 $s"; return 1; }
  done
  echo; ok "All steps done. Take the screenshots, then run '$0 stop' when finished."
}

# ---------------------------------------------------------------------------
STEPS=(prereqs python pipeline docker k8s access smoke load status stop all)
DESC=("Check / install tools (git, make, kubectl, minikube, docker)"
      "Create .venv and install requirements"
      "ML pipeline: data, EDA, train, tests, lint"
      "Build the image, run it, curl /predict"
      "Start minikube (ingress + metrics-server) and deploy"
      "Expose API, Prometheus, Grafana on localhost"
      "Smoke test the deployed API (+ ingress)"
      "Generate traffic for Grafana / HPA"
      "Show cluster and port-forward status"
      "Stop port-forwards, optionally tear down"
      "Run everything from 'prereqs' to 'load'")

if [[ $# -gt 0 ]]; then
  declare -F "step_$1" >/dev/null || { err "Unknown step '$1'. Steps: ${STEPS[*]}"; exit 2; }
  "step_$1"; exit $?
fi

while true; do
  echo; echo "Heart-risk MLOps – Linux runner"
  for i in "${!STEPS[@]}"; do printf "  %2d) %-9s %s\n" $((i + 1)) "${STEPS[$i]}" "${DESC[$i]}"; done
  echo "   q) quit"
  read -r -p "Choose: " c
  [[ $c == q ]] && exit 0
  if [[ $c =~ ^[0-9]+$ ]] && (( c >= 1 && c <= ${#STEPS[@]} )); then
    "step_${STEPS[$((c - 1))]}" || err "Step failed (see output above)."
  else
    warn "Invalid choice"
  fi
done
