#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

env_file="$HOME/.config/traffic-analyzer/api.env"
if [[ ! -f "$env_file" ]]; then
	install -Dm600 deploy/api.env.example "$env_file"
	echo "Created $env_file from the example; edit it and re-run." >&2
	exit 1
fi

git pull --ff-only
podman build --format docker -t localhost/traffic-analyzer:latest .

install -Dm644 deploy/traffic-analyzer.container "$HOME/.config/containers/systemd/traffic-analyzer.container"
systemctl --user daemon-reload
systemctl --user restart traffic-analyzer

for _ in $(seq 30); do
	if curl -fs http://127.0.0.1:8000/api/health >/dev/null; then
		echo "API healthy"
		podman image prune -f >/dev/null
		exit 0
	fi
	sleep 2
done
echo "API did not become healthy; see: journalctl --user -u traffic-analyzer -n 100" >&2
exit 1
