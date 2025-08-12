#!/usr/bin/env bash
set -euo pipefail

# Initialize IPFS repo and start daemon if not running
if [ ! -d "$HOME/.ipfs" ]; then
  echo ">>> Initializing IPFS repo"
  ipfs init
fi

# Start ipfs daemon in background if not already running
if ! pgrep -x ipfs >/dev/null 2>&1; then
  echo ">>> Starting IPFS daemon"
  nohup ipfs daemon --init >/tmp/ipfs.log 2>&1 &
fi

# Show quick help
echo ">>> DALRN Codespace ready"
echo "Run: make run-gateway   # starts FastAPI gateway on 0.0.0.0:$GATEWAY_PORT"
echo "     make run-search    # starts Search on :8100"
echo "     make run-neg       # starts Negotiation on :8300"
echo "     make run-eps       # starts ε-ledger on :8400"
echo "     make run-all       # starts all (background)"
