#!/usr/bin/env bash
set -euo pipefail

echo ">>> Updating apt and installing basics"
sudo apt-get update -y
sudo apt-get install -y curl build-essential pkg-config ca-certificates git jq

# Foundry (forge/cast)
if ! command -v forge >/dev/null 2>&1; then
  echo ">>> Installing Foundry (forge)"
  curl -L https://foundry.paradigm.xyz | bash
  ~/.foundry/bin/foundryup
fi

# IPFS (kubo) install
if ! command -v ipfs >/dev/null 2>&1; then
  echo ">>> Installing IPFS (kubo)"
  KVER="v0.29.0"
  ARCH="linux-amd64"
  curl -L -o /tmp/kubo.tar.gz https://dist.ipfs.tech/kubo/${KVER}/kubo_${KVER}_${ARCH}.tar.gz
  tar -xzf /tmp/kubo.tar.gz -C /tmp
  sudo bash /tmp/kubo/install.sh
  rm -rf /tmp/kubo /tmp/kubo.tar.gz || true
fi

# Python deps
if [ -f requirements.txt ]; then
  echo ">>> Installing Python requirements"
  python -m pip install --upgrade pip
  pip install -r requirements.txt
fi

echo ">>> Setup complete"
