# DALRN Devcontainer

This devcontainer sets up Python 3.11, Node 20, Foundry (forge), and IPFS (kubo). 
Ports are forwarded so you can open the Gateway URL in your browser.

## First run
- Codespaces will execute `.devcontainer/setup.sh` to install tooling, then `.devcontainer/postStart.sh` to start IPFS.
- Open the **Ports** tab to find the Gateway (8000) URL.

## Start services
Use the Makefile (expected in repo root):
```
make run-all
# or individually:
make run-gateway
make run-search
make run-neg
make run-eps
```
