#!/usr/bin/env python
"""
Install PyTorch Geometric with proper dependencies
"""
import subprocess
import sys
import torch

def install_pytorch_geometric():
    """Install PyTorch Geometric and its dependencies"""

    # Get PyTorch version and CUDA info
    torch_version = torch.__version__
    cuda_version = torch.version.cuda

    print(f"PyTorch version: {torch_version}")
    print(f"CUDA version: {cuda_version if cuda_version else 'CPU only'}")

    # Install packages
    packages = []

    if cuda_version:
        # With CUDA support
        cuda_version_str = cuda_version.replace('.', '')[:3]  # e.g., "118" for CUDA 11.8
        index_url = f"https://data.pyg.org/whl/torch-{torch_version.split('+')[0]}+cu{cuda_version_str}.html"
        print(f"Installing with CUDA support from: {index_url}")
        packages = [
            f"torch-scatter",
            f"torch-sparse",
            f"torch-cluster",
            f"torch-spline-conv",
            f"torch-geometric"
        ]
        for package in packages:
            cmd = [sys.executable, "-m", "pip", "install", package, "-f", index_url]
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd)
    else:
        # CPU only
        print("Installing CPU-only version")
        packages = [
            "torch-scatter",
            "torch-sparse",
            "torch-cluster",
            "torch-spline-conv",
            "torch-geometric"
        ]
        for package in packages:
            cmd = [sys.executable, "-m", "pip", "install", package]
            print(f"Running: {' '.join(cmd)}")
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError:
                print(f"Warning: Failed to install {package}, trying without dependencies")
                cmd = [sys.executable, "-m", "pip", "install", package, "--no-deps"]
                subprocess.check_call(cmd)

    print("\nVerifying installation...")
    try:
        import torch_geometric
        print(f"✓ PyTorch Geometric {torch_geometric.__version__} installed successfully")
        return True
    except ImportError as e:
        print(f"✗ Installation failed: {e}")
        return False

if __name__ == "__main__":
    success = install_pytorch_geometric()
    sys.exit(0 if success else 1)