#!/usr/bin/env python3
"""
Import verification script to test actual dependencies.
This will reveal missing imports and fake implementations.
"""

import sys
import traceback
from pathlib import Path

def test_import(module_path, description):
    """Test importing a module and report results."""
    try:
        exec(f"import {module_path}")
        return f"[OK] {description}: SUCCESS"
    except ImportError as e:
        return f"[FAIL] {description}: MISSING - {str(e)}"
    except Exception as e:
        return f"[ERROR] {description}: ERROR - {str(e)}"

def test_service_imports():
    """Test critical service imports."""
    results = []

    # Core dependencies
    results.append(test_import("fastapi", "FastAPI framework"))
    results.append(test_import("pydantic", "Pydantic validation"))
    results.append(test_import("uvicorn", "ASGI server"))
    results.append(test_import("httpx", "HTTP client"))

    # Authentication
    results.append(test_import("jwt", "JWT library"))
    results.append(test_import("bcrypt", "Password hashing"))

    # Search service
    results.append(test_import("faiss", "FAISS vector search"))
    results.append(test_import("numpy", "NumPy arrays"))
    results.append(test_import("grpc", "gRPC framework"))

    # FHE service
    results.append(test_import("tenseal", "TenSEAL homomorphic encryption"))

    # Negotiation service
    results.append(test_import("nashpy", "Nash equilibrium computation"))

    # FL service
    results.append(test_import("flwr", "Flower federated learning"))
    results.append(test_import("opacus", "Opacus differential privacy"))

    # Agents/GNN
    results.append(test_import("torch", "PyTorch"))
    results.append(test_import("torch_geometric", "PyTorch Geometric"))
    results.append(test_import("networkx", "NetworkX graphs"))

    # Blockchain
    results.append(test_import("web3", "Web3 Ethereum"))
    results.append(test_import("eth_hash", "Ethereum hashing"))

    # Database
    results.append(test_import("psycopg2", "PostgreSQL driver"))
    results.append(test_import("sqlite3", "SQLite (built-in)"))

    # IPFS
    results.append(test_import("ipfshttpclient", "IPFS client"))

    return results

def test_service_modules():
    """Test importing actual service modules."""
    results = []

    # Add services to path
    sys.path.insert(0, str(Path(__file__).parent))

    service_modules = [
        ("services.gateway.app", "Gateway service"),
        ("services.auth.jwt_auth", "JWT authentication"),
        ("services.database.connection", "Database connection"),
        ("services.search.service", "Search service"),
        ("services.fhe.service", "FHE service"),
        ("services.negotiation.service", "Negotiation service"),
        ("services.fl.service", "FL service"),
        ("services.agents.gnn_predictor", "GNN predictor"),
        ("services.common.podp", "PoDP receipts"),
        ("services.chain.client", "Blockchain client"),
    ]

    for module, desc in service_modules:
        results.append(test_import(module, desc))

    return results

def main():
    print("=" * 60)
    print("DALRN IMPORT VERIFICATION REPORT")
    print("=" * 60)

    print("\n[1] Testing Core Dependencies:")
    print("-" * 40)
    for result in test_service_imports():
        print(result)

    print("\n[2] Testing Service Modules:")
    print("-" * 40)
    for result in test_service_modules():
        print(result)

    # Summary
    all_results = test_service_imports() + test_service_modules()
    success_count = sum(1 for r in all_results if r.startswith("[OK]"))
    total_count = len(all_results)

    print("\n" + "=" * 60)
    print(f"SUMMARY: {success_count}/{total_count} imports successful")
    print(f"Success Rate: {(success_count/total_count)*100:.1f}%")

    if success_count < total_count:
        print("\nMISSING DEPENDENCIES DETECTED!")
        print("The codebase has unmet dependencies.")

    return success_count == total_count

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)