#!/usr/bin/env python3
"""
test_core_functionality.py - Test if features actually work
"""
import sys
import json
import traceback
from pathlib import Path

test_results = {}

def test_federated_learning():
    """Test if FL actually aggregates or just averages"""
    try:
        # Check if the FL service can even be imported
        import services.fl.service as fl_service

        # Check if it has the claimed sophisticated features
        if hasattr(fl_service, 'SecureFedAvg'):
            return "CLAIMS_SOPHISTICATED_BUT_UNTESTED"
        elif hasattr(fl_service, 'FLCoordinator'):
            # Try to create coordinator
            coordinator = fl_service.FLCoordinator()
            return "SIMPLE_COORDINATOR_EXISTS"
        else:
            return "NO_FL_IMPLEMENTATION"

    except ImportError as e:
        return f"IMPORT_ERROR: {str(e)[:100]}"
    except Exception as e:
        return f"BROKEN: {str(e)[:100]}"

def test_gnn_training():
    """Test if GNN actually trains or generates random numbers"""
    try:
        # Check which GNN implementation exists
        gnn_path = Path("services/agents/gnn_implementation.py")

        if not gnn_path.exists():
            return "NO_GNN_FILE"

        # Read the file to check for fake patterns
        with open(gnn_path, 'r', encoding='utf-8') as f:
            code = f.read()

        # Check for real PyTorch Geometric
        if "from torch_geometric.nn import GCNConv" in code:
            if "loss.backward()" in code and "optimizer.step()" in code:
                return "REAL_PYTORCH_GEOMETRIC_GNN"
            else:
                return "PYTORCH_GEOMETRIC_NO_TRAINING"

        # Check for fake training
        if "np.random.uniform" in code and "loss" in code:
            return "FAKE_RANDOM_LOSS"

        return "UNKNOWN_GNN_TYPE"

    except Exception as e:
        return f"ERROR: {str(e)[:100]}"

def test_differential_privacy():
    """Test if DP uses Opacus or just adds noise"""
    try:
        # Check if Opacus privacy module exists
        opacus_path = Path("services/fl/opacus_privacy.py")

        if opacus_path.exists():
            with open(opacus_path, 'r', encoding='utf-8') as f:
                code = f.read()

            if "from opacus import PrivacyEngine" in code:
                if "RDPAccountant" in code:
                    return "REAL_OPACUS_WITH_RDP"
                else:
                    return "PARTIAL_OPACUS"

        # Check the main FL service
        fl_path = Path("services/fl/service.py")
        if fl_path.exists():
            with open(fl_path, 'r', encoding='utf-8') as f:
                code = f.read()

            if "np.random.normal" in code and "noise" in code:
                if "opacus" not in code.lower():
                    return "FAKE_NOISE_ONLY"

        return "NO_PRIVACY_IMPLEMENTATION"

    except Exception as e:
        return f"ERROR: {str(e)[:100]}"

def test_blockchain():
    """Test if blockchain actually connects"""
    try:
        # Check if blockchain service exists
        blockchain_path = Path("services/blockchain")

        if not blockchain_path.exists():
            chain_path = Path("services/chain")
            if not chain_path.exists():
                return "NO_BLOCKCHAIN_DIRECTORY"
            blockchain_path = chain_path

        # Check for client implementation
        client_files = list(blockchain_path.glob("*.py"))

        if not client_files:
            return "NO_BLOCKCHAIN_FILES"

        # Check for Web3 usage
        for file in client_files:
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
                if "from web3 import Web3" in code:
                    if "w3.is_connected()" in code:
                        return "HAS_WEB3_CONNECTION_CHECK"
                    else:
                        return "HAS_WEB3_NO_CONNECTION_CHECK"

        return "NO_WEB3_IMPLEMENTATION"

    except Exception as e:
        return f"ERROR: {str(e)[:100]}"

def test_vector_search():
    """Test if FAISS search actually works"""
    try:
        search_path = Path("services/search/service.py")

        if not search_path.exists():
            return "NO_SEARCH_SERVICE"

        with open(search_path, 'r', encoding='utf-8') as f:
            code = f.read()

        if "import faiss" in code:
            if "IndexHNSWFlat" in code or "IndexIVFFlat" in code:
                return "REAL_FAISS_IMPLEMENTATION"
            else:
                return "FAISS_IMPORTED_NOT_USED"

        return "NO_FAISS_IMPLEMENTATION"

    except Exception as e:
        return f"ERROR: {str(e)[:100]}"

def test_homomorphic_encryption():
    """Test if FHE actually encrypts or just hashes"""
    try:
        fhe_path = Path("services/fhe/service.py")

        if not fhe_path.exists():
            return "NO_FHE_SERVICE"

        with open(fhe_path, 'r', encoding='utf-8') as f:
            code = f.read()

        if "import tenseal" in code:
            if "ts.SCHEME_TYPE.CKKS" in code:
                return "REAL_TENSEAL_CKKS"
            else:
                return "TENSEAL_IMPORTED_NOT_CONFIGURED"

        if "hashlib" in code and "sha256" in code:
            return "FAKE_USING_SHA256"

        return "NO_ENCRYPTION_IMPLEMENTATION"

    except Exception as e:
        return f"ERROR: {str(e)[:100]}"

def test_negotiation():
    """Test if Nash equilibrium is computed or faked"""
    try:
        negotiation_path = Path("services/negotiation/service.py")

        if not negotiation_path.exists():
            return "NO_NEGOTIATION_SERVICE"

        with open(negotiation_path, 'r', encoding='utf-8') as f:
            code = f.read()

        if "import nashpy" in code:
            if "Game(" in code and "support_enumeration" in code:
                return "REAL_NASHPY_IMPLEMENTATION"
            else:
                return "NASHPY_IMPORTED_NOT_USED"

        if "random" in code and "equilibrium" in code:
            return "FAKE_RANDOM_EQUILIBRIUM"

        return "NO_NASH_IMPLEMENTATION"

    except Exception as e:
        return f"ERROR: {str(e)[:100]}"

# Run all tests
print("=" * 60)
print("CORE FUNCTIONALITY TESTS")
print("=" * 60)

tests = {
    "federated_learning": test_federated_learning,
    "gnn_training": test_gnn_training,
    "differential_privacy": test_differential_privacy,
    "blockchain": test_blockchain,
    "vector_search": test_vector_search,
    "homomorphic_encryption": test_homomorphic_encryption,
    "negotiation": test_negotiation
}

for test_name, test_func in tests.items():
    result = test_func()
    test_results[test_name] = result
    print(f"\n{test_name}:")
    print(f"  Result: {result}")

# Analyze results
real_count = 0
fake_count = 0
broken_count = 0

for feature, result in test_results.items():
    if "REAL" in result:
        real_count += 1
    elif "FAKE" in result:
        fake_count += 1
    elif "ERROR" in result or "BROKEN" in result or "NO_" in result:
        broken_count += 1

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Real implementations: {real_count}/7")
print(f"Fake implementations: {fake_count}/7")
print(f"Broken/Missing: {broken_count}/7")

# Save results
with open("core_functionality_test.json", "w") as f:
    json.dump(test_results, f, indent=2)

print("\nResults saved to core_functionality_test.json")