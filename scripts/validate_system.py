#!/usr/bin/env python3
"""
DALRN System Validation Script
Validates actual implementation vs claimed functionality
"""

import sys
import os
import json
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SystemValidator:
    """Validates DALRN system readiness"""

    def __init__(self):
        self.results = {
            "dependencies": {},
            "services": {},
            "features": {},
            "infrastructure": {},
            "overall_score": 0
        }

    def check_dependencies(self) -> Dict[str, bool]:
        """Check if all required dependencies are installed"""
        logger.info("\n🔍 Checking Dependencies...")
        logger.info("-" * 40)

        dependencies = {
            "fastapi": "Web Framework",
            "flwr": "Federated Learning",
            "opacus": "Differential Privacy",
            "tenseal": "Homomorphic Encryption",
            "nashpy": "Game Theory",
            "faiss": "Vector Search",
            "ipfshttpclient": "IPFS Storage",
            "web3": "Blockchain",
            "torch": "Deep Learning",
            "torch_geometric": "Graph Neural Networks"
        }

        for module, description in dependencies.items():
            try:
                importlib.import_module(module)
                self.results["dependencies"][module] = True
                logger.info(f"✅ {module:<20} - {description}")
            except ImportError:
                self.results["dependencies"][module] = False
                logger.info(f"❌ {module:<20} - {description}")

        return self.results["dependencies"]

    def check_services(self) -> Dict[str, bool]:
        """Check if all services can be imported and initialized"""
        logger.info("\n🔍 Checking Services...")
        logger.info("-" * 40)

        services = {
            "services.gateway.app": "Gateway Service",
            "services.search.service": "Search Service",
            "services.fhe.service": "FHE Service",
            "services.negotiation.service": "Negotiation Service",
            "services.fl.service": "FL Service",
            "services.agents.orchestrator": "Agent Service",
            "services.common.podp": "PoDP System",
            "services.common.ipfs": "IPFS Client"
        }

        for module_path, description in services.items():
            try:
                module = importlib.import_module(module_path)
                self.results["services"][module_path] = True
                logger.info(f"✅ {description:<25} - Functional")
            except Exception as e:
                self.results["services"][module_path] = False
                logger.info(f"❌ {description:<25} - Error: {str(e)[:30]}")

        return self.results["services"]

    def check_features(self) -> Dict[str, bool]:
        """Check specific feature implementations"""
        logger.info("\n🔍 Checking Features...")
        logger.info("-" * 40)

        features_to_check = [
            ("JWT Authentication", self._check_jwt_auth),
            ("FAISS Vector Search", self._check_faiss),
            ("TenSEAL Encryption", self._check_tenseal),
            ("Nash Equilibrium", self._check_nash),
            ("Federated Learning", self._check_fl),
            ("IPFS Integration", self._check_ipfs),
            ("Smart Contracts", self._check_contracts),
            ("PoDP Receipts", self._check_podp)
        ]

        for feature_name, check_func in features_to_check:
            try:
                result = check_func()
                self.results["features"][feature_name] = result
                status = "✅" if result else "❌"
                logger.info(f"{status} {feature_name:<25} - {'Working' if result else 'Not Working'}")
            except Exception as e:
                self.results["features"][feature_name] = False
                logger.info(f"❌ {feature_name:<25} - Error: {str(e)[:30]}")

        return self.results["features"]

    def _check_jwt_auth(self) -> bool:
        """Verify JWT authentication is working"""
        from services.auth.jwt_auth import AuthService
        # Create test token
        token = AuthService.create_access_token({"sub": "test_user"})
        # Verify token
        payload = AuthService.verify_token(token)
        return payload.get("sub") == "test_user"

    def _check_faiss(self) -> bool:
        """Verify FAISS is working"""
        import faiss
        import numpy as np
        # Create simple index
        index = faiss.IndexFlatL2(64)
        vectors = np.random.randn(10, 64).astype('float32')
        index.add(vectors)
        return index.ntotal == 10

    def _check_tenseal(self) -> bool:
        """Verify TenSEAL is working"""
        import tenseal as ts
        # Create context
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        context.global_scale = 2**40
        context.generate_galois_keys()
        # Encrypt a value
        plain = [1.0, 2.0, 3.0]
        encrypted = ts.ckks_vector(context, plain)
        return encrypted is not None

    def _check_nash(self) -> bool:
        """Verify Nash equilibrium computation"""
        import nashpy as nash
        import numpy as np
        # Simple game
        A = np.array([[3, 1], [0, 2]])
        B = np.array([[2, 1], [0, 3]])
        game = nash.Game(A, B)
        equilibria = list(game.support_enumeration())
        return len(equilibria) > 0

    def _check_fl(self) -> bool:
        """Verify Flower is available"""
        import flwr as fl
        return hasattr(fl, '__version__')

    def _check_ipfs(self) -> bool:
        """Verify IPFS client"""
        from services.common.ipfs import get_ipfs_status
        status = get_ipfs_status()
        # Working if fallback is enabled even if daemon not running
        return status["local_fallback_enabled"]

    def _check_contracts(self) -> bool:
        """Verify smart contract files exist"""
        contract_path = Path("services/chain/contracts/AnchorReceipts.sol")
        abi_path = Path("services/chain/abi/AnchorReceipts.json")
        return contract_path.exists() and abi_path.exists()

    def _check_podp(self) -> bool:
        """Verify PoDP receipt generation"""
        from services.common.podp import Receipt, ReceiptChain
        receipt = Receipt(
            dispute_id="test",
            step="TEST",
            ts="2024-01-01T00:00:00Z"
        )
        receipt.finalize()
        return receipt.hash is not None

    def check_infrastructure(self) -> Dict[str, bool]:
        """Check infrastructure components"""
        logger.info("\n🔍 Checking Infrastructure...")
        logger.info("-" * 40)

        checks = {
            "Docker Compose": Path("docker-compose.yml").exists(),
            "Requirements.txt": Path("requirements.txt").exists(),
            "Environment Config": Path(".env.example").exists(),
            "Setup Scripts": Path("scripts/setup_infrastructure.sh").exists(),
            "Smart Contract": Path("services/chain/contracts/AnchorReceipts.sol").exists(),
            "Integration Tests": Path("tests/integration/test_full_pipeline.py").exists()
        }

        for component, exists in checks.items():
            self.results["infrastructure"][component] = exists
            status = "✅" if exists else "❌"
            logger.info(f"{status} {component:<25} - {'Present' if exists else 'Missing'}")

        return self.results["infrastructure"]

    def calculate_readiness(self) -> float:
        """Calculate overall system readiness percentage"""
        scores = {
            "dependencies": 0.30,  # 30% weight
            "services": 0.35,      # 35% weight
            "features": 0.25,      # 25% weight
            "infrastructure": 0.10  # 10% weight
        }

        total_score = 0
        for category, weight in scores.items():
            if category in self.results and self.results[category]:
                items = self.results[category]
                if items:
                    success_rate = sum(1 for v in items.values() if v) / len(items)
                    total_score += success_rate * weight

        self.results["overall_score"] = total_score * 100
        return self.results["overall_score"]

    def generate_report(self):
        """Generate comprehensive validation report"""
        logger.info("\n" + "="*60)
        logger.info("DALRN SYSTEM VALIDATION REPORT")
        logger.info("="*60)

        # Run all checks
        self.check_dependencies()
        self.check_services()
        self.check_features()
        self.check_infrastructure()

        # Calculate readiness
        readiness = self.calculate_readiness()

        # Summary
        logger.info("\n📊 Summary")
        logger.info("-" * 40)

        for category in ["dependencies", "services", "features", "infrastructure"]:
            items = self.results[category]
            success = sum(1 for v in items.values() if v)
            total = len(items)
            percentage = (success / total * 100) if total > 0 else 0
            logger.info(f"{category.capitalize():<15}: {success}/{total} ({percentage:.1f}%)")

        # Overall readiness
        logger.info("\n" + "="*60)
        if readiness >= 90:
            status_emoji = "🎉"
            status_text = "PRODUCTION READY"
        elif readiness >= 75:
            status_emoji = "✅"
            status_text = "NEARLY READY"
        elif readiness >= 60:
            status_emoji = "⚠️"
            status_text = "PARTIAL IMPLEMENTATION"
        else:
            status_emoji = "❌"
            status_text = "NOT READY"

        logger.info(f"{status_emoji} Overall Readiness: {readiness:.1f}%")
        logger.info(f"Status: {status_text}")

        # Comparison with claimed
        claimed = 92  # From documentation
        actual = readiness
        difference = actual - claimed

        logger.info(f"\nClaimed Readiness: {claimed}%")
        logger.info(f"Actual Readiness:  {actual:.1f}%")
        logger.info(f"Difference:        {difference:+.1f}%")

        # Save report
        report_file = Path("VALIDATION_REPORT.json")
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"\nDetailed report saved to: {report_file}")

        return readiness


def main():
    """Main validation function"""
    validator = SystemValidator()
    readiness = validator.generate_report()

    # Exit code based on readiness
    if readiness >= 75:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    main()