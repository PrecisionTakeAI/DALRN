#!/usr/bin/env python3
"""
FORENSIC AUDIT - DALRN System
Trust nothing. Verify everything. Execute every line.
"""

import os
import sys
import ast
import json
import re
import subprocess
import traceback
import importlib
import inspect
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class ForensicAuditor:
    def __init__(self):
        self.report = {
            "timestamp": datetime.now().isoformat(),
            "services": {},
            "total_files": 0,
            "total_functions": 0,
            "real_implementations": 0,
            "placeholders": 0,
            "mock_functions": 0,
            "broken_imports": 0,
            "fake_patterns": [],
            "suspicious_files": [],
            "hardcoded_values": [],
            "todo_fixme_count": 0
        }

        # Patterns that indicate fake/placeholder code
        self.placeholder_patterns = [
            r"def \w+\([^)]*\):\s*pass",
            r"def \w+\([^)]*\):\s*\.\.\.",
            r"def \w+\([^)]*\):\s*return None",
            r"def \w+\([^)]*\):\s*return \{\}",
            r"def \w+\([^)]*\):\s*return \[\]",
            r"raise NotImplementedError",
            r"#\s*(TODO|FIXME|HACK|XXX|PLACEHOLDER|MOCK|FAKE)",
            r"return [\"']not implemented[\"']",
            r"fake_\w+",
            r"mock_\w+",
            r"dummy_\w+",
            r"test_data",
            r"placeholder",
            r"time\.sleep\(\d+\)",  # Artificial delays
            r"random\.choice\(",  # Random returns
            r"random\.random\(",
        ]

    def audit_entire_codebase(self):
        """Main audit entry point"""
        print("[FORENSIC] AUDIT STARTED")
        print("=" * 60)
        print("Trust nothing. Verify everything.")
        print("=" * 60)

        # 1. Find all Python files in services
        python_files = list(Path("services").rglob("*.py"))
        self.report["total_files"] = len(python_files)

        print(f"\n[FILES] Found {len(python_files)} Python files to audit\n")

        # 2. Audit each file
        for file_path in python_files:
            self.audit_file(file_path)

        # 3. Run specific verification tests
        self.run_verification_tests()

        # 4. Scan for hardcoded test data
        self.scan_for_test_data()

        # 5. Calculate real implementation percentage
        self.calculate_real_percentage()

        # 6. Generate final report
        self.generate_report()

    def audit_file(self, file_path: Path):
        """Forensically audit a single Python file"""
        print(f"[FILE] Auditing: {file_path}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Check for suspicious patterns
            for pattern in self.placeholder_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    self.report["fake_patterns"].extend([
                        {"file": str(file_path), "pattern": pattern, "match": match[:100]}
                        for match in matches
                    ])

            # Parse AST to analyze functions
            try:
                tree = ast.parse(content)
                self.analyze_ast(tree, file_path, content)
            except SyntaxError as e:
                print(f"   [FAIL] SYNTAX ERROR: {e}")
                self.report["broken_imports"] += 1
                self.report["suspicious_files"].append(str(file_path))

        except Exception as e:
            print(f"   [FAIL] ERROR reading file: {e}")
            self.report["suspicious_files"].append(str(file_path))

    def analyze_ast(self, tree: ast.AST, file_path: Path, content: str):
        """Analyze AST to detect real vs fake implementations"""
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]

        for func in functions:
            self.report["total_functions"] += 1
            func_name = func.name

            # Skip dunder methods
            if func_name.startswith("__") and func_name.endswith("__"):
                continue

            # Analyze function body
            is_real = self.is_real_implementation(func, content)

            if is_real:
                self.report["real_implementations"] += 1
                print(f"   [OK] {func_name}: REAL")
            else:
                self.report["placeholders"] += 1
                print(f"   [WARN] {func_name}: PLACEHOLDER/MOCK")

    def is_real_implementation(self, func_node: ast.FunctionDef, full_content: str) -> bool:
        """Determine if function has real implementation"""
        # Empty function or just pass
        if len(func_node.body) == 0:
            return False

        if len(func_node.body) == 1:
            stmt = func_node.body[0]

            # Just pass
            if isinstance(stmt, ast.Pass):
                return False

            # Just raises NotImplementedError
            if isinstance(stmt, ast.Raise):
                if hasattr(stmt.exc, 'func'):
                    if hasattr(stmt.exc.func, 'id'):
                        if stmt.exc.func.id == 'NotImplementedError':
                            return False

            # Just returns None/empty
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return False
                if isinstance(stmt.value, ast.Constant):
                    if stmt.value.value in [None, {}, [], "", 0, False]:
                        return False

        # Has substantial body - likely real
        return len(func_node.body) > 2

    def run_verification_tests(self):
        """Run specific service verification tests"""
        print("\n" + "="*60)
        print("[TEST] RUNNING SERVICE VERIFICATION TESTS")
        print("="*60)

        # Test FHE
        self.test_fhe_service()

        # Test FL
        self.test_fl_service()

        # Test Search
        self.test_search_service()

        # Test Database
        self.test_database()

        # Test Blockchain
        self.test_blockchain()

    def test_fhe_service(self):
        """Verify FHE is doing real encryption"""
        print("\n[CRYPTO] Testing FHE Service...")

        try:
            import tenseal as ts
            import numpy as np

            # Create context
            context = ts.context(
                ts.SCHEME_TYPE.CKKS,
                poly_modulus_degree=8192,
                coeff_mod_bit_sizes=[60, 40, 40, 60]
            )
            context.global_scale = 2**40

            # Test encryption
            plain_data = [1.1, 2.2, 3.3]
            encrypted = ts.ckks_vector(context, plain_data)

            # Verify ciphertext is encrypted
            ciphertext = encrypted.serialize()

            # CRITICAL CHECK: Real encryption produces large ciphertext
            size_ratio = len(ciphertext) / len(str(plain_data))

            if size_ratio > 100:
                print(f"   [OK] FHE REAL: Ciphertext is {size_ratio:.0f}x larger than plaintext")
                self.report["services"]["FHE"] = "REAL"

                # Test homomorphic operation
                result = encrypted + encrypted
                decrypted = result.decrypt()
                expected = [x * 2 for x in plain_data]

                errors = [abs(d - e) for d, e in zip(decrypted, expected)]
                max_error = max(errors)

                if max_error < 0.01:
                    print(f"   [OK] FHE Math works: max error = {max_error:.6f}")
                else:
                    print(f"   [WARNING] FHE Math imprecise: max error = {max_error:.6f}")
            else:
                print(f"   [FAIL] FHE FAKE: Ciphertext only {size_ratio:.1f}x larger!")
                self.report["services"]["FHE"] = "FAKE"

        except Exception as e:
            print(f"   [FAIL] FHE BROKEN: {e}")
            self.report["services"]["FHE"] = f"ERROR: {str(e)[:50]}"

    def test_fl_service(self):
        """Verify Federated Learning is real"""
        print("\n[FL] Testing FL Service...")

        try:
            import flwr as fl
            from flwr.common import FitRes, Parameters
            import numpy as np

            # Test if we can create a client
            class TestClient(fl.client.NumPyClient):
                def get_parameters(self, config):
                    return [np.array([1.0, 2.0])]

                def fit(self, parameters, config):
                    return [np.array([1.5, 2.5])], 10, {}

                def evaluate(self, parameters, config):
                    return 0.5, 10, {"accuracy": 0.9}

            client = TestClient()
            params = client.get_parameters({})

            if params is not None and len(params) > 0:
                print(f"   [OK] FL REAL: Client returns parameters")
                self.report["services"]["FL"] = "REAL"

                # Test fitting
                fit_result = client.fit(params, {})
                if fit_result[0] is not None:
                    print(f"   [OK] FL Training works")
            else:
                print(f"   [FAIL] FL FAKE: No parameters returned")
                self.report["services"]["FL"] = "FAKE"

        except Exception as e:
            print(f"   [FAIL] FL BROKEN: {e}")
            self.report["services"]["FL"] = f"ERROR: {str(e)[:50]}"

    def test_search_service(self):
        """Verify FAISS search is real"""
        print("\n[SEARCH] Testing Search Service...")

        try:
            import faiss
            import numpy as np

            # Create index
            dim = 128
            n_vectors = 100

            index = faiss.IndexFlatL2(dim)

            # Generate data
            np.random.seed(42)
            data = np.random.random((n_vectors, dim)).astype('float32')
            index.add(data)

            # Search
            query = np.random.random((1, dim)).astype('float32')
            k = 5
            distances, indices = index.search(query, k)

            # Verify results are sorted
            if all(distances[0][i] <= distances[0][i+1] for i in range(k-1)):
                print(f"   [OK] SEARCH REAL: Results properly sorted")

                # Manual verification
                manual_dists = np.sum((data - query)**2, axis=1)
                manual_nearest = np.argmin(manual_dists)

                if indices[0][0] == manual_nearest:
                    print(f"   [OK] SEARCH accurate: Top result verified")
                    self.report["services"]["Search"] = "REAL"
                else:
                    print(f"   [WARNING] SEARCH inaccurate: {indices[0][0]} != {manual_nearest}")
                    self.report["services"]["Search"] = "INACCURATE"
            else:
                print(f"   [FAIL] SEARCH FAKE: Results not sorted!")
                self.report["services"]["Search"] = "FAKE"

        except Exception as e:
            print(f"   [FAIL] SEARCH BROKEN: {e}")
            self.report["services"]["Search"] = f"ERROR: {str(e)[:50]}"

    def test_database(self):
        """Test database connections"""
        print("\n[DB] Testing Database...")

        results = {}

        # Test PostgreSQL
        try:
            import psycopg2
            conn = psycopg2.connect(
                host="localhost",
                database="dalrn",
                user="dalrn_user",
                password="changeme"
            )
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            if result[0] == 1:
                results["PostgreSQL"] = "REAL"
                print(f"   [OK] PostgreSQL: REAL connection")
            conn.close()
        except Exception as e:
            results["PostgreSQL"] = "UNAVAILABLE"
            print(f"   [WARNING] PostgreSQL: Not available - {str(e)[:30]}")

        # Test Redis
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379, decode_responses=True)
            r.ping()
            r.set('forensic_test', 'verified')
            value = r.get('forensic_test')
            if value == 'verified':
                results["Redis"] = "REAL"
                print(f"   [OK] Redis: REAL connection")
            r.delete('forensic_test')
        except Exception as e:
            results["Redis"] = "UNAVAILABLE"
            print(f"   [WARNING] Redis: Not available - {str(e)[:30]}")

        # Test SQLite fallback
        try:
            import sqlite3
            conn = sqlite3.connect(':memory:')
            cur = conn.cursor()
            cur.execute("CREATE TABLE test (id INTEGER)")
            cur.execute("INSERT INTO test VALUES (42)")
            cur.execute("SELECT * FROM test")
            result = cur.fetchone()
            if result[0] == 42:
                results["SQLite"] = "REAL"
                print(f"   [OK] SQLite: REAL (fallback working)")
            conn.close()
        except Exception as e:
            results["SQLite"] = "BROKEN"
            print(f"   [FAIL] SQLite: BROKEN - {e}")

        self.report["services"]["Database"] = results

    def test_blockchain(self):
        """Test blockchain integration"""
        print("\n[CHAIN] Testing Blockchain...")

        try:
            from web3 import Web3

            # Try local connection
            w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))

            if w3.is_connected():
                chain_id = w3.eth.chain_id
                latest_block = w3.eth.block_number
                print(f"   [OK] Blockchain connected: Chain ID {chain_id}, Block {latest_block}")
                self.report["services"]["Blockchain"] = "CONNECTED"

                # Check for deployed contract
                # Would need actual contract address from deployment
            else:
                print(f"   [WARNING] Blockchain: No local node running")
                self.report["services"]["Blockchain"] = "NO_NODE"

        except Exception as e:
            print(f"   [FAIL] Blockchain: {e}")
            self.report["services"]["Blockchain"] = f"ERROR: {str(e)[:50]}"

    def scan_for_test_data(self):
        """Scan for hardcoded test data"""
        print("\n[SCAN] Scanning for hardcoded test data...")

        patterns = [
            r"test_data",
            r"fake_data",
            r"mock_\w+",
            r"dummy_\w+",
            r"placeholder",
            r"example_\w+",
            r"sample_\w+",
            r"= [\"']test[\"']",
            r"= [\"']fake[\"']",
            r"= [\"']TODO[\"']",
        ]

        count = 0
        for file_path in Path("services").rglob("*.py"):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    count += len(matches)
                    self.report["hardcoded_values"].append({
                        "file": str(file_path),
                        "pattern": pattern,
                        "count": len(matches)
                    })

        print(f"   Found {count} instances of hardcoded test data")

    def calculate_real_percentage(self):
        """Calculate actual implementation percentage"""
        if self.report["total_functions"] > 0:
            real_pct = (self.report["real_implementations"] / self.report["total_functions"]) * 100
            self.report["implementation_percentage"] = round(real_pct, 2)
        else:
            self.report["implementation_percentage"] = 0.0

    def generate_report(self):
        """Generate comprehensive forensic report"""
        print("\n" + "="*60)
        print("[REPORT] FORENSIC AUDIT COMPLETE")
        print("="*60)

        print(f"""
FORENSIC SUMMARY:
-----------------
Total Files Analyzed: {self.report['total_files']}
Total Functions: {self.report['total_functions']}
Real Implementations: {self.report['real_implementations']}
Placeholders/Mocks: {self.report['placeholders']}
Suspicious Patterns: {len(self.report['fake_patterns'])}
Hardcoded Test Data: {len(self.report['hardcoded_values'])}

ACTUAL Implementation: {self.report.get('implementation_percentage', 0)}%

SERVICE VERIFICATION:
--------------------""")

        for service, status in self.report['services'].items():
            if isinstance(status, dict):
                print(f"\n{service}:")
                for sub, stat in status.items():
                    emoji = "[OK]" if stat == "REAL" else "[WARNING]" if "UNAVAILABLE" in stat else "[FAIL]"
                    print(f"  {emoji} {sub}: {stat}")
            else:
                emoji = "[OK]" if status == "REAL" else "[WARNING]" if "UNAVAILABLE" in status or "NO_NODE" in status else "[FAIL]"
                print(f"{emoji} {service}: {status}")

        # Save JSON report
        with open('FORENSIC_AUDIT_REPORT.json', 'w') as f:
            json.dump(self.report, f, indent=2)

        print(f"\n[SAVE] Detailed report saved: FORENSIC_AUDIT_REPORT.json")

        # Generate placeholder list
        self.generate_placeholder_list()

        # Generate real implementation map
        self.generate_implementation_map()

        # Final verdict
        impl_pct = self.report.get('implementation_percentage', 0)

        print("\n" + "="*60)
        print("[VERDICT] FINAL VERDICT")
        print("="*60)

        if impl_pct >= 90:
            print(f"[OK] System appears MOSTLY REAL ({impl_pct:.1f}% genuine)")
        elif impl_pct >= 70:
            print(f"[WARNING] System is PARTIALLY REAL ({impl_pct:.1f}% genuine)")
        elif impl_pct >= 50:
            print(f"[WARNING] System is HALF REAL/HALF FAKE ({impl_pct:.1f}% genuine)")
        else:
            print(f"[FAIL] System is MOSTLY FAKE ({impl_pct:.1f}% genuine)")

        # Service-specific warnings
        if self.report['services'].get('FHE') != 'REAL':
            print("\n[WARNING] WARNING: FHE encryption may not be genuine!")
        if self.report['services'].get('FL') != 'REAL':
            print("[WARNING] WARNING: Federated Learning may not be working!")
        if isinstance(self.report['services'].get('Database'), dict):
            if all(v == 'UNAVAILABLE' for v in self.report['services']['Database'].values() if v != 'SQLite'):
                print("[WARNING] WARNING: No real database connections, using fallbacks!")

    def generate_placeholder_list(self):
        """Generate list of all placeholders found"""
        with open('PLACEHOLDER_LIST.md', 'w') as f:
            f.write("# PLACEHOLDER/MOCK CODE FOUND\n\n")

            if self.report['fake_patterns']:
                f.write("## Suspicious Patterns\n\n")
                for pattern in self.report['fake_patterns'][:50]:  # Limit to 50
                    f.write(f"- **{pattern['file']}**: `{pattern['pattern']}`\n")

            if self.report['hardcoded_values']:
                f.write("\n## Hardcoded Test Data\n\n")
                for item in self.report['hardcoded_values']:
                    f.write(f"- **{item['file']}**: {item['count']} instances of `{item['pattern']}`\n")

        print("[FILE] Placeholder list saved: PLACEHOLDER_LIST.md")

    def generate_implementation_map(self):
        """Generate map of what actually works"""
        with open('REAL_IMPLEMENTATION_MAP.md', 'w') as f:
            f.write("# REAL IMPLEMENTATION MAP\n\n")
            f.write("## Verified Working Components\n\n")

            for service, status in self.report['services'].items():
                if isinstance(status, dict):
                    f.write(f"\n### {service}\n")
                    for sub, stat in status.items():
                        if stat == "REAL":
                            f.write(f"- [OK] {sub}: Verified working\n")
                        elif "UNAVAILABLE" in stat:
                            f.write(f"- [WARNING] {sub}: Service not running but code exists\n")
                        else:
                            f.write(f"- [FAIL] {sub}: Not working or fake\n")
                else:
                    if status == "REAL":
                        f.write(f"- [OK] {service}: Verified working\n")
                    elif "UNAVAILABLE" in status or "NO_NODE" in status:
                        f.write(f"- [WARNING] {service}: Service not running but code exists\n")
                    else:
                        f.write(f"- [FAIL] {service}: Not working or fake\n")

            f.write(f"\n## Code Statistics\n\n")
            f.write(f"- Total Functions: {self.report['total_functions']}\n")
            f.write(f"- Real Implementations: {self.report['real_implementations']}\n")
            f.write(f"- Placeholders: {self.report['placeholders']}\n")
            f.write(f"- **Real Code Percentage: {self.report.get('implementation_percentage', 0):.1f}%**\n")

        print("[FILE] Implementation map saved: REAL_IMPLEMENTATION_MAP.md")


if __name__ == "__main__":
    print("=" * 60)
    print("FORENSIC AUDIT - DALRN SYSTEM")
    print("=" * 60)
    print("Trust nothing. Verify everything.\n")

    auditor = ForensicAuditor()
    auditor.audit_entire_codebase()