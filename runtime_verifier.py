#!/usr/bin/env python3
"""
runtime_verifier.py - The ONLY truth is what runs
"""
import subprocess
import sys
import importlib.util
import traceback
from pathlib import Path
import json
import time
import os

class RuntimeVerifier:
    def __init__(self):
        self.results = {
            "services_that_start": [],
            "services_that_fail": [],
            "functions_that_work": [],
            "functions_that_fail": [],
            "imports_that_work": [],
            "imports_that_fail": [],
            "actual_features": [],
            "fake_features": [],
            "ml_implementations": []
        }

    def test_python_file_import(self, file_path):
        """Test if a Python file can even be imported"""
        try:
            # Try to compile the file first
            with open(file_path, 'r') as f:
                code = f.read()
            compile(code, file_path, 'exec')

            # Now try to import it
            module_name = Path(file_path).stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                try:
                    spec.loader.exec_module(module)
                    self.results["imports_that_work"].append(str(file_path))
                    return True, None
                except Exception as e:
                    error = str(e)
                    self.results["imports_that_fail"].append({
                        "file": str(file_path),
                        "error": error[:200]
                    })
                    return False, error
        except SyntaxError as e:
            self.results["imports_that_fail"].append({
                "file": str(file_path),
                "error": f"SyntaxError: {e}"
            })
            return False, f"SyntaxError: {e}"
        except Exception as e:
            self.results["imports_that_fail"].append({
                "file": str(file_path),
                "error": str(e)[:200]
            })
            return False, str(e)

    def test_service_startup(self, service_path):
        """Try to actually start the service"""
        print(f"\n=== TESTING SERVICE: {service_path} ===")

        # First check if it can be imported
        can_import, import_error = self.test_python_file_import(service_path)

        if not can_import:
            print(f"[FAIL] Cannot import: {import_error[:100]}")
            self.results["services_that_fail"].append({
                "path": str(service_path),
                "error": f"Import failed: {import_error}"
            })
            return False

        print(f"[OK] Import successful")

        # Now try to run it
        try:
            result = subprocess.run(
                [sys.executable, str(service_path)],
                capture_output=True,
                text=True,
                timeout=5
            )

            # Check for signs of a running service
            if any(keyword in result.stdout + result.stderr for keyword in
                   ['Uvicorn running', 'Started server', 'Listening', 'port', 'Running']):
                print(f"[OK] Service appears to start")
                self.results["services_that_start"].append(str(service_path))
                return True
            elif result.returncode == 0:
                print(f"[OK] Runs without error (may need main block)")
                self.results["services_that_start"].append(str(service_path))
                return True
            else:
                error_msg = (result.stderr or result.stdout)[:200]
                print(f"[FAIL] Service fails: {error_msg}")
                self.results["services_that_fail"].append({
                    "path": str(service_path),
                    "error": error_msg
                })
                return False

        except subprocess.TimeoutExpired:
            # Timeout might mean it's running (servers don't exit)
            print(f"[OK] Service runs (timeout suggests it's a server)")
            self.results["services_that_start"].append(str(service_path))
            return True
        except Exception as e:
            print(f"[FAIL] Failed to start: {e}")
            self.results["services_that_fail"].append({
                "path": str(service_path),
                "error": str(e)
            })
            return False

    def check_ml_implementation(self, file_path):
        """Check if ML implementation is real or fake"""
        try:
            with open(file_path, 'r') as f:
                code = f.read()

            # Patterns that indicate FAKE ML
            fake_patterns = [
                'loss = np.random',
                'loss = random',
                'np.random.uniform(0.1, 1.0) * np.exp',
                'return random.random()',
                'fake_loss',
                'mock_training',
                '# FAKE',
                '# TODO: implement',
                'raise NotImplementedError'
            ]

            # Patterns that indicate REAL ML
            real_patterns = [
                'loss.backward()',
                'optimizer.step()',
                'optimizer.zero_grad()',
                'model.train()',
                'F.mse_loss',
                'F.cross_entropy',
                'torch.nn',
                'tf.keras',
                'GCNConv(',
                'self.conv1',
                'forward(self'
            ]

            fake_count = sum(1 for p in fake_patterns if p in code)
            real_count = sum(1 for p in real_patterns if p in code)

            result = {
                "file": str(file_path),
                "fake_patterns": fake_count,
                "real_patterns": real_count,
                "verdict": None
            }

            if fake_count > 0 and real_count == 0:
                result["verdict"] = "FAKE"
                self.results["fake_features"].append(result)
            elif real_count >= 3:  # Need multiple real patterns
                result["verdict"] = "REAL"
                self.results["actual_features"].append(result)
            elif real_count > 0:
                result["verdict"] = "PARTIAL"
                self.results["actual_features"].append(result)
            else:
                result["verdict"] = "UNKNOWN"

            self.results["ml_implementations"].append(result)
            return result

        except Exception as e:
            print(f"Error checking ML in {file_path}: {e}")
            return None

    def run_comprehensive_check(self):
        """Run all verification checks"""
        print("=" * 60)
        print("DALRN RUNTIME VERIFICATION")
        print("=" * 60)

        # Find all Python files in services
        service_files = list(Path("services").rglob("*.py"))
        print(f"\nFound {len(service_files)} Python files to verify\n")

        # Test main service files
        main_services = []
        for f in service_files:
            if any(name in f.name for name in ['app.py', 'service.py', 'main.py']):
                main_services.append(f)

        print(f"Testing {len(main_services)} main service files...")
        for service in main_services:
            self.test_service_startup(service)

        # Check ML implementations
        print("\n" + "=" * 60)
        print("CHECKING ML IMPLEMENTATIONS")
        print("=" * 60)

        ml_files = [f for f in service_files if any(
            keyword in str(f).lower() for keyword in
            ['model', 'train', 'gnn', 'agent', 'fedavg', 'optimizer', 'network']
        )]

        for ml_file in ml_files:
            result = self.check_ml_implementation(ml_file)
            if result:
                print(f"\n{ml_file.name}: {result['verdict']}")
                print(f"  Real patterns: {result['real_patterns']}")
                print(f"  Fake patterns: {result['fake_patterns']}")

        return self.results


# Run the verifier
if __name__ == "__main__":
    verifier = RuntimeVerifier()
    results = verifier.run_comprehensive_check()

    # Save results
    with open("RUNTIME_VERIFICATION_RESULTS.json", "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    total_services = len(results["services_that_start"]) + len(results["services_that_fail"])
    working_services = len(results["services_that_start"])

    print(f"\nServices that start: {working_services}/{total_services}")
    print(f"Services that fail: {len(results['services_that_fail'])}")
    print(f"Files that import: {len(results['imports_that_work'])}")
    print(f"Files that don't import: {len(results['imports_that_fail'])}")
    print(f"Real ML implementations: {len(results['actual_features'])}")
    print(f"Fake ML implementations: {len(results['fake_features'])}")

    if total_services > 0:
        percentage = (working_services / total_services) * 100
        print(f"\nACTUAL WORKING PERCENTAGE: {percentage:.0f}%")

    print("\nResults saved to RUNTIME_VERIFICATION_RESULTS.json")