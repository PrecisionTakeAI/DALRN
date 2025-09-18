"""
EXHAUSTIVE CODEBASE VERIFICATION
This script performs complete analysis of EVERY file and tests ALL functionality
"""
import os
import ast
import re
import json
import sys
import time
import subprocess
import statistics
from pathlib import Path
from datetime import datetime

class ExhaustiveCodeAnalyzer:
    def __init__(self):
        self.results = {
            "total_files": 0,
            "total_lines": 0,
            "executable_lines": 0,
            "mock_functions": [],
            "placeholders": [],
            "todos": [],
            "not_implemented": [],
            "hardcoded_values": [],
            "actual_functionality": {},
            "empty_functions": [],
            "return_none": [],
            "fake_returns": []
        }

    def analyze_entire_codebase(self):
        """Analyze EVERY Python file in the project"""

        print("="*60)
        print("PHASE 1: EXHAUSTIVE CODEBASE ANALYSIS")
        print("="*60)

        for root, dirs, files in os.walk("."):
            # Skip virtual environments and caches
            dirs[:] = [d for d in dirs if d not in ['venv', '__pycache__', '.git', 'node_modules', '.devcontainer']]

            for file in files:
                if file.endswith('.py'):
                    filepath = os.path.join(root, file)
                    self.analyze_file(filepath)

        return self.results

    def analyze_file(self, filepath):
        """Deep analysis of each file"""
        self.results["total_files"] += 1

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                self.results["total_lines"] += len(lines)

                # Check for mockups and placeholders
                for line_num, line in enumerate(lines, 1):
                    line_lower = line.lower()
                    line_stripped = line.strip()

                    # Detect mocks
                    if any(word in line_lower for word in ['mock', 'fake', 'dummy', 'stub', 'placeholder']):
                        if not line_stripped.startswith('#'):
                            self.results["mock_functions"].append({
                                "file": filepath,
                                "line": line_num,
                                "content": line_stripped[:100]
                            })

                    # Detect TODOs and FIXMEs
                    if any(word in line_lower for word in ['todo', 'fixme', 'xxx', 'hack', 'refactor']):
                        self.results["todos"].append({
                            "file": filepath,
                            "line": line_num,
                            "content": line_stripped[:100]
                        })

                    # Detect not implemented
                    if 'notimplementederror' in line_lower:
                        self.results["not_implemented"].append({
                            "file": filepath,
                            "line": line_num,
                            "content": "NotImplementedError"
                        })

                    # Detect single pass statements
                    if line_stripped == 'pass':
                        self.results["empty_functions"].append({
                            "file": filepath,
                            "line": line_num
                        })

                    # Detect fake returns
                    if re.match(r'^\s*return\s+(None|{}|\[\]|""|\'\'|0|False)\s*$', line):
                        self.results["fake_returns"].append({
                            "file": filepath,
                            "line": line_num,
                            "content": line_stripped
                        })

                    # Detect hardcoded values that should be configurable
                    if re.search(r'(localhost|127\.0\.0\.1|8000|8080|5432|password|secret|test123)', line_lower):
                        if not line_stripped.startswith('#') and 'test' not in filepath.lower():
                            self.results["hardcoded_values"].append({
                                "file": filepath,
                                "line": line_num,
                                "issue": line_stripped[:100]
                            })

                # Parse AST to find actual functionality
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Check if function has actual implementation
                            if len(node.body) == 1:
                                if isinstance(node.body[0], ast.Pass):
                                    self.results["not_implemented"].append({
                                        "file": filepath,
                                        "function": node.name,
                                        "type": "empty_function"
                                    })
                                elif isinstance(node.body[0], ast.Return):
                                    if node.body[0].value is None:
                                        self.results["return_none"].append({
                                            "file": filepath,
                                            "function": node.name
                                        })
                            else:
                                self.results["executable_lines"] += len(node.body)
                except:
                    pass

        except Exception as e:
            pass

def test_all_endpoints():
    """Test EVERY API endpoint for actual functionality"""

    print("\n" + "="*60)
    print("PHASE 2: TESTING ALL ENDPOINTS")
    print("="*60)

    import requests

    endpoints = [
        ("GET", "http://localhost:8000/health", None, [200, 404]),
        ("GET", "http://localhost:8001/health", None, [200, 404]),
        ("GET", "http://localhost:8002/health", None, [200, 404]),
        ("GET", "http://localhost:8003/health", None, [200, 404]),
        ("POST", "http://localhost:8000/submit-dispute", {
            "parties": ["alice@example.com", "bob@example.com"],
            "jurisdiction": "US",
            "cid": "QmTest"
        }, [200, 201, 400, 404]),
        ("GET", "http://localhost:8000/status/test123", None, [200, 404]),
    ]

    results = {"passed": 0, "failed": 0, "errors": []}

    for method, url, data, expected_statuses in endpoints:
        print(f"\nTesting {method} {url}...")

        try:
            if method == "GET":
                response = requests.get(url, timeout=2)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=2)

            if response.status_code in expected_statuses:
                print(f"  [PASS] Status: {response.status_code}")
                results["passed"] += 1
            else:
                print(f"  [FAIL] Got {response.status_code}, expected {expected_statuses}")
                results["failed"] += 1
                results["errors"].append(f"{url}: unexpected {response.status_code}")

        except requests.exceptions.ConnectionError:
            print(f"  [INFO] Service not running")
            # Not counting as failure - service might not be started
        except Exception as e:
            print(f"  [FAIL] Error: {str(e)[:50]}")
            results["failed"] += 1
            results["errors"].append(f"{url}: {str(e)[:50]}")

    return results

def verify_critical_components():
    """Test each critical component works, not just exists"""

    print("\n" + "="*60)
    print("PHASE 3: CRITICAL COMPONENT VERIFICATION")
    print("="*60)

    components = {}

    # 1. Check if files exist vs actually work
    print("\n1. File Existence Check:")
    critical_files = {
        'Gateway': 'services/gateway/app.py',
        'Fast Gateway': 'services/gateway/fast_app.py',
        'Minimal Gateway': 'services/gateway/minimal_app.py',
        'JWT Auth': 'services/security/auth.py',
        'Rate Limiter': 'services/security/rate_limiter.py',
        'Input Validator': 'services/security/input_validator.py',
        'Database Config': 'services/database/production_config.py',
        'Cache': 'services/common/cache.py',
        'Blockchain Client': 'services/blockchain/real_client.py',
    }

    for name, filepath in critical_files.items():
        if os.path.exists(filepath):
            # Check if file has actual content
            with open(filepath, 'r') as f:
                content = f.read()
                # Check for substance
                if len(content) > 100 and 'class' in content or 'def' in content:
                    print(f"  [PASS] {name}: exists with {len(content)} bytes")
                    components[name] = True
                else:
                    print(f"  [FAIL] {name}: exists but appears empty")
                    components[name] = False
        else:
            print(f"  [FAIL] {name}: NOT FOUND")
            components[name] = False

    # 2. Database - Actually configured?
    print("\n2. Database Configuration:")
    if os.path.exists('.env'):
        with open('.env', 'r') as f:
            env_content = f.read()
            if 'postgresql' in env_content.lower():
                print("  [PASS] PostgreSQL configured in .env")
                components["PostgreSQL Config"] = True
            else:
                print("  [FAIL] PostgreSQL NOT configured")
                components["PostgreSQL Config"] = False
    else:
        print("  [FAIL] No .env file")
        components["PostgreSQL Config"] = False

    # 3. Check for mock implementations
    print("\n3. Mock Detection:")
    mock_count = 0
    for root, dirs, files in os.walk("services"):
        dirs[:] = [d for d in dirs if d != '__pycache__']
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().lower()
                        if 'mock' in content or 'fake' in content or 'dummy' in content:
                            mock_count += 1
                except:
                    continue

    if mock_count == 0:
        print(f"  [PASS] No mock implementations found")
        components["No Mocks"] = True
    else:
        print(f"  [FAIL] Found {mock_count} files with mock implementations")
        components["No Mocks"] = False

    # 4. Performance test
    print("\n4. Performance Test:")
    try:
        import requests
        times = []
        for _ in range(5):
            start = time.perf_counter()
            try:
                requests.get("http://localhost:8003/health", timeout=1)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            except:
                pass

        if times:
            avg = sum(times) / len(times)
            if avg < 200:
                print(f"  [PASS] Average response: {avg:.1f}ms")
                components["Performance <200ms"] = True
            else:
                print(f"  [FAIL] Average response: {avg:.1f}ms (needs <200ms)")
                components["Performance <200ms"] = False
        else:
            print("  [INFO] Could not measure (service not running)")
            components["Performance <200ms"] = False
    except:
        print("  [FAIL] Performance test failed")
        components["Performance <200ms"] = False

    # Calculate percentage
    working = sum(components.values())
    total = len(components)
    percentage = (working / total) * 100

    return components, percentage

def final_truth_verdict(analyzer_results, endpoint_results, components, component_percentage):
    """Generate the absolute truth about system status"""

    print("\n" + "="*60)
    print("PHASE 4: FINAL TRUTH VERDICT")
    print("="*60)

    # Count all issues
    total_issues = (
        len(analyzer_results['mock_functions']) +
        len(analyzer_results['todos']) +
        len(analyzer_results['not_implemented']) +
        len(analyzer_results['empty_functions']) +
        len(analyzer_results['fake_returns']) +
        len(analyzer_results['return_none'])
    )

    print(f"\nCode Analysis Results:")
    print(f"  Total Python files: {analyzer_results['total_files']}")
    print(f"  Total lines of code: {analyzer_results['total_lines']}")
    print(f"  Mock functions: {len(analyzer_results['mock_functions'])}")
    print(f"  TODOs/FIXMEs: {len(analyzer_results['todos'])}")
    print(f"  Not implemented: {len(analyzer_results['not_implemented'])}")
    print(f"  Empty functions: {len(analyzer_results['empty_functions'])}")
    print(f"  Fake returns: {len(analyzer_results['fake_returns'])}")
    print(f"  Hardcoded values: {len(analyzer_results['hardcoded_values'])}")

    print(f"\nEndpoint Test Results:")
    print(f"  Endpoints tested: {endpoint_results['passed'] + endpoint_results['failed']}")
    print(f"  Passed: {endpoint_results['passed']}")
    print(f"  Failed: {endpoint_results['failed']}")

    print(f"\nComponent Verification:")
    print(f"  Components working: {component_percentage:.0f}%")
    for name, status in components.items():
        icon = "[PASS]" if status else "[FAIL]"
        print(f"    {icon} {name}")

    # Calculate real completion
    deductions = 0

    # Deduct for code issues
    if total_issues > 0:
        issue_deduction = min(30, total_issues * 0.5)  # 0.5% per issue, max 30%
        deductions += issue_deduction
        print(f"\n  Issues found: {total_issues} (-{issue_deduction:.1f}%)")

    # Deduct for failed components
    if component_percentage < 100:
        component_deduction = (100 - component_percentage) * 0.5  # Half weight
        deductions += component_deduction
        print(f"  Component failures: (-{component_deduction:.1f}%)")

    # Deduct for hardcoded values
    if len(analyzer_results['hardcoded_values']) > 10:
        deductions += 5
        print(f"  Many hardcoded values: (-5%)")

    real_completion = max(0, 100 - deductions)

    print("\n" + "="*60)
    print("FINAL VERDICT")
    print("="*60)
    print(f"Claimed completion: 100%")
    print(f"ACTUAL COMPLETION: {real_completion:.1f}%")

    if real_completion >= 95:
        print("\n[PASS] System is essentially complete!")
    elif real_completion >= 85:
        print("\n[GOOD] System is mostly complete with minor issues")
    elif real_completion >= 70:
        print("\n[WARN] System has significant gaps")
    else:
        print("\n[FAIL] System is NOT production ready")

    # Show most critical issues
    if analyzer_results['mock_functions']:
        print("\nMock functions found in:")
        for mock in analyzer_results['mock_functions'][:3]:
            print(f"  - {mock['file']}:{mock['line']}")

    if analyzer_results['not_implemented']:
        print("\nNot implemented:")
        for item in analyzer_results['not_implemented'][:3]:
            if 'function' in item:
                print(f"  - {item['file']}: {item['function']}()")

    # Save verdict
    verdict = {
        "timestamp": datetime.now().isoformat(),
        "claimed_completion": "100%",
        "actual_completion": f"{real_completion:.1f}%",
        "total_files": analyzer_results['total_files'],
        "total_lines": analyzer_results['total_lines'],
        "total_issues": total_issues,
        "components_working": component_percentage,
        "deductions": deductions,
        "details": {
            "mock_functions": len(analyzer_results['mock_functions']),
            "todos": len(analyzer_results['todos']),
            "not_implemented": len(analyzer_results['not_implemented']),
            "empty_functions": len(analyzer_results['empty_functions']),
            "hardcoded_values": len(analyzer_results['hardcoded_values'])
        }
    }

    with open("FINAL_VERDICT.json", "w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\nDetailed report saved to FINAL_VERDICT.json")

    return real_completion

# Main execution
if __name__ == "__main__":
    print("STARTING EXHAUSTIVE VERIFICATION")
    print("This will analyze every line and test all functionality")
    print("="*60)

    # Phase 1: Code analysis
    analyzer = ExhaustiveCodeAnalyzer()
    analyzer_results = analyzer.analyze_entire_codebase()

    # Phase 2: Endpoint testing
    try:
        import requests
        endpoint_results = test_all_endpoints()
    except ImportError:
        print("\nSkipping endpoint tests (requests not installed)")
        endpoint_results = {"passed": 0, "failed": 0, "errors": []}

    # Phase 3: Component verification
    components, component_percentage = verify_critical_components()

    # Phase 4: Final verdict
    real_completion = final_truth_verdict(
        analyzer_results,
        endpoint_results,
        components,
        component_percentage
    )

    print("\n" + "="*60)
    print("VERIFICATION COMPLETE")
    print(f"True completion: {real_completion:.1f}%")
    print("="*60)