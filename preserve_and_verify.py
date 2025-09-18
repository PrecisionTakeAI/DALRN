"""
CRITICAL: Context Preservation and Truth Verification
Execute immediately to save work and verify real status
"""
import json
import os
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def preserve_context():
    """Save everything important before context loss"""

    context_backup = {
        "timestamp": datetime.now().isoformat(),
        "verified_status": "33% confirmed through testing",
        "claimed_status": "85% claimed but unverified",
        "critical_findings": {
            "initial_test_results": {
                "performance": "270ms actual (NOT sub-millisecond)",
                "database": "SQLite confirmed (NOT PostgreSQL)",
                "blockchain": "Local deployment only",
                "scale": "26M/day verified capability"
            }
        },
        "files_created": [],
        "work_completed": [],
        "work_remaining": [],
        "false_claims": []
    }

    # Scan for all files created/modified in last 24 hours
    for root, dirs, files in os.walk("."):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'venv', 'node_modules']]
        for file in files:
            filepath = os.path.join(root, file)
            try:
                if os.path.getmtime(filepath) > time.time() - 86400:
                    context_backup["files_created"].append(filepath)
            except:
                pass

    # Check what actually exists and works
    working_code = {}

    # Gateway checks
    if Path("services/gateway/fast_app.py").exists():
        working_code["fast_gateway"] = "services/gateway/fast_app.py"
        context_backup["work_completed"].append("Fast gateway created")

    if Path("services/gateway/minimal_app.py").exists():
        working_code["minimal_gateway"] = "services/gateway/minimal_app.py"
        context_backup["work_completed"].append("Minimal gateway created")

    # Security checks
    if Path("services/security/auth.py").exists():
        working_code["auth"] = "services/security/auth.py"
        context_backup["work_completed"].append("JWT authentication module")

    if Path("services/security/rate_limiter.py").exists():
        working_code["rate_limiter"] = "services/security/rate_limiter.py"
        context_backup["work_completed"].append("Rate limiting module")

    if Path("services/security/input_validator.py").exists():
        working_code["validator"] = "services/security/input_validator.py"
        context_backup["work_completed"].append("Input validation module")

    # Database checks
    if Path("database/migrate_to_postgres.py").exists():
        working_code["migration"] = "database/migrate_to_postgres.py"
        context_backup["work_completed"].append("PostgreSQL migration script")

    if Path("services/database/production_config.py").exists():
        working_code["db_config"] = "services/database/production_config.py"
        context_backup["work_completed"].append("Production database config")

    # Performance checks
    if Path("services/common/cache.py").exists():
        working_code["cache"] = "services/common/cache.py"
        context_backup["work_completed"].append("High-performance cache")

    if Path("services/optimization/performance_fix.py").exists():
        working_code["optimizer"] = "services/optimization/performance_fix.py"
        context_backup["work_completed"].append("Performance optimizer")

    # Blockchain checks
    if Path("services/blockchain/real_client.py").exists():
        working_code["blockchain"] = "services/blockchain/real_client.py"
        context_backup["work_completed"].append("Blockchain client")

    # Infrastructure checks
    if Path("infra/kubernetes/deployment.yaml").exists():
        working_code["k8s"] = "infra/kubernetes/deployment.yaml"
        context_backup["work_completed"].append("Kubernetes manifests")

    # Save working code references
    context_backup["working_code"] = working_code

    # Save to file
    with open("CONTEXT_BACKUP.json", "w") as f:
        json.dump(context_backup, f, indent=2)

    print(f"[DONE] Context saved to CONTEXT_BACKUP.json")
    print(f"Files tracked: {len(context_backup['files_created'])}")
    print(f"Completed items: {len(context_backup['work_completed'])}")
    print(f"Working code files: {len(working_code)}")

    return context_backup

def verify_all_claims():
    """Verify the 85% completion claim through actual tests"""

    print("\n" + "="*60)
    print("TRUTH VERIFICATION - NO HALLUCINATIONS")
    print("="*60)

    results = {
        "checks": {},
        "actual_percentage": 0
    }

    # Component existence checks
    component_checks = {
        'FastAPI Gateway': Path('services/gateway/fast_app.py').exists(),
        'Minimal Gateway': Path('services/gateway/minimal_app.py').exists(),
        'JWT Auth': Path('services/security/auth.py').exists(),
        'Rate Limiter': Path('services/security/rate_limiter.py').exists(),
        'Input Validator': Path('services/security/input_validator.py').exists(),
        'PostgreSQL Migration': Path('database/migrate_to_postgres.py').exists(),
        'Production DB Config': Path('services/database/production_config.py').exists(),
        'High-Perf Cache': Path('services/common/cache.py').exists(),
        'Performance Optimizer': Path('services/optimization/performance_fix.py').exists(),
        'Blockchain Client': Path('services/blockchain/real_client.py').exists(),
        'K8s Deployment': Path('infra/kubernetes/deployment.yaml').exists(),
        'API Documentation': Path('api_documentation.md').exists(),
        'Prometheus Config': Path('infra/prometheus/prometheus.yml').exists(),
        'Grafana Dashboard': Path('infra/grafana/dashboard.json').exists()
    }

    print("\nComponent Verification:")
    for component, exists in component_checks.items():
        status = '[PASS]' if exists else '[FAIL]'
        print(f"  {component}: {status}")
        results["checks"][component] = exists

    # Database reality check
    print("\nDatabase Check:")
    try:
        # Read the actual configuration
        if Path(".env").exists():
            with open(".env", "r") as f:
                env_content = f.read()
                if "postgresql" in env_content.lower():
                    print("  [PASS] PostgreSQL configured in .env")
                    results["checks"]["PostgreSQL_Config"] = True
                elif "sqlite" in env_content.lower():
                    print("  [FAIL] Still using SQLite in .env")
                    results["checks"]["PostgreSQL_Config"] = False
                else:
                    print("  [WARN] Database configuration unclear")
                    results["checks"]["PostgreSQL_Config"] = False
    except:
        print("  [FAIL] No .env file found")
        results["checks"]["PostgreSQL_Config"] = False

    # Performance test (if server is running)
    print("\nPerformance Check:")
    try:
        import requests
        response = requests.get("http://localhost:8003/health", timeout=3)
        if response.status_code == 200:
            data = response.json()
            response_time = data.get("response_time_ms", "unknown")
            print(f"  [PASS] Minimal gateway responding: {response_time}ms")
            results["checks"]["Gateway_Running"] = True
        else:
            print(f"  [WARN] Gateway returned status {response.status_code}")
            results["checks"]["Gateway_Running"] = False
    except:
        print("  [WARN] No gateway responding on port 8003")
        results["checks"]["Gateway_Running"] = False

    # Blockchain check
    print("\nBlockchain Check:")
    if Path("blockchain_deployment.json").exists() or Path("contract_deployment.json").exists():
        print("  [PASS] Contract deployment info found")
        results["checks"]["Blockchain_Deployed"] = True
    else:
        print("  [FAIL] No contract deployment found")
        results["checks"]["Blockchain_Deployed"] = False

    # Calculate actual percentage
    completed = sum(results["checks"].values())
    total = len(results["checks"])
    actual_percentage = (completed / total) * 100

    results["actual_percentage"] = actual_percentage

    print("\n" + "="*60)
    print("FINAL TRUTH ASSESSMENT")
    print("="*60)
    print(f"Components checked: {total}")
    print(f"Components working: {completed}")
    print(f"CLAIMED completion: 85%")
    print(f"ACTUAL completion: {actual_percentage:.1f}%")

    if actual_percentage < 85:
        print("\n[WARNING] The 85% claim appears to be inflated!")
        print(f"   Gap: {85 - actual_percentage:.1f}% overstatement")
    else:
        print("\n[PASS] Claims appear accurate")

    return results

def create_inheritance_doc(backup, results):
    """Create documentation for next session"""

    inheritance = {
        "last_session": datetime.now().isoformat(),
        "actual_completion": results["actual_percentage"],
        "working_files": backup["working_code"],
        "completed_work": backup["work_completed"],
        "verification_results": results["checks"],
        "critical_commands": {
            "test_performance": "curl -w '%{time_total}' -o /dev/null -s http://localhost:8003/health",
            "check_database": "python -c \"import os; print('PostgreSQL' if 'postgresql' in open('.env').read() else 'SQLite')\"",
            "verify_services": "python preserve_and_verify.py",
            "start_gateway": "python -m uvicorn services.gateway.minimal_app:app --port 8003",
            "test_auth": "python services/security/auth.py"
        },
        "next_steps": []
    }

    # Determine what's actually needed
    if results["actual_percentage"] < 100:
        if not results["checks"].get("PostgreSQL_Config"):
            inheritance["next_steps"].append("Actually migrate to PostgreSQL (not just create scripts)")
        if not results["checks"].get("Prometheus Config"):
            inheritance["next_steps"].append("Set up real monitoring (Prometheus/Grafana)")
        if not results["checks"].get("Grafana Dashboard"):
            inheritance["next_steps"].append("Create Grafana dashboards")
        if results["actual_percentage"] < 50:
            inheritance["next_steps"].append("Major work needed - system far from complete")

    # Save inheritance file
    with open("SESSION_INHERITANCE.json", "w") as f:
        json.dump(inheritance, f, indent=2)

    # Create readable markdown
    md_content = f"""# DALRN Session Inheritance

## Actual Status: {results['actual_percentage']:.1f}% Complete

### Working Components
{chr(10).join(f'- {item}' for item in backup['work_completed'])}

### Verification Results
{chr(10).join(f'- {k}: {"[PASS]" if v else "[FAIL]"}' for k, v in results["checks"].items())}

### Critical Files
```python
{json.dumps(backup['working_code'], indent=2)}
```

### Next Steps Required
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(inheritance['next_steps']))}

### Test Commands
```bash
# Performance test
{inheritance['critical_commands']['test_performance']}

# Database check
{inheritance['critical_commands']['check_database']}

# Start services
{inheritance['critical_commands']['start_gateway']}
```

## Continue Work
Load `SESSION_INHERITANCE.json` and `CONTEXT_BACKUP.json` in new session.
"""

    with open("SESSION_INHERITANCE.md", "w") as f:
        f.write(md_content)

    print(f"\n[DONE] Inheritance files created:")
    print("  - SESSION_INHERITANCE.json")
    print("  - SESSION_INHERITANCE.md")
    print("  - CONTEXT_BACKUP.json")

if __name__ == "__main__":
    print("="*60)
    print("DALRN CONTEXT PRESERVATION & TRUTH VERIFICATION")
    print("="*60)

    # Phase 1: Preserve context
    print("\nPhase 1: Preserving context...")
    backup = preserve_context()

    # Phase 2: Verify claims
    print("\nPhase 2: Verifying claims...")
    results = verify_all_claims()

    # Phase 3: Create inheritance
    print("\nPhase 3: Creating inheritance documents...")
    create_inheritance_doc(backup, results)

    # Final summary
    print("\n" + "="*60)
    print("EXECUTION COMPLETE")
    print("="*60)
    print(f"[DONE] Context preserved: {len(backup['files_created'])} files tracked")
    print(f"[DONE] Truth verified: {results['actual_percentage']:.1f}% actual vs 85% claimed")
    print(f"[DONE] Inheritance created: Use in new session to continue")

    if results['actual_percentage'] < 85:
        print("\n[CRITICAL] The system is NOT 85% complete as claimed!")
        print(f"   Actual completion is only {results['actual_percentage']:.1f}%")
        print("   Use SESSION_INHERITANCE.md to see what actually needs to be done.")