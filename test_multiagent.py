"""
Test Multi-Agent Functionality
"""
import requests
import json

print("=" * 60)
print("MULTI-AGENT FUNCTIONALITY TEST")
print("=" * 60)

# Test 1: Check if service is running
print("\n1. Service Health Check:")
try:
    response = requests.get("http://localhost:8500/health")
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Service running: {data['service']}")
        print(f"   [OK] Status: {data['status']}")
        print(f"   [OK] Active networks: {data['active_networks']}")
    else:
        print(f"   [FAIL] Service returned status: {response.status_code}")
except Exception as e:
    print(f"   [FAIL] Service not reachable: {e}")

# Test 2: Try to initialize a network
print("\n2. Initialize Multi-Agent Network:")
try:
    payload = {
        "n_agents": 10,
        "k": 4,
        "p": 0.3
    }
    response = requests.post(
        "http://localhost:8500/api/v1/soan/initialize",
        json=payload
    )

    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Network created successfully")
        print(f"   [OK] Network ID: {data.get('network_id', 'N/A')}")
        print(f"   [OK] Number of nodes: {data.get('n_nodes', 'N/A')}")
        print(f"   [OK] Number of edges: {data.get('n_edges', 'N/A')}")
    else:
        print(f"   [FAIL] Failed with status {response.status_code}")
        print(f"   [FAIL] Error: {response.json()}")
except Exception as e:
    print(f"   [FAIL] Request failed: {e}")

# Test 3: Check available endpoints
print("\n3. Available Endpoints:")
try:
    response = requests.get("http://localhost:8500/openapi.json")
    if response.status_code == 200:
        api_spec = response.json()
        endpoints = list(api_spec.get('paths', {}).keys())
        for endpoint in endpoints[:5]:  # Show first 5
            print(f"   • {endpoint}")
        if len(endpoints) > 5:
            print(f"   • ... and {len(endpoints)-5} more")
except Exception as e:
    print(f"   [FAIL] Could not fetch endpoints: {e}")

# Test 4: Analyze actual functionality
print("\n4. FUNCTIONALITY ANALYSIS:")
print("-" * 40)

try:
    # Check what the service claims to do
    response = requests.get("http://localhost:8500/openapi.json")
    if response.status_code == 200:
        api_spec = response.json()
        operations = []
        for path, methods in api_spec.get('paths', {}).items():
            for method, details in methods.items():
                if method in ['get', 'post', 'put', 'delete']:
                    operations.append(f"{method.upper()} {path}: {details.get('summary', 'No description')}")

        print("Claimed capabilities:")
        for op in operations[:10]:
            print(f"   • {op}")
except:
    pass

print("\n" + "=" * 60)
print("CONCLUSION:")
print("=" * 60)

# Final verdict
issues = []
working = []

try:
    # Test basic health
    r = requests.get("http://localhost:8500/health")
    if r.status_code == 200:
        working.append("Service starts and responds to health checks")

    # Test network initialization
    r = requests.post("http://localhost:8500/api/v1/soan/initialize",
                     json={"n_agents": 5, "k": 2, "p": 0.1})
    if r.status_code == 200:
        working.append("Can initialize agent networks")
    else:
        issues.append(f"Network initialization fails: {r.json().get('detail', 'Unknown error')}")

except Exception as e:
    issues.append(f"Service communication error: {e}")

if working:
    print("[OK] What works:")
    for w in working:
        print(f"  • {w}")

if issues:
    print("\n[FAIL] What doesn't work:")
    for i in issues:
        print(f"  • {i}")

print("\nVERDICT: ", end="")
if not issues:
    print("Multi-agent functionality is WORKING")
elif len(working) > len(issues):
    print("Multi-agent functionality is PARTIALLY WORKING")
else:
    print("Multi-agent functionality has SIGNIFICANT ISSUES")

print("=" * 60)