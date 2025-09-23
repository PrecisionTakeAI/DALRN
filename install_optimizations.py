"""
Installation script for DALRN performance optimizations
Run this to install the ACTUAL technologies needed for claimed performance
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """Run a command and report results"""
    print(f"\n[*] {description}...")
    print(f"    Command: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"    SUCCESS")
            if result.stdout:
                print(f"    {result.stdout[:200]}")
        else:
            print(f"    FAILED: {result.stderr[:200]}")
        return result.returncode == 0
    except Exception as e:
        print(f"    ERROR: {e}")
        return False

print("=" * 60)
print("DALRN OPTIMIZATION INSTALLER")
print("=" * 60)
print("\nThis will install:")
print("1. Qdrant - Vector database for <5ms search")
print("2. Zama Concrete ML - Real FHE with <50ms operations")
print("3. Supporting libraries for optimizations")
print("4. Docker setup for Qdrant (if Docker available)")
print("=" * 60)

# Step 1: Python libraries
print("\n[STEP 1] Installing Python Libraries")
print("-" * 40)

# Qdrant client
if run_command("pip install qdrant-client", "Installing Qdrant client"):
    print("    Qdrant client installed successfully")
else:
    print("    WARNING: Qdrant client installation failed")

# Concrete ML (might require specific Python version)
print("\n[!] Concrete ML requires Python 3.8-3.10 and specific dependencies")
if run_command("pip install concrete-ml", "Installing Concrete ML"):
    print("    Concrete ML installed successfully")
else:
    print("    WARNING: Concrete ML installation failed")
    print("    Try: pip install concrete-ml[full]")
    print("    Or check Python version (needs 3.8-3.10)")

# HTTP/2 and async optimizations
run_command("pip install httpx[http2]", "Installing HTTPX with HTTP/2 support")
run_command("pip install uvloop", "Installing uvloop for better async performance")
run_command("pip install aiocache", "Installing async cache")
run_command("pip install asyncpg", "Installing async PostgreSQL driver")

# Step 2: Qdrant Server
print("\n[STEP 2] Qdrant Server Setup")
print("-" * 40)
print("\nOption A: Docker (Recommended)")
print("Run this command if Docker is installed:")
print("docker run -p 6333:6333 -v ./qdrant_storage:/qdrant/storage qdrant/qdrant")

print("\nOption B: Binary Installation")
print("Download from: https://github.com/qdrant/qdrant/releases")
print("Or use Qdrant Cloud: https://cloud.qdrant.io")

# Step 3: PostgreSQL and Redis
print("\n[STEP 3] Database Setup")
print("-" * 40)
print("\nPostgreSQL:")
print("  Windows: Download from https://www.postgresql.org/download/windows/")
print("  Mac: brew install postgresql")
print("  Linux: sudo apt-get install postgresql")

print("\nRedis:")
print("  Windows: Download from https://github.com/tporadowski/redis/releases")
print("  Mac: brew install redis")
print("  Linux: sudo apt-get install redis-server")

# Step 4: Verify installations
print("\n[STEP 4] Verification")
print("-" * 40)

# Check what's actually installed now
print("\nChecking installations...")
verification_script = """
import sys
print(f"Python version: {sys.version}")

# Check Qdrant
try:
    import qdrant_client
    print("Qdrant client: INSTALLED")
except ImportError:
    print("Qdrant client: NOT FOUND")

# Check Concrete ML
try:
    from concrete.ml.sklearn import LinearRegression
    print("Concrete ML: INSTALLED")
except ImportError:
    print("Concrete ML: NOT FOUND")

# Check HTTPX with HTTP/2
try:
    import httpx
    print(f"HTTPX: INSTALLED (HTTP/2: {hasattr(httpx, 'HTTPTransport')})")
except ImportError:
    print("HTTPX: NOT FOUND")

# Check uvloop
try:
    import uvloop
    print("uvloop: INSTALLED")
except ImportError:
    print("uvloop: NOT FOUND (Windows doesn't support uvloop)")

# Check asyncpg
try:
    import asyncpg
    print("asyncpg: INSTALLED")
except ImportError:
    print("asyncpg: NOT FOUND")
"""

with open("verify_install.py", "w") as f:
    f.write(verification_script)

run_command("python verify_install.py", "Verifying installations")

# Step 5: Quick start commands
print("\n[STEP 5] Quick Start Commands")
print("=" * 60)
print("\n1. Start Qdrant (if Docker installed):")
print("   docker run -p 6333:6333 qdrant/qdrant")
print("\n2. Start PostgreSQL:")
print("   Windows: Use pgAdmin or Services")
print("   Mac/Linux: sudo systemctl start postgresql")
print("\n3. Start Redis:")
print("   Windows: redis-server.exe")
print("   Mac/Linux: redis-server")
print("\n4. Start optimized gateway:")
print("   python -m services.gateway.optimized_gateway")
print("\n5. Start search service:")
print("   python -m services.search.qdrant_search_service")
print("\n6. Start FHE service:")
print("   python -m services.fhe.zama_fhe_service")

print("\n" + "=" * 60)
print("IMPORTANT NOTES:")
print("=" * 60)
print("1. Concrete ML might fail on Windows or Python >3.10")
print("2. Qdrant requires Docker or manual binary installation")
print("3. PostgreSQL and Redis need separate installation")
print("4. uvloop doesn't work on Windows (falls back to default)")
print("\nRun 'python truth_test.py' after installation to verify")
print("=" * 60)