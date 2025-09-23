
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
