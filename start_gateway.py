#!/usr/bin/env python3
"""
Start the DALRN Gateway service with proper Python path configuration.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the gateway
if __name__ == "__main__":
    from services.gateway.simple_app import app
    import uvicorn

    port = int(os.getenv("GATEWAY_PORT", 8000))
    print(f"Starting DALRN Gateway on port {port}")
    print(f"Project root: {project_root}")
    print(f"Database will use SQLite fallback if PostgreSQL not available")
    print(f"Cache will use in-memory fallback if Redis not available")
    print("-" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")