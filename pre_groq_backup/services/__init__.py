"""
DALRN Services Package
"""
import sys
from pathlib import Path

# Add parent directory to Python path for imports
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))