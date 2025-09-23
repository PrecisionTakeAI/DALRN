#!/usr/bin/env python3
"""
Fix all import statements across the DALRN codebase.
Converts relative imports to absolute imports and fixes broken import paths.
"""

import os
import re
from pathlib import Path
import sys


def fix_imports_in_file(filepath: Path, project_root: Path):
    """Fix import statements in a single file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return False

    original_content = content
    file_changed = False

    # Get the module path relative to services/
    rel_path = filepath.relative_to(project_root / 'services')
    module_parts = rel_path.parts[:-1]  # Remove filename
    current_module = '.'.join(module_parts)

    # Define replacement patterns
    replacements = [
        # Fix relative imports from parent directories
        (r'from \.\.\s*import', f'from services.{module_parts[0] if module_parts else ""} import'),
        (r'from \.\.([a-zA-Z_]\w*)', r'from services.\1'),

        # Fix relative imports in same package
        (r'from \.\s+import', f'from services.{current_module} import'),
        (r'from \.([a-zA-Z_]\w*)', rf'from services.{current_module}.\1'),

        # Fix common broken imports
        (r'from auth\.jwt_auth', 'from services.auth.jwt_auth'),
        (r'from auth import', 'from services.auth import'),
        (r'^from auth$', 'from services.auth'),

        (r'from database\.models', 'from services.database.models'),
        (r'from database\.connection', 'from services.database.connection'),
        (r'from database import', 'from services.database import'),
        (r'^from database$', 'from services.database'),

        (r'from common\.podp', 'from services.common.podp'),
        (r'from common\.ipfs', 'from services.common.ipfs'),
        (r'from common\.logging', 'from services.common.logging_config'),
        (r'from common import', 'from services.common import'),
        (r'^from common$', 'from services.common'),

        (r'from cache\.connection', 'from services.cache.connection'),
        (r'from cache import', 'from services.cache import'),

        # Fix standalone module imports
        (r'^import auth$', 'import services.auth'),
        (r'^import database$', 'import services.database'),
        (r'^import common$', 'import services.common'),
        (r'^import cache$', 'import services.cache'),
    ]

    for pattern, replacement in replacements:
        new_content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        if new_content != content:
            file_changed = True
            content = new_content

    # Special case: negotiation service validation import
    if 'services/negotiation/service.py' in str(filepath):
        content = content.replace(
            'from .validation import',
            'from services.negotiation.validation import'
        )
        content = content.replace(
            'from .explanation import',
            'from services.negotiation.explanation import'
        )
        content = content.replace(
            'from .cid_generator import',
            'from services.negotiation.cid_generator import'
        )
        file_changed = True

    # Fix any double services.services patterns
    content = re.sub(r'services\.services\.', 'services.', content)

    if file_changed and content != original_content:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[FIXED] {filepath.relative_to(project_root)}")
            return True
        except Exception as e:
            print(f"[ERROR] Writing {filepath}: {e}")
            return False

    return False


def add_init_files(project_root: Path):
    """Ensure all service directories have __init__.py files."""
    services_dir = project_root / 'services'

    for dirpath, dirnames, filenames in os.walk(services_dir):
        # Skip __pycache__ and other hidden directories
        dirnames[:] = [d for d in dirnames if not d.startswith('__') and not d.startswith('.')]

        dirpath = Path(dirpath)
        init_file = dirpath / '__init__.py'

        # Check if directory contains Python files
        has_python = any(f.endswith('.py') for f in filenames)

        if has_python and not init_file.exists():
            init_file.write_text('"""Package initialization."""\n')
            print(f"[CREATED] __init__.py in: {dirpath.relative_to(project_root)}")


def fix_all_imports():
    """Fix imports in all Python files."""
    # Get project root
    project_root = Path(__file__).parent
    services_dir = project_root / 'services'

    if not services_dir.exists():
        print(f"Error: services directory not found at {services_dir}")
        return False

    print("=" * 60)
    print("DALRN Import Fixer")
    print("=" * 60)

    # First, ensure all directories have __init__.py
    print("\n1. Ensuring all packages have __init__.py...")
    add_init_files(project_root)

    # Find all Python files
    print("\n2. Finding Python files to fix...")
    python_files = list(services_dir.rglob('*.py'))
    print(f"   Found {len(python_files)} Python files")

    # Fix imports in each file
    print("\n3. Fixing imports...")
    fixed_count = 0

    for py_file in python_files:
        # Skip __pycache__ and test files
        if '__pycache__' in str(py_file) or 'test_' in py_file.name:
            continue

        if fix_imports_in_file(py_file, project_root):
            fixed_count += 1

    print("\n" + "=" * 60)
    print(f"Summary: Fixed imports in {fixed_count} files")
    print("=" * 60)

    return True


def verify_imports():
    """Verify that key services can be imported."""
    print("\n4. Verifying key imports...")

    # Add project root to path
    sys.path.insert(0, str(Path(__file__).parent))

    test_imports = [
        'services.database.connection',
        'services.cache.connection',
        'services.common.podp',
        'services.auth.jwt_auth',
    ]

    all_good = True
    for module_name in test_imports:
        try:
            __import__(module_name)
            print(f"   [OK] {module_name}")
        except ImportError as e:
            print(f"   [FAIL] {module_name}: {e}")
            all_good = False

    return all_good


if __name__ == "__main__":
    if fix_all_imports():
        if verify_imports():
            print("\n[SUCCESS] All imports fixed successfully!")
            sys.exit(0)
        else:
            print("\n[WARNING] Some imports still have issues. Manual intervention needed.")
            sys.exit(1)
    else:
        print("\n[FAILED] Import fixing failed!")
        sys.exit(1)