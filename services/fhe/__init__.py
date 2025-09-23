"""
TenSEAL CKKS FHE Service for DALRN.

This module provides privacy-preserving dot-product operations using
fully homomorphic encryption.
"""

from services.fhe.service import (
    app,
    CKKSConfig,
    ContextManager,
    FHEOperations,
    create_test_vectors,
    encrypt_vector  # Real encryption function, no placeholders
)

__all__ = [
    'app',
    'CKKSConfig',
    'ContextManager',
    'FHEOperations',
    'create_test_vectors',
    'encrypt_vector'  # Real encryption only
]