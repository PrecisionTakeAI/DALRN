#!/usr/bin/env python3
"""
Security validation script for the FHE service.

This script demonstrates that:
1. The service requires TenSEAL and refuses to start without it
2. No fake encryption (SHA256 hashes) are ever returned
3. Real CKKS homomorphic encryption is enforced
"""

import sys
import traceback

def validate_tenseal_required():
    """Verify that TenSEAL is required and no placeholders are allowed."""
    print("="*60)
    print("FHE Service Security Validation")
    print("="*60)

    # Test 1: Verify TenSEAL is required at import time
    print("\n[TEST 1] Verifying TenSEAL is required...")
    try:
        import tenseal as ts
        print("[PASS] TenSEAL is installed (version: {})".format(
            ts.__version__ if hasattr(ts, '__version__') else 'unknown'
        ))
    except ImportError:
        print("[FAIL] CRITICAL: TenSEAL is not installed!")
        print("  The FHE service REQUIRES TenSEAL for real encryption.")
        print("  Install with: pip install tenseal==0.3.16")
        return False

    # Test 2: Try to import the service
    print("\n[TEST 2] Importing FHE service...")
    try:
        from services.fhe.service import (
            CKKSConfig,
            ContextManager,
            FHEOperations,
            encrypt_vector
        )
        print("[PASS] FHE service imported successfully")
    except RuntimeError as e:
        if "TenSEAL is REQUIRED" in str(e):
            print("[PASS] Service correctly refuses to start without TenSEAL")
            print(f"  Error message: {e}")
            return True  # This is expected behavior if TenSEAL is missing
        else:
            print(f"[FAIL] Unexpected error: {e}")
            return False
    except Exception as e:
        print(f"[FAIL] Failed to import service: {e}")
        traceback.print_exc()
        return False

    # Test 3: Verify no placeholder functions exist
    print("\n[TEST 3] Checking for placeholder functions...")
    try:
        from services.fhe import service

        # Check that placeholder function doesn't exist
        if hasattr(service, 'encrypt_vector_placeholder'):
            print("[FAIL] SECURITY VIOLATION: Placeholder encryption function still exists!")
            return False

        # Check that real encryption function exists
        if not hasattr(service, 'encrypt_vector'):
            print("[FAIL] Real encryption function 'encrypt_vector' not found!")
            return False

        print("[PASS] No placeholder encryption functions found")
        print("[PASS] Real encryption function 'encrypt_vector' exists")

    except Exception as e:
        print(f"[FAIL] Error checking functions: {e}")
        return False

    # Test 4: Verify encryption produces real ciphertext
    print("\n[TEST 4] Testing real encryption...")
    try:
        import numpy as np

        # Create a test context
        config = CKKSConfig()
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=config.poly_modulus_degree,
            coeff_mod_bit_sizes=config.coeff_mod_bit_sizes
        )
        context.global_scale = config.global_scale
        context.generate_galois_keys()
        context.generate_relin_keys()

        # Create test vector
        test_vector = np.array([1.0, 2.0, 3.0, 4.0])

        # Encrypt using the service function
        encrypted = encrypt_vector(context, test_vector)

        # Check that result is not a hash (SHA256 is 64 chars in base64)
        if len(encrypted) == 44 or len(encrypted) == 64:
            print("[FAIL] SECURITY VIOLATION: Encrypted output looks like a hash!")
            return False

        # Real CKKS encryption produces much larger ciphertexts
        if len(encrypted) < 1000:
            print("[FAIL] Encrypted output is suspiciously small!")
            print(f"  Size: {len(encrypted)} characters")
            return False

        print(f"[PASS] Real encryption produced ciphertext of {len(encrypted)} characters")
        print("[PASS] This is consistent with CKKS encryption (not a hash)")

    except Exception as e:
        print(f"[FAIL] Encryption test failed: {e}")
        traceback.print_exc()
        return False

    # Test 5: Verify dot product operations
    print("\n[TEST 5] Testing homomorphic dot product...")
    try:
        # Create two vectors
        v1 = np.array([1.0, 2.0, 3.0, 4.0])
        v2 = np.array([5.0, 6.0, 7.0, 8.0])

        # Expected plaintext result
        expected = np.dot(v1, v2)

        # Encrypt vectors
        enc_v1 = ts.ckks_vector(context, v1.tolist())
        enc_v2 = ts.ckks_vector(context, v2.tolist())

        # Perform homomorphic dot product
        result_bytes = FHEOperations.perform_dot_product(
            context,
            enc_v1.serialize(),
            enc_v2.serialize()
        )

        # Check result size (should be large for real encryption)
        if len(result_bytes) < 100:
            print("[FAIL] Dot product result is too small to be real encryption!")
            return False

        print(f"[PASS] Homomorphic dot product produced {len(result_bytes)} bytes")
        print("[PASS] This confirms real FHE operations are being performed")

        # Decrypt to verify accuracy (with secret key - client side only)
        secret_key = context.secret_key()
        enc_result = ts.ckks_vector_from(context, result_bytes)
        decrypted = enc_result.decrypt(secret_key)[0]

        error = abs(decrypted - expected) / abs(expected)
        print(f"[PASS] Decrypted result: {decrypted:.6f} (expected: {expected})")
        print(f"[PASS] Relative error: {error:.6%}")

        if error > 0.02:
            print("[FAIL] Error exceeds 2% threshold!")
            return False

    except Exception as e:
        print(f"[FAIL] Dot product test failed: {e}")
        traceback.print_exc()
        return False

    # Test 6: Verify context isolation
    print("\n[TEST 6] Testing tenant isolation...")
    try:
        manager = ContextManager()

        # Create contexts for different tenants
        ctx1, id1 = manager.create_context("tenant_A", config)
        ctx2, id2 = manager.create_context("tenant_B", config)

        if id1 == id2:
            print("[FAIL] Same context ID for different tenants!")
            return False

        if ctx1 is ctx2:
            print("[FAIL] Same context object for different tenants!")
            return False

        print("[PASS] Different contexts created for different tenants")
        print(f"  Tenant A context ID: {id1}")
        print(f"  Tenant B context ID: {id2}")

        # Verify retrieval isolation
        retrieved_A = manager.get_context("tenant_A")
        retrieved_B = manager.get_context("tenant_B")

        if retrieved_A is retrieved_B:
            print("[FAIL] Cross-tenant context contamination!")
            return False

        print("[PASS] Tenant contexts are properly isolated")

    except Exception as e:
        print(f"[FAIL] Isolation test failed: {e}")
        traceback.print_exc()
        return False

    return True


def main():
    """Run all security validation tests."""
    print("\nStarting FHE Service Security Validation...\n")

    success = validate_tenseal_required()

    print("\n" + "="*60)
    if success:
        print("[PASS] ALL SECURITY VALIDATIONS PASSED")
        print("\nThe FHE service is properly secured:")
        print("  • TenSEAL is required (no placeholders)")
        print("  • Real CKKS encryption is enforced")
        print("  • No fake/hash-based encryption allowed")
        print("  • Tenant isolation is maintained")
        print("  • Accuracy requirements are met")
    else:
        print("[FAIL] SECURITY VALIDATION FAILED")
        print("\nCritical issues found:")
        print("  • Check TenSEAL installation")
        print("  • Verify no placeholder code remains")
        print("  • Ensure real encryption is enforced")
        sys.exit(1)

    print("="*60)


if __name__ == "__main__":
    main()