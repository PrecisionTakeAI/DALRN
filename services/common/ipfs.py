"""
IPFS client with proper error handling and fallback mechanisms
Provides resilient storage for receipts and dispute data
"""
import os
import json
import base64
import logging
from typing import Optional, Dict, Any
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

logger = logging.getLogger(__name__)

# Configuration
IPFS_API = os.getenv("IPFS_API", "http://127.0.0.1:5001")
IPFS_GATEWAY = os.getenv("IPFS_GATEWAY", "http://127.0.0.1:8080")
IPFS_TIMEOUT = int(os.getenv("IPFS_TIMEOUT", "10"))  # seconds
IPFS_RETRY_COUNT = int(os.getenv("IPFS_RETRY_COUNT", "3"))
USE_LOCAL_FALLBACK = os.getenv("IPFS_USE_LOCAL_FALLBACK", "true").lower() == "true"

# Local fallback storage for when IPFS is unavailable
_local_storage: Dict[str, Any] = {}

class IPFSError(Exception):
    """Base exception for IPFS operations"""
    pass

class IPFSConnectionError(IPFSError):
    """Raised when IPFS daemon is not accessible"""
    pass

class IPFSStorageError(IPFSError):
    """Raised when storage operation fails"""
    pass

def check_ipfs_health() -> bool:
    """Check if IPFS daemon is accessible and healthy"""
    try:
        response = requests.get(
            f"{IPFS_API}/api/v0/version",
            timeout=2
        )
        return response.status_code == 200
    except (RequestException, Timeout, ConnectionError):
        return False

def put_json(obj: Any, fallback_to_local: bool = True) -> str:
    """
    Store JSON object in IPFS with proper error handling

    Args:
        obj: JSON-serializable object to store
        fallback_to_local: Whether to use local storage if IPFS fails

    Returns:
        IPFS URI (ipfs://...) or local URI (local://...) if fallback used

    Raises:
        IPFSStorageError: If storage fails and no fallback available
    """
    # Validate input
    if obj is None:
        raise ValueError("Cannot store None object in IPFS")

    # Serialize object
    try:
        data = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Object is not JSON serializable: {e}")

    # Try IPFS storage with retries
    for attempt in range(IPFS_RETRY_COUNT):
        try:
            files = {"file": ("receipt_chain.json", data, "application/json")}
            response = requests.post(
                f"{IPFS_API}/api/v0/add",
                files=files,
                timeout=IPFS_TIMEOUT
            )

            if response.status_code == 200:
                result = response.json()
                cid = result.get("Hash")
                if cid:
                    logger.info(f"Successfully stored object in IPFS: {cid}")
                    return f"ipfs://{cid}/receipt_chain.json"
                else:
                    logger.error(f"Invalid IPFS response: {result}")

        except ConnectionError as e:
            logger.warning(f"IPFS connection failed (attempt {attempt + 1}/{IPFS_RETRY_COUNT}): {e}")
            if attempt == IPFS_RETRY_COUNT - 1:
                if fallback_to_local and USE_LOCAL_FALLBACK:
                    return _store_local_fallback(data, obj)
                raise IPFSConnectionError(f"IPFS daemon not accessible at {IPFS_API}")

        except Timeout as e:
            logger.warning(f"IPFS request timed out (attempt {attempt + 1}/{IPFS_RETRY_COUNT}): {e}")
            if attempt == IPFS_RETRY_COUNT - 1:
                if fallback_to_local and USE_LOCAL_FALLBACK:
                    return _store_local_fallback(data, obj)
                raise IPFSStorageError(f"IPFS storage timed out after {IPFS_TIMEOUT}s")

        except RequestException as e:
            logger.error(f"IPFS storage failed: {e}")
            if fallback_to_local and USE_LOCAL_FALLBACK:
                return _store_local_fallback(data, obj)
            raise IPFSStorageError(f"Failed to store in IPFS: {e}")

    # All retries failed
    if fallback_to_local and USE_LOCAL_FALLBACK:
        return _store_local_fallback(data, obj)
    raise IPFSStorageError(f"Failed to store in IPFS after {IPFS_RETRY_COUNT} attempts")

def get_json(cid_uri: str, fallback_to_local: bool = True) -> Optional[Any]:
    """
    Retrieve JSON object from IPFS with proper error handling

    Args:
        cid_uri: IPFS URI (ipfs://...) or local URI (local://...)
        fallback_to_local: Whether to check local storage if IPFS fails

    Returns:
        Deserialized JSON object or None if not found

    Raises:
        IPFSError: If retrieval fails and no fallback available
    """
    if not cid_uri:
        raise ValueError("CID URI cannot be empty")

    # Handle local storage URIs
    if cid_uri.startswith("local://"):
        return _get_local_fallback(cid_uri)

    # Validate IPFS URI format
    if not cid_uri.startswith("ipfs://"):
        raise ValueError(f"Invalid URI format: {cid_uri}")

    # Extract CID from URI
    cid_parts = cid_uri[len("ipfs://"):].split("/")
    cid = cid_parts[0]

    if not cid:
        raise ValueError(f"Invalid IPFS URI: {cid_uri}")

    # Try IPFS retrieval with retries
    for attempt in range(IPFS_RETRY_COUNT):
        try:
            response = requests.get(
                f"{IPFS_API}/api/v0/cat",
                params={"arg": cid},
                timeout=IPFS_TIMEOUT
            )

            if response.status_code == 200:
                try:
                    return json.loads(response.text)
                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON from IPFS: {e}")
                    raise IPFSError(f"Retrieved data is not valid JSON: {e}")

            elif response.status_code == 404:
                logger.warning(f"CID not found in IPFS: {cid}")
                # Check local fallback
                if fallback_to_local and USE_LOCAL_FALLBACK:
                    local_uri = f"local://{cid}"
                    local_data = _get_local_fallback(local_uri)
                    if local_data is not None:
                        return local_data
                return None

        except ConnectionError as e:
            logger.warning(f"IPFS connection failed (attempt {attempt + 1}/{IPFS_RETRY_COUNT}): {e}")
            if attempt == IPFS_RETRY_COUNT - 1:
                if fallback_to_local and USE_LOCAL_FALLBACK:
                    local_uri = f"local://{cid}"
                    local_data = _get_local_fallback(local_uri)
                    if local_data is not None:
                        return local_data
                raise IPFSConnectionError(f"IPFS daemon not accessible at {IPFS_API}")

        except Timeout as e:
            logger.warning(f"IPFS request timed out (attempt {attempt + 1}/{IPFS_RETRY_COUNT}): {e}")
            if attempt == IPFS_RETRY_COUNT - 1:
                if fallback_to_local and USE_LOCAL_FALLBACK:
                    local_uri = f"local://{cid}"
                    local_data = _get_local_fallback(local_uri)
                    if local_data is not None:
                        return local_data
                raise IPFSError(f"IPFS retrieval timed out after {IPFS_TIMEOUT}s")

        except RequestException as e:
            logger.error(f"IPFS retrieval failed: {e}")
            if fallback_to_local and USE_LOCAL_FALLBACK:
                local_uri = f"local://{cid}"
                local_data = _get_local_fallback(local_uri)
                if local_data is not None:
                    return local_data
            raise IPFSError(f"Failed to retrieve from IPFS: {e}")

    # All retries failed
    raise IPFSError(f"Failed to retrieve from IPFS after {IPFS_RETRY_COUNT} attempts")

def _store_local_fallback(data: bytes, obj: Any) -> str:
    """Store data locally when IPFS is unavailable"""
    import hashlib

    # Generate a hash-based ID for local storage
    hash_id = hashlib.sha256(data).hexdigest()[:32]
    local_id = f"local_{hash_id}"

    # Store in memory (in production, this should be persistent storage)
    _local_storage[local_id] = obj

    logger.warning(f"IPFS unavailable, using local storage: {local_id}")
    return f"local://{local_id}"

def _get_local_fallback(uri: str) -> Optional[Any]:
    """Retrieve data from local fallback storage"""
    if not uri.startswith("local://"):
        return None

    local_id = uri[len("local://"):]
    data = _local_storage.get(local_id)

    if data is not None:
        logger.info(f"Retrieved from local fallback: {local_id}")

    return data

def clear_local_storage():
    """Clear local fallback storage (for testing)"""
    global _local_storage
    _local_storage = {}
    logger.info("Local fallback storage cleared")

# Backward compatibility with existing code
def put_receipt_chain(chain: Dict) -> str:
    """Store receipt chain in IPFS with enhanced error handling"""
    return put_json(chain, fallback_to_local=True)

def get_receipt_chain(cid_uri: str) -> Optional[Dict]:
    """Retrieve receipt chain from IPFS with enhanced error handling"""
    return get_json(cid_uri, fallback_to_local=True)

# Health check endpoint
def get_ipfs_status() -> Dict[str, Any]:
    """Get IPFS service status and statistics"""
    status = {
        "available": False,
        "api_endpoint": IPFS_API,
        "gateway_endpoint": IPFS_GATEWAY,
        "local_fallback_enabled": USE_LOCAL_FALLBACK,
        "local_storage_count": len(_local_storage)
    }

    try:
        if check_ipfs_health():
            status["available"] = True
            # Try to get additional stats
            try:
                response = requests.get(
                    f"{IPFS_API}/api/v0/stats/repo",
                    timeout=2
                )
                if response.status_code == 200:
                    stats = response.json()
                    status["repo_size"] = stats.get("RepoSize", 0)
                    status["num_objects"] = stats.get("NumObjects", 0)
            except:
                pass  # Stats are optional
    except:
        pass  # Health check already handles errors

    return status