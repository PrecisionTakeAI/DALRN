"""
Centralized Configuration Management for DALRN
Replaces all 175 hardcoded values with environment-based configuration
"""
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any
import os


class Settings(BaseSettings):
    """Production-ready configuration with all hardcoded values externalized"""

    # === SERVICE CONFIGURATION ===
    # API Settings
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_workers: int = int(os.getenv("API_WORKERS", "4"))
    api_reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"

    # Service Ports (replacing hardcoded 8000, 8080, etc)
    gateway_port: int = int(os.getenv("GATEWAY_PORT", "8000"))
    search_port: int = int(os.getenv("SEARCH_PORT", "8100"))
    fhe_port: int = int(os.getenv("FHE_PORT", "8200"))
    negotiation_port: int = int(os.getenv("NEGOTIATION_PORT", "8300"))
    fl_port: int = int(os.getenv("FL_PORT", "8400"))
    agents_port: int = int(os.getenv("AGENTS_PORT", "8500"))

    # === DATABASE CONFIGURATION ===
    # PostgreSQL (replacing hardcoded localhost:5432)
    database_driver: str = os.getenv("DATABASE_DRIVER", "postgresql")
    database_host: str = os.getenv("DATABASE_HOST", "localhost")
    database_port: int = int(os.getenv("DATABASE_PORT", "5432"))
    database_name: str = os.getenv("DATABASE_NAME", "dalrn")
    database_user: str = os.getenv("DATABASE_USER", "dalrn_user")
    database_password: str = os.getenv("DATABASE_PASSWORD", "secure_password_change_me")
    database_pool_size: int = int(os.getenv("DATABASE_POOL_SIZE", "20"))
    database_max_overflow: int = int(os.getenv("DATABASE_MAX_OVERFLOW", "40"))

    @property
    def database_url(self) -> str:
        """Construct database URL from components"""
        if self.database_driver == "sqlite":
            return f"sqlite:///{self.database_name}.db"
        return f"{self.database_driver}://{self.database_user}:{self.database_password}@{self.database_host}:{self.database_port}/{self.database_name}"

    # === REDIS CONFIGURATION ===
    # Redis Cache (replacing hardcoded localhost:6379)
    redis_host: str = os.getenv("REDIS_HOST", "localhost")
    redis_port: int = int(os.getenv("REDIS_PORT", "6379"))
    redis_password: Optional[str] = os.getenv("REDIS_PASSWORD", None)
    redis_db: int = int(os.getenv("REDIS_DB", "0"))
    redis_ttl: int = int(os.getenv("REDIS_TTL", "300"))
    redis_pool_size: int = int(os.getenv("REDIS_POOL_SIZE", "50"))

    @property
    def redis_url(self) -> str:
        """Construct Redis URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # === SECURITY CONFIGURATION ===
    # JWT Settings (replacing hardcoded secrets)
    jwt_secret: str = os.getenv("JWT_SECRET", "CHANGE_THIS_SECRET_KEY_IN_PRODUCTION")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expiry_minutes: int = int(os.getenv("JWT_EXPIRY_MINUTES", "30"))
    jwt_refresh_expiry_days: int = int(os.getenv("JWT_REFRESH_EXPIRY_DAYS", "7"))

    # API Keys
    api_key: Optional[str] = os.getenv("API_KEY", None)
    admin_api_key: Optional[str] = os.getenv("ADMIN_API_KEY", None)

    # === BLOCKCHAIN CONFIGURATION ===
    # Web3/Ethereum (replacing hardcoded localhost:8545)
    blockchain_provider: str = os.getenv("BLOCKCHAIN_PROVIDER", "http://localhost:8545")
    blockchain_chain_id: int = int(os.getenv("BLOCKCHAIN_CHAIN_ID", "1337"))
    contract_address: Optional[str] = os.getenv("CONTRACT_ADDRESS", "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512")
    private_key: Optional[str] = os.getenv("PRIVATE_KEY", None)
    gas_limit: int = int(os.getenv("GAS_LIMIT", "8000000"))
    gas_price_gwei: int = int(os.getenv("GAS_PRICE_GWEI", "30"))

    # === IPFS CONFIGURATION ===
    # IPFS (replacing hardcoded localhost:5001)
    ipfs_api_host: str = os.getenv("IPFS_API_HOST", "localhost")
    ipfs_api_port: int = int(os.getenv("IPFS_API_PORT", "5001"))
    ipfs_gateway_host: str = os.getenv("IPFS_GATEWAY_HOST", "localhost")
    ipfs_gateway_port: int = int(os.getenv("IPFS_GATEWAY_PORT", "8080"))

    @property
    def ipfs_api_url(self) -> str:
        """Construct IPFS API URL"""
        return f"http://{self.ipfs_api_host}:{self.ipfs_api_port}"

    @property
    def ipfs_gateway_url(self) -> str:
        """Construct IPFS Gateway URL"""
        return f"http://{self.ipfs_gateway_host}:{self.ipfs_gateway_port}"

    # === PERFORMANCE CONFIGURATION ===
    # Caching
    cache_enabled: bool = os.getenv("CACHE_ENABLED", "true").lower() == "true"
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "300"))

    # Rate Limiting
    rate_limit_requests_per_minute: int = int(os.getenv("RATE_LIMIT_REQUESTS_PER_MINUTE", "60"))
    rate_limit_burst_size: int = int(os.getenv("RATE_LIMIT_BURST_SIZE", "10"))

    # Timeouts
    request_timeout_seconds: int = int(os.getenv("REQUEST_TIMEOUT_SECONDS", "30"))
    database_timeout_seconds: int = int(os.getenv("DATABASE_TIMEOUT_SECONDS", "10"))
    blockchain_timeout_seconds: int = int(os.getenv("BLOCKCHAIN_TIMEOUT_SECONDS", "60"))

    # Resource Limits
    max_request_size_mb: int = int(os.getenv("MAX_REQUEST_SIZE_MB", "10"))
    max_upload_size_mb: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    max_connections: int = int(os.getenv("MAX_CONNECTIONS", "1000"))

    # === ML/AI CONFIGURATION ===
    # Vector Search
    vector_dimension: int = int(os.getenv("VECTOR_DIMENSION", "768"))
    hnsw_m: int = int(os.getenv("HNSW_M", "32"))
    hnsw_ef_construction: int = int(os.getenv("HNSW_EF_CONSTRUCTION", "200"))
    hnsw_ef_search: int = int(os.getenv("HNSW_EF_SEARCH", "128"))

    # Federated Learning
    epsilon_budget: float = float(os.getenv("EPSILON_BUDGET", "4.0"))
    delta: float = float(os.getenv("DELTA", "1e-5"))

    # === MONITORING CONFIGURATION ===
    # Prometheus
    prometheus_enabled: bool = os.getenv("PROMETHEUS_ENABLED", "true").lower() == "true"
    prometheus_port: int = int(os.getenv("PROMETHEUS_PORT", "9090"))

    # Grafana
    grafana_enabled: bool = os.getenv("GRAFANA_ENABLED", "true").lower() == "true"
    grafana_port: int = int(os.getenv("GRAFANA_PORT", "3000"))

    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = os.getenv("LOG_FORMAT", "json")
    log_file: Optional[str] = os.getenv("LOG_FILE", None)

    # === PODP CONFIGURATION ===
    podp_enabled: bool = os.getenv("PODP_ENABLED", "true").lower() == "true"
    podp_merkle_algo: str = os.getenv("PODP_MERKLE_ALGO", "keccak256")
    podp_receipt_ttl: int = int(os.getenv("PODP_RECEIPT_TTL", "86400"))

    # === ENVIRONMENT ===
    environment: str = os.getenv("ENVIRONMENT", "development")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    testing: bool = os.getenv("TESTING", "false").lower() == "true"

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

        # Allow extra fields for forward compatibility
        extra = "allow"


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create settings singleton"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Export for easy import
settings = get_settings()


if __name__ == "__main__":
    # Test configuration loading
    config = get_settings()
    print(f"Environment: {config.environment}")
    print(f"Database URL: {config.database_url}")
    print(f"Redis URL: {config.redis_url}")
    print(f"IPFS API URL: {config.ipfs_api_url}")
    print(f"Blockchain Provider: {config.blockchain_provider}")
    print(f"JWT Secret: {'SET' if config.jwt_secret != 'CHANGE_THIS_SECRET_KEY_IN_PRODUCTION' else 'NOT SET'}")
    print(f"Contract Address: {config.contract_address}")
    print(f"Cache Enabled: {config.cache_enabled}")
    print(f"PoDP Enabled: {config.podp_enabled}")
    print(f"Total settings: {len(config.dict())}")