"""Configuration management for DALRN Gateway"""
import os
from typing import Optional
from pydantic import BaseSettings, Field
from enum import Enum

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class GatewayConfig(BaseSettings):
    """Gateway configuration settings"""

    # Application settings
    app_name: str = Field(default="DALRN Gateway", description="Application name")
    app_version: str = Field(default="2.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="Environment")
    debug: bool = Field(default=False, description="Debug mode")

    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of workers")

    # CORS settings
    cors_origins: str = Field(default="*", description="Allowed CORS origins (comma-separated)")
    cors_credentials: bool = Field(default=True, description="Allow credentials in CORS")

    # Redis settings
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_db: int = Field(default=0, description="Redis database")
    redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_ttl: int = Field(default=86400, description="Default TTL in seconds")

    # PostgreSQL settings
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="dalrn", description="PostgreSQL database")
    postgres_user: str = Field(default="dalrn_user", description="PostgreSQL user")
    postgres_password: str = Field(default="dalrn_pass", description="PostgreSQL password")

    # IPFS settings
    ipfs_host: str = Field(default="localhost", description="IPFS host")
    ipfs_api_port: int = Field(default=5001, description="IPFS API port")
    ipfs_gateway_port: int = Field(default=8080, description="IPFS gateway port")

    # Blockchain settings
    web3_provider_url: str = Field(default="http://localhost:8545", description="Web3 provider URL")
    anchor_contract_address: Optional[str] = Field(default=None, description="Anchor contract address")
    chain_id: int = Field(default=31337, description="Blockchain chain ID")

    # Service URLs
    fhe_service_url: str = Field(default="http://localhost:8200", description="FHE service URL")
    search_service_url: str = Field(default="http://localhost:8100", description="Search service URL")
    negotiation_service_url: str = Field(default="http://localhost:8300", description="Negotiation service URL")
    fl_service_url: str = Field(default="http://localhost:8400", description="FL service URL")
    agents_service_url: str = Field(default="http://localhost:8500", description="Agents service URL")

    # Rate limiting
    rate_limit_default: int = Field(default=100, description="Default rate limit per minute")
    rate_limit_authenticated: int = Field(default=500, description="Authenticated rate limit per minute")
    rate_limit_premium: int = Field(default=2000, description="Premium rate limit per minute")
    rate_limit_window_seconds: int = Field(default=60, description="Rate limit window in seconds")

    # Privacy budget
    epsilon_total_budget: float = Field(default=4.0, description="Total epsilon budget")
    epsilon_search_cost: float = Field(default=0.1, description="Epsilon cost for search")
    epsilon_negotiation_cost: float = Field(default=0.2, description="Epsilon cost for negotiation")
    epsilon_fhe_cost: float = Field(default=0.3, description="Epsilon cost for FHE")
    epsilon_agent_cost: float = Field(default=0.05, description="Epsilon cost for agent routing")

    # Circuit breaker settings
    circuit_failure_threshold: int = Field(default=5, description="Circuit breaker failure threshold")
    circuit_timeout_seconds: float = Field(default=60.0, description="Circuit breaker timeout")
    circuit_half_open_timeout: float = Field(default=30.0, description="Circuit breaker half-open timeout")

    # PoDP settings
    podp_enabled: bool = Field(default=True, description="Enable PoDP")
    podp_merkle_algo: str = Field(default="keccak256", description="Merkle tree algorithm")
    podp_receipt_ttl: int = Field(default=86400, description="Receipt TTL in seconds")

    # Monitoring
    metrics_enabled: bool = Field(default=True, description="Enable Prometheus metrics")
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")

    # Logging
    log_level: str = Field(default="INFO", description="Log level")
    log_format: str = Field(default="json", description="Log format (json or text)")
    log_redact_pii: bool = Field(default=True, description="Redact PII in logs")

    # Security
    jwt_secret_key: Optional[str] = Field(default=None, description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_expiration_minutes: int = Field(default=30, description="JWT expiration in minutes")

    # Feature flags
    enable_soan_routing: bool = Field(default=True, description="Enable SOAN network routing")
    enable_auto_anchoring: bool = Field(default=True, description="Enable automatic anchoring")
    enable_evidence_validation: bool = Field(default=True, description="Enable evidence validation")

    class Config:
        """Pydantic config"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    def get_database_url(self) -> str:
        """Get PostgreSQL database URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def get_redis_url(self) -> str:
        """Get Redis URL"""
        password_part = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{password_part}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    def get_ipfs_api_url(self) -> str:
        """Get IPFS API URL"""
        return f"http://{self.ipfs_host}:{self.ipfs_api_port}"

    def get_ipfs_gateway_url(self) -> str:
        """Get IPFS gateway URL"""
        return f"http://{self.ipfs_host}:{self.ipfs_gateway_port}"

    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == Environment.DEVELOPMENT

# Create singleton instance
config = GatewayConfig()

# Export commonly used values
APP_NAME = config.app_name
APP_VERSION = config.app_version
ENVIRONMENT = config.environment
DEBUG = config.debug

# Service health check configuration
HEALTH_CHECK_TIMEOUT = 5.0  # seconds
HEALTH_CHECK_RETRIES = 3

# Request timeout configuration
DEFAULT_REQUEST_TIMEOUT = 30.0  # seconds
LONG_REQUEST_TIMEOUT = 120.0  # seconds

# Batch processing configuration
BATCH_SIZE = 100
MAX_BATCH_SIZE = 1000

# Cache configuration
CACHE_TTL_SHORT = 300  # 5 minutes
CACHE_TTL_MEDIUM = 3600  # 1 hour
CACHE_TTL_LONG = 86400  # 24 hours

# File size limits
MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100MB
MAX_EVIDENCE_SIZE = 50 * 1024 * 1024  # 50MB

# Dispute configuration
MAX_PARTIES_PER_DISPUTE = 100
MAX_EVIDENCE_PER_DISPUTE = 1000
MAX_RECEIPTS_PER_CHAIN = 10000

# Network configuration
SOAN_NODE_COUNT = 100
SOAN_EDGE_COUNT = 6
SOAN_REWIRE_PROBABILITY = 0.1

if __name__ == "__main__":
    # Print configuration for verification
    print(f"Gateway Configuration:")
    print(f"  Environment: {config.environment}")
    print(f"  Redis: {config.redis_host}:{config.redis_port}")
    print(f"  PostgreSQL: {config.postgres_host}:{config.postgres_port}")
    print(f"  IPFS: {config.ipfs_host}:{config.ipfs_api_port}")
    print(f"  Web3: {config.web3_provider_url}")
    print(f"  PoDP: {'Enabled' if config.podp_enabled else 'Disabled'}")
    print(f"  Epsilon Budget: {config.epsilon_total_budget}")