"""
DALRN Rate Limiting Module
Implements API rate limiting with token bucket algorithm and quotas
"""
import time
import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
import hashlib
import logging
from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Rate limit configuration"""
    requests_per_minute: int = 60
    requests_per_hour: int = 1000
    requests_per_day: int = 10000
    burst_size: int = 10
    cooldown_seconds: int = 60

@dataclass
class TokenBucket:
    """Token bucket for rate limiting"""
    capacity: int
    tokens: float
    refill_rate: float
    last_refill: float

    def consume(self, tokens: int = 1) -> bool:
        """Try to consume tokens from bucket"""
        self.refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def refill(self):
        """Refill tokens based on elapsed time"""
        now = time.time()
        elapsed = now - self.last_refill

        # Add tokens based on refill rate
        self.tokens = min(
            self.capacity,
            self.tokens + (elapsed * self.refill_rate)
        )
        self.last_refill = now

class RateLimiter:
    """Advanced rate limiter with multiple strategies"""

    def __init__(self):
        # Configuration per tier
        self.tiers = {
            "free": RateLimitConfig(
                requests_per_minute=30,
                requests_per_hour=500,
                requests_per_day=5000,
                burst_size=5
            ),
            "basic": RateLimitConfig(
                requests_per_minute=60,
                requests_per_hour=1000,
                requests_per_day=10000,
                burst_size=10
            ),
            "premium": RateLimitConfig(
                requests_per_minute=120,
                requests_per_hour=5000,
                requests_per_day=50000,
                burst_size=20
            ),
            "enterprise": RateLimitConfig(
                requests_per_minute=600,
                requests_per_hour=20000,
                requests_per_day=200000,
                burst_size=50
            )
        }

        # Storage for rate limit tracking
        self.minute_buckets: Dict[str, TokenBucket] = {}
        self.hour_windows: Dict[str, deque] = defaultdict(deque)
        self.day_windows: Dict[str, deque] = defaultdict(deque)
        self.blocked_clients: Dict[str, float] = {}

        # Quota tracking
        self.quotas: Dict[str, Dict[str, int]] = defaultdict(lambda: {
            "disputes_per_day": 100,
            "searches_per_day": 1000,
            "ml_operations_per_day": 50,
            "storage_mb": 100
        })

        self.quota_usage: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def get_client_id(self, request: Request, user_id: Optional[str] = None) -> str:
        """Get unique client identifier"""
        if user_id:
            return f"user:{user_id}"

        # Use IP address as fallback
        client_ip = request.client.host if request.client else "unknown"

        # Include user agent for better identification
        user_agent = request.headers.get("User-Agent", "")
        identifier = f"{client_ip}:{user_agent}"

        # Hash for privacy
        return hashlib.sha256(identifier.encode()).hexdigest()[:16]

    def get_tier(self, client_id: str) -> str:
        """Get tier for client"""
        # In production, query from database
        if "user:usr_admin" in client_id:
            return "enterprise"
        elif "user:" in client_id:
            return "basic"
        else:
            return "free"

    def check_rate_limit(
        self,
        client_id: str,
        endpoint: str,
        tier: str = "free"
    ) -> Tuple[bool, Optional[Dict]]:
        """Check if request is within rate limits"""

        # Check if client is blocked
        if client_id in self.blocked_clients:
            blocked_until = self.blocked_clients[client_id]
            if time.time() < blocked_until:
                remaining = int(blocked_until - time.time())
                return False, {
                    "error": "Rate limit exceeded - Client blocked",
                    "retry_after": remaining
                }
            else:
                del self.blocked_clients[client_id]

        config = self.tiers[tier]
        now = time.time()

        # Check minute rate limit (token bucket)
        if client_id not in self.minute_buckets:
            self.minute_buckets[client_id] = TokenBucket(
                capacity=config.burst_size,
                tokens=config.burst_size,
                refill_rate=config.requests_per_minute / 60,
                last_refill=now
            )

        bucket = self.minute_buckets[client_id]
        if not bucket.consume():
            # Block client for cooldown period
            self.blocked_clients[client_id] = now + config.cooldown_seconds
            return False, {
                "error": "Burst rate limit exceeded",
                "limit": config.burst_size,
                "retry_after": config.cooldown_seconds
            }

        # Check hourly rate limit (sliding window)
        hour_window = self.hour_windows[client_id]
        hour_cutoff = now - 3600

        # Remove old entries
        while hour_window and hour_window[0] < hour_cutoff:
            hour_window.popleft()

        if len(hour_window) >= config.requests_per_hour:
            return False, {
                "error": "Hourly rate limit exceeded",
                "limit": config.requests_per_hour,
                "window": "1 hour",
                "retry_after": int(hour_window[0] + 3600 - now)
            }

        hour_window.append(now)

        # Check daily rate limit
        day_window = self.day_windows[client_id]
        day_cutoff = now - 86400

        # Remove old entries
        while day_window and day_window[0] < day_cutoff:
            day_window.popleft()

        if len(day_window) >= config.requests_per_day:
            return False, {
                "error": "Daily rate limit exceeded",
                "limit": config.requests_per_day,
                "window": "24 hours",
                "retry_after": int(day_window[0] + 86400 - now)
            }

        day_window.append(now)

        # Calculate remaining limits
        remaining = {
            "minute": int(bucket.tokens),
            "hour": config.requests_per_hour - len(hour_window),
            "day": config.requests_per_day - len(day_window)
        }

        return True, {
            "tier": tier,
            "remaining": remaining,
            "limits": {
                "burst": config.burst_size,
                "minute": config.requests_per_minute,
                "hour": config.requests_per_hour,
                "day": config.requests_per_day
            }
        }

    def check_quota(
        self,
        client_id: str,
        resource: str,
        amount: int = 1
    ) -> Tuple[bool, Optional[Dict]]:
        """Check if client has quota for resource"""

        client_quotas = self.quotas[client_id]
        client_usage = self.quota_usage[client_id]

        if resource not in client_quotas:
            return False, {"error": f"Unknown resource: {resource}"}

        quota = client_quotas[resource]
        current_usage = client_usage[resource]

        if current_usage + amount > quota:
            return False, {
                "error": f"Quota exceeded for {resource}",
                "quota": quota,
                "used": current_usage,
                "remaining": max(0, quota - current_usage)
            }

        # Update usage
        client_usage[resource] += amount

        return True, {
            "resource": resource,
            "quota": quota,
            "used": current_usage + amount,
            "remaining": quota - (current_usage + amount)
        }

    def reset_quota(self, client_id: str, resource: Optional[str] = None):
        """Reset quota usage for client"""
        if resource:
            self.quota_usage[client_id][resource] = 0
        else:
            self.quota_usage[client_id].clear()

    def get_stats(self, client_id: str) -> Dict:
        """Get rate limit and quota statistics for client"""
        tier = self.get_tier(client_id)
        config = self.tiers[tier]

        # Current rate limit status
        bucket = self.minute_buckets.get(client_id)
        hour_count = len(self.hour_windows.get(client_id, []))
        day_count = len(self.day_windows.get(client_id, []))

        # Quota status
        quotas = self.quotas[client_id]
        usage = self.quota_usage[client_id]

        return {
            "tier": tier,
            "rate_limits": {
                "burst_tokens": int(bucket.tokens) if bucket else config.burst_size,
                "hour_used": hour_count,
                "hour_limit": config.requests_per_hour,
                "day_used": day_count,
                "day_limit": config.requests_per_day
            },
            "quotas": {
                resource: {
                    "limit": quota,
                    "used": usage.get(resource, 0),
                    "remaining": quota - usage.get(resource, 0)
                }
                for resource, quota in quotas.items()
            },
            "blocked": client_id in self.blocked_clients
        }

# Singleton instance
rate_limiter = RateLimiter()

# FastAPI middleware
async def rate_limit_middleware(request: Request, call_next):
    """Rate limiting middleware for FastAPI"""

    # Skip rate limiting for health checks
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)

    # Get client ID
    user_id = request.headers.get("X-User-ID")
    client_id = rate_limiter.get_client_id(request, user_id)

    # Get tier
    tier = rate_limiter.get_tier(client_id)

    # Check rate limit
    allowed, info = rate_limiter.check_rate_limit(
        client_id,
        request.url.path,
        tier
    )

    if not allowed:
        # Return 429 Too Many Requests
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": info.get("error", "Rate limit exceeded"),
                "retry_after": info.get("retry_after", 60)
            },
            headers={
                "X-RateLimit-Limit": str(info.get("limit", 60)),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(time.time() + info.get("retry_after", 60))),
                "Retry-After": str(info.get("retry_after", 60))
            }
        )

    # Add rate limit headers to response
    response = await call_next(request)

    if info:
        response.headers["X-RateLimit-Tier"] = tier
        response.headers["X-RateLimit-Limit-Minute"] = str(info["limits"]["minute"])
        response.headers["X-RateLimit-Remaining-Hour"] = str(info["remaining"]["hour"])
        response.headers["X-RateLimit-Remaining-Day"] = str(info["remaining"]["day"])

    return response

# Quota check decorator
def check_quota(resource: str, amount: int = 1):
    """Decorator to check quota before executing function"""
    def decorator(func):
        async def wrapper(request: Request, *args, **kwargs):
            user_id = request.headers.get("X-User-ID")
            client_id = rate_limiter.get_client_id(request, user_id)

            allowed, info = rate_limiter.check_quota(client_id, resource, amount)

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_402_PAYMENT_REQUIRED,
                    detail=info
                )

            return await func(request, *args, **kwargs)
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test rate limiter
    limiter = RateLimiter()

    # Simulate requests
    client_id = "test_client_123"

    print("Testing rate limiter...")
    for i in range(15):
        allowed, info = limiter.check_rate_limit(client_id, "/api/test", "free")
        print(f"Request {i+1}: {'Allowed' if allowed else 'Blocked'}")
        if info:
            print(f"  Info: {info}")

        time.sleep(0.1)

    print("\nTesting quota system...")
    for i in range(5):
        allowed, info = limiter.check_quota(client_id, "disputes_per_day", 25)
        print(f"Quota check {i+1}: {'Allowed' if allowed else 'Blocked'}")
        print(f"  Info: {info}")

    print(f"\nClient stats: {limiter.get_stats(client_id)}")