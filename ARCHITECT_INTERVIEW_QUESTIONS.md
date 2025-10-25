# 🏗️ Senior Python Architect Interview Questions
## FastAPI, Python & Apache Airflow

### Position: Senior Software Architect
### Focus Areas: Python, FastAPI, Apache Airflow, System Design

---

## 📋 Table of Contents

1. [Python Architecture & Design](#python-architecture--design)
2. [FastAPI - Advanced Concepts](#fastapi---advanced-concepts)
3. [Apache Airflow - Orchestration](#apache-airflow---orchestration)
4. [System Design & Scalability](#system-design--scalability)
5. [Microservices Architecture](#microservices-architecture)
6. [Performance & Optimization](#performance--optimization)
7. [Security & Best Practices](#security--best-practices)
8. [DevOps & CI/CD](#devops--cicd)

---

## Python Architecture & Design

### Q1: Design a scalable event-driven architecture in Python

**Expected Answer:**

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Callable, Any
from dataclasses import dataclass
from datetime import datetime
import asyncio
from enum import Enum

class EventPriority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Event:
    """Represents an event in the system."""
    event_type: str
    payload: Dict[str, Any]
    timestamp: datetime
    priority: EventPriority = EventPriority.MEDIUM
    correlation_id: str = None

class EventHandler(ABC):
    """Abstract base class for event handlers."""
    
    @abstractmethod
    async def handle(self, event: Event) -> None:
        """Handle the event."""
        pass
    
    @abstractmethod
    def can_handle(self, event: Event) -> bool:
        """Check if this handler can process the event."""
        pass

class EventBus:
    """
    Central event bus for event-driven architecture.
    
    Features:
    - Async event processing
    - Priority-based handling
    - Multiple subscribers per event type
    - Dead letter queue for failed events
    """
    
    def __init__(self):
        self.handlers: Dict[str, List[EventHandler]] = {}
        self.middleware: List[Callable] = []
        self.dead_letter_queue: List[Event] = []
        self._running = False
    
    def subscribe(self, event_type: str, handler: EventHandler) -> None:
        """
        Subscribe handler to event type.
        
        Args:
            event_type: Type of event to handle
            handler: Event handler instance
        """
        if event_type not in self.handlers:
            self.handlers[event_type] = []
        self.handlers[event_type].append(handler)
    
    def add_middleware(self, middleware: Callable) -> None:
        """Add middleware to event processing pipeline."""
        self.middleware.append(middleware)
    
    async def publish(self, event: Event) -> None:
        """
        Publish event to all subscribers.
        
        Args:
            event: Event to publish
        """
        # Run middleware
        for mw in self.middleware:
            event = await mw(event)
        
        # Get handlers for this event type
        handlers = self.handlers.get(event.event_type, [])
        
        # Process handlers based on priority
        sorted_handlers = sorted(
            [h for h in handlers if h.can_handle(event)],
            key=lambda h: getattr(h, 'priority', 0),
            reverse=True
        )
        
        # Execute handlers concurrently
        tasks = [self._handle_event(handler, event) for handler in sorted_handlers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle failures
        for handler, result in zip(sorted_handlers, results):
            if isinstance(result, Exception):
                await self._handle_failure(event, handler, result)
    
    async def _handle_event(self, handler: EventHandler, event: Event) -> None:
        """Handle event with error handling."""
        try:
            await handler.handle(event)
        except Exception as e:
            raise e
    
    async def _handle_failure(self, event: Event, handler: EventHandler, error: Exception) -> None:
        """Handle event processing failure."""
        print(f"Event handling failed: {handler.__class__.__name__} - {error}")
        self.dead_letter_queue.append(event)

# Concrete Event Handlers
class UserRegistrationHandler(EventHandler):
    """Handles user registration events."""
    
    async def handle(self, event: Event) -> None:
        """Process user registration."""
        user_data = event.payload
        print(f"Creating user: {user_data['email']}")
        # Send welcome email, create profile, etc.
        await asyncio.sleep(0.1)  # Simulate processing
    
    def can_handle(self, event: Event) -> bool:
        return event.event_type == "user.registered"

class EmailNotificationHandler(EventHandler):
    """Sends email notifications."""
    
    async def handle(self, event: Event) -> None:
        """Send email notification."""
        print(f"Sending email for event: {event.event_type}")
        await asyncio.sleep(0.05)
    
    def can_handle(self, event: Event) -> bool:
        return True  # Handles all events

# Middleware example
async def logging_middleware(event: Event) -> Event:
    """Log all events."""
    print(f"[{event.timestamp}] {event.event_type}: {event.correlation_id}")
    return event

# Usage
async def main():
    bus = EventBus()
    
    # Register handlers
    bus.subscribe("user.registered", UserRegistrationHandler())
    bus.subscribe("user.registered", EmailNotificationHandler())
    
    # Add middleware
    bus.add_middleware(logging_middleware)
    
    # Publish event
    event = Event(
        event_type="user.registered",
        payload={"email": "user@example.com", "name": "John Doe"},
        timestamp=datetime.now(),
        priority=EventPriority.HIGH,
        correlation_id="req-123"
    )
    
    await bus.publish(event)

# Run
# asyncio.run(main())
```

**Key Discussion Points:**
- Event sourcing vs event-driven architecture
- Guaranteed delivery mechanisms
- Event replay capabilities
- Saga pattern for distributed transactions
- CQRS (Command Query Responsibility Segregation)

---

## FastAPI - Advanced Concepts

### Q2: Implement dependency injection with FastAPI for a multi-tenant application

**Expected Answer:**

```python
from fastapi import FastAPI, Depends, HTTPException, Header
from typing import Optional, Annotated
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from contextlib import asynccontextmanager
import redis.asyncio as aioredis

# Database Models
class Base(DeclarativeBase):
    pass

# Database dependency
class DatabaseManager:
    """Manages database connections per tenant."""
    
    def __init__(self):
        self.engines = {}
        self.session_makers = {}
    
    def get_engine(self, tenant_id: str):
        """Get or create engine for tenant."""
        if tenant_id not in self.engines:
            db_url = f"postgresql+asyncpg://user:pass@localhost/{tenant_id}_db"
            self.engines[tenant_id] = create_async_engine(db_url, pool_size=10)
            self.session_makers[tenant_id] = async_sessionmaker(
                self.engines[tenant_id],
                class_=AsyncSession,
                expire_on_commit=False
            )
        return self.engines[tenant_id]
    
    async def get_session(self, tenant_id: str) -> AsyncSession:
        """Get database session for tenant."""
        if tenant_id not in self.session_makers:
            self.get_engine(tenant_id)
        
        async with self.session_makers[tenant_id]() as session:
            yield session

# Dependency for tenant identification
async def get_tenant_id(
    x_tenant_id: Annotated[Optional[str], Header()] = None
) -> str:
    """Extract tenant ID from request headers."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")
    return x_tenant_id

# Cache dependency
class CacheManager:
    """Redis cache manager."""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection."""
        self.redis = await aioredis.from_url(
            "redis://localhost",
            encoding="utf-8",
            decode_responses=True
        )
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()

# Global instances
db_manager = DatabaseManager()
cache_manager = CacheManager()

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    await cache_manager.connect()
    yield
    # Shutdown
    await cache_manager.close()

# FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Dependency for cache
async def get_cache() -> aioredis.Redis:
    """Get Redis cache instance."""
    return cache_manager.redis

# Dependency for database session
async def get_db(
    tenant_id: str = Depends(get_tenant_id)
) -> AsyncSession:
    """Get database session for current tenant."""
    async for session in db_manager.get_session(tenant_id):
        yield session

# Authentication dependency
class AuthService:
    """Authentication and authorization service."""
    
    def __init__(self, cache: aioredis.Redis, tenant_id: str):
        self.cache = cache
        self.tenant_id = tenant_id
    
    async def get_current_user(self, token: str) -> dict:
        """Validate token and get user."""
        # Check cache first
        cached_user = await self.cache.get(f"token:{token}")
        if cached_user:
            return eval(cached_user)  # In production, use proper deserialization
        
        # Validate token (simplified)
        user = {"id": 1, "tenant_id": self.tenant_id, "role": "admin"}
        
        # Cache for 1 hour
        await self.cache.setex(f"token:{token}", 3600, str(user))
        return user

async def get_auth_service(
    cache: aioredis.Redis = Depends(get_cache),
    tenant_id: str = Depends(get_tenant_id)
) -> AuthService:
    """Get authentication service instance."""
    return AuthService(cache, tenant_id)

async def get_current_user(
    authorization: Annotated[Optional[str], Header()] = None,
    auth_service: AuthService = Depends(get_auth_service)
) -> dict:
    """Get current authenticated user."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authentication")
    
    token = authorization.replace("Bearer ", "")
    return await auth_service.get_current_user(token)

# API Endpoints using dependencies
@app.get("/users/me")
async def read_current_user(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    cache: aioredis.Redis = Depends(get_cache)
):
    """
    Get current user details.
    
    Dependencies injected:
    - current_user: Authenticated user
    - db: Database session for user's tenant
    - cache: Redis cache instance
    """
    return {
        "user": current_user,
        "tenant_id": current_user["tenant_id"]
    }

@app.post("/items/")
async def create_item(
    item: dict,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create item in tenant's database."""
    # Use tenant-specific database
    # Insert item using db session
    return {"item": item, "created_by": current_user["id"]}

# Dependency with cleanup
class ServiceWithCleanup:
    """Service that requires cleanup."""
    
    def __init__(self):
        self.connection = None
    
    async def connect(self):
        """Initialize connection."""
        self.connection = "connected"
    
    async def close(self):
        """Close connection."""
        self.connection = None

async def get_service_with_cleanup():
    """Dependency with proper cleanup."""
    service = ServiceWithCleanup()
    await service.connect()
    try:
        yield service
    finally:
        await service.close()
```

**Discussion Points:**
- How to handle tenant isolation at database level
- Caching strategies for multi-tenant applications
- Connection pooling per tenant
- Security considerations (data isolation, RBAC)
- Dependency injection patterns and testing

---

### Q3: Implement rate limiting middleware in FastAPI

**Expected Answer:**

```python
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Dict, Deque
from collections import deque
import time
from datetime import datetime, timedelta

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using sliding window algorithm.
    
    Features:
    - Per-user rate limiting
    - Configurable limits per endpoint
    - Redis-backed for distributed systems
    """
    
    def __init__(
        self,
        app,
        requests_per_minute: int = 60,
        requests_per_hour: int = 1000
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        
        # In-memory storage (use Redis for production)
        self.user_requests: Dict[str, Deque[float]] = {}
    
    async def dispatch(self, request: Request, call_next):
        """Process request with rate limiting."""
        # Get user identifier (IP or user ID from token)
        user_id = self._get_user_id(request)
        
        # Check rate limit
        if not self._is_allowed(user_id):
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "retry_after": self._get_retry_after(user_id)
                },
                headers={
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(self._get_retry_after(user_id))
                }
            )
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self._get_remaining_requests(user_id)
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        
        return response
    
    def _get_user_id(self, request: Request) -> str:
        """Extract user ID from request."""
        # Try to get from auth token
        auth = request.headers.get("Authorization", "")
        if auth:
            # In production, decode JWT and get user_id
            return auth
        
        # Fall back to IP address
        return request.client.host
    
    def _is_allowed(self, user_id: str) -> bool:
        """Check if request is allowed under rate limit."""
        current_time = time.time()
        
        if user_id not in self.user_requests:
            self.user_requests[user_id] = deque()
        
        requests = self.user_requests[user_id]
        
        # Remove requests outside 1-minute window
        while requests and requests[0] <= current_time - 60:
            requests.popleft()
        
        # Check minute limit
        if len(requests) >= self.requests_per_minute:
            return False
        
        # Add current request
        requests.append(current_time)
        return True
    
    def _get_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user."""
        if user_id not in self.user_requests:
            return self.requests_per_minute
        
        current_time = time.time()
        requests = self.user_requests[user_id]
        
        # Count valid requests
        valid_requests = sum(
            1 for req_time in requests 
            if req_time > current_time - 60
        )
        
        return max(0, self.requests_per_minute - valid_requests)
    
    def _get_retry_after(self, user_id: str) -> int:
        """Get seconds until next request is allowed."""
        if user_id not in self.user_requests:
            return 0
        
        requests = self.user_requests[user_id]
        if not requests:
            return 0
        
        oldest_request = requests[0]
        elapsed = time.time() - oldest_request
        return max(0, int(60 - elapsed))

# Redis-backed rate limiter for distributed systems
import aioredis

class DistributedRateLimiter:
    """
    Distributed rate limiter using Redis.
    
    Suitable for multi-instance deployments.
    """
    
    def __init__(self, redis_url: str, max_requests: int = 100, window: int = 60):
        self.redis_url = redis_url
        self.max_requests = max_requests
        self.window = window
        self.redis = None
    
    async def connect(self):
        """Initialize Redis connection."""
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def is_allowed(self, user_id: str) -> bool:
        """Check rate limit using Redis."""
        key = f"rate_limit:{user_id}"
        current_time = time.time()
        
        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()
        
        # Remove old requests
        pipe.zremrangebyscore(key, 0, current_time - self.window)
        
        # Count requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, self.window)
        
        results = await pipe.execute()
        request_count = results[1]
        
        return request_count < self.max_requests

# Apply middleware
app = FastAPI()
app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
```

**Discussion Points:**
- Token bucket vs sliding window algorithms
- Distributed rate limiting with Redis
- Per-endpoint vs global rate limits
- Handling burst traffic
- Rate limit strategies for different user tiers

---

### Q4: Design a robust API versioning strategy in FastAPI

**Expected Answer:**

```python
from fastapi import FastAPI, APIRouter, Request
from fastapi.routing import APIRoute
from typing import Callable, Optional
from pydantic import BaseModel
from enum import Enum

# Version 1 Models
class UserV1(BaseModel):
    id: int
    name: str
    email: str

# Version 2 Models (with breaking changes)
class UserV2(BaseModel):
    id: int
    first_name: str  # Breaking change: split name
    last_name: str
    email: str
    phone: Optional[str] = None  # New field

# Versioning Strategy 1: URL Path Versioning
app = FastAPI()

v1_router = APIRouter(prefix="/api/v1", tags=["v1"])
v2_router = APIRouter(prefix="/api/v2", tags=["v2"])

@v1_router.get("/users/{user_id}", response_model=UserV1)
async def get_user_v1(user_id: int):
    """Version 1 endpoint."""
    return UserV1(id=user_id, name="John Doe", email="john@example.com")

@v2_router.get("/users/{user_id}", response_model=UserV2)
async def get_user_v2(user_id: int):
    """Version 2 endpoint with enhanced response."""
    return UserV2(
        id=user_id,
        first_name="John",
        last_name="Doe",
        email="john@example.com",
        phone="+1234567890"
    )

app.include_router(v1_router)
app.include_router(v2_router)

# Versioning Strategy 2: Header-based Versioning
class APIVersion(str, Enum):
    V1 = "1.0"
    V2 = "2.0"

async def get_api_version(request: Request) -> APIVersion:
    """Extract API version from headers."""
    version = request.headers.get("API-Version", "1.0")
    try:
        return APIVersion(version)
    except ValueError:
        return APIVersion.V1

@app.get("/users/{user_id}")
async def get_user_versioned(
    user_id: int,
    version: APIVersion = Depends(get_api_version)
):
    """Version-aware endpoint."""
    if version == APIVersion.V1:
        return UserV1(id=user_id, name="John Doe", email="john@example.com")
    else:
        return UserV2(
            id=user_id,
            first_name="John",
            last_name="Doe",
            email="john@example.com",
            phone="+1234567890"
        )

# Versioning Strategy 3: Content Negotiation
@app.get("/users/{user_id}")
async def get_user_content_negotiation(
    user_id: int,
    request: Request
):
    """Use Accept header for versioning."""
    accept = request.headers.get("Accept", "application/vnd.api.v1+json")
    
    if "v2" in accept:
        return UserV2(
            id=user_id,
            first_name="John",
            last_name="Doe",
            email="john@example.com"
        )
    else:
        return UserV1(id=user_id, name="John Doe", email="john@example.com")

# Version deprecation middleware
from datetime import datetime

class VersionDeprecationMiddleware(BaseHTTPMiddleware):
    """Middleware to handle API version deprecation."""
    
    def __init__(self, app, deprecated_versions: Dict[str, datetime]):
        super().__init__(app)
        self.deprecated_versions = deprecated_versions
    
    async def dispatch(self, request: Request, call_next):
        """Add deprecation warnings."""
        # Extract version from path
        path_parts = request.url.path.split("/")
        if len(path_parts) > 2 and path_parts[2].startswith("v"):
            version = path_parts[2]
            
            if version in self.deprecated_versions:
                sunset_date = self.deprecated_versions[version]
                response = await call_next(request)
                response.headers["Sunset"] = sunset_date.isoformat()
                response.headers["Deprecation"] = "true"
                response.headers["Link"] = f'</api/v2>; rel="successor-version"'
                return response
        
        return await call_next(request)

# Apply deprecation middleware
app.add_middleware(
    VersionDeprecationMiddleware,
    deprecated_versions={
        "v1": datetime(2025, 12, 31)
    }
)
```

**Discussion Points:**
- Pros/cons of different versioning strategies
- Backward compatibility strategies
- API deprecation timeline management
- Documentation per version (OpenAPI/Swagger)
- Client migration strategies

---

### Q5: Implement comprehensive error handling and logging in FastAPI

**Expected Answer:**

```python
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import logging
import sys
import traceback
from typing import Union
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

# Custom exceptions
class BusinessLogicError(Exception):
    """Base exception for business logic errors."""
    def __init__(self, message: str, error_code: str = "BUSINESS_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)

class ResourceNotFoundError(BusinessLogicError):
    """Resource not found exception."""
    def __init__(self, resource_type: str, resource_id: Union[int, str]):
        message = f"{resource_type} with id {resource_id} not found"
        super().__init__(message, "RESOURCE_NOT_FOUND")

class InsufficientPermissionsError(BusinessLogicError):
    """Permission denied exception."""
    def __init__(self, action: str, resource: str):
        message = f"Insufficient permissions to {action} {resource}"
        super().__init__(message, "PERMISSION_DENIED")

# Error response model
class ErrorResponse(BaseModel):
    """Standardized error response."""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    request_id: str
    timestamp: datetime

app = FastAPI()

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID to each request."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Log request
    logger.info(
        f"Request started",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host
        }
    )
    
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    # Log response
    logger.info(
        f"Request completed",
        extra={
            "request_id": request_id,
            "status_code": response.status_code
        }
    )
    
    return response

# Custom exception handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error_code=f"HTTP_{exc.status_code}",
            message=exc.detail,
            request_id=getattr(request.state, "request_id", "unknown"),
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.warning(
        f"Validation error",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "errors": exc.errors()
        }
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            details={"errors": exc.errors()},
            request_id=getattr(request.state, "request_id", "unknown"),
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(BusinessLogicError)
async def business_logic_exception_handler(request: Request, exc: BusinessLogicError):
    """Handle business logic exceptions."""
    status_map = {
        "RESOURCE_NOT_FOUND": status.HTTP_404_NOT_FOUND,
        "PERMISSION_DENIED": status.HTTP_403_FORBIDDEN,
        "BUSINESS_ERROR": status.HTTP_400_BAD_REQUEST
    }
    
    logger.error(
        f"Business logic error: {exc.message}",
        extra={
            "request_id": getattr(request.state, "request_id", "unknown"),
            "error_code": exc.error_code
        }
    )
    
    return JSONResponse(
        status_code=status_map.get(exc.error_code, status.HTTP_400_BAD_REQUEST),
        content=ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            request_id=getattr(request.state, "request_id", "unknown"),
            timestamp=datetime.now()
        ).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    
    logger.error(
        f"Unhandled exception",
        extra={
            "request_id": request_id,
            "exception": str(exc),
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            details={"error": str(exc)} if app.debug else None,
            request_id=request_id,
            timestamp=datetime.now()
        ).dict()
    )

# Usage in endpoints
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Example endpoint with error handling."""
    if user_id < 0:
        raise BusinessLogicError("Invalid user ID", "INVALID_INPUT")
    
    # Simulate database lookup
    if user_id > 1000:
        raise ResourceNotFoundError("User", user_id)
    
    return {"id": user_id, "name": "John Doe"}
```

**Discussion Points:**
- Structured logging (JSON format for log aggregation)
- Centralized error handling patterns
- Error monitoring and alerting (Sentry, DataDog)
- Request tracing across microservices
- PII masking in logs

---

## Apache Airflow - Orchestration

### Q6: Design a complex Airflow DAG with dynamic task generation

**Expected Answer:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.task_group import TaskGroup
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime, timedelta
from typing import List, Dict
import logging

# Default DAG arguments
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
    'execution_timeout': timedelta(hours=2),
}

# DAG with complex dependencies
dag = DAG(
    'multi_tenant_data_pipeline',
    default_args=default_args,
    description='Process data for multiple tenants with dynamic tasks',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
    max_active_runs=1,
    tags=['production', 'multi-tenant', 'etl']
)

def get_tenant_list(**context) -> List[str]:
    """
    Dynamically fetch list of tenants to process.
    
    Returns:
        List of tenant IDs
    """
    # In production, fetch from database or config
    tenants = Variable.get("active_tenants", deserialize_json=True)
    logger = logging.getLogger(__name__)
    logger.info(f"Processing {len(tenants)} tenants: {tenants}")
    
    # Push to XCom for downstream tasks
    context['task_instance'].xcom_push(key='tenant_list', value=tenants)
    return tenants

def extract_data(tenant_id: str, **context):
    """Extract data for specific tenant."""
    logger = logging.getLogger(__name__)
    logger.info(f"Extracting data for tenant: {tenant_id}")
    
    # Simulate data extraction
    data = {
        'tenant_id': tenant_id,
        'records': 1000,
        'timestamp': datetime.now().isoformat()
    }
    
    # Store in XCom
    context['task_instance'].xcom_push(
        key=f'extract_data_{tenant_id}',
        value=data
    )

def transform_data(tenant_id: str, **context):
    """Transform data for specific tenant."""
    # Pull data from extract step
    ti = context['task_instance']
    extract_data = ti.xcom_pull(
        task_ids=f'extract_tenant_{tenant_id}',
        key=f'extract_data_{tenant_id}'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Transforming {extract_data['records']} records for {tenant_id}")
    
    # Transform logic here
    transformed_data = {
        **extract_data,
        'transformed': True,
        'transform_time': datetime.now().isoformat()
    }
    
    ti.xcom_push(key=f'transform_data_{tenant_id}', value=transformed_data)

def load_data(tenant_id: str, **context):
    """Load data for specific tenant."""
    ti = context['task_instance']
    transform_data = ti.xcom_pull(
        task_ids=f'transform_tenant_{tenant_id}',
        key=f'transform_data_{tenant_id}'
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data for tenant: {tenant_id}")
    
    # Load to database/warehouse
    # db.load(transform_data)

def validate_pipeline(**context):
    """Validate that all tenants were processed successfully."""
    ti = context['task_instance']
    tenant_list = ti.xcom_pull(task_ids='get_tenants', key='tenant_list')
    
    successful = 0
    failed = []
    
    for tenant_id in tenant_list:
        try:
            data = ti.xcom_pull(
                task_ids=f'load_tenant_{tenant_id}',
                key=f'load_data_{tenant_id}'
            )
            if data:
                successful += 1
            else:
                failed.append(tenant_id)
        except Exception:
            failed.append(tenant_id)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Pipeline validation: {successful} successful, {len(failed)} failed")
    
    if failed:
        raise ValueError(f"Pipeline failed for tenants: {failed}")

# Build DAG
with dag:
    # Initial task to get tenant list
    get_tenants = PythonOperator(
        task_id='get_tenants',
        python_callable=get_tenant_list,
        provide_context=True
    )
    
    # Dynamic task generation based on tenant list
    tenant_list = Variable.get("active_tenants", deserialize_json=True, default_var=["tenant1", "tenant2"])
    
    tenant_tasks = []
    
    for tenant_id in tenant_list:
        # Create task group for each tenant
        with TaskGroup(group_id=f'process_tenant_{tenant_id}') as tenant_group:
            # Extract
            extract = PythonOperator(
                task_id=f'extract_tenant_{tenant_id}',
                python_callable=extract_data,
                op_kwargs={'tenant_id': tenant_id},
                provide_context=True
            )
            
            # Transform
            transform = PythonOperator(
                task_id=f'transform_tenant_{tenant_id}',
                python_callable=transform_data,
                op_kwargs={'tenant_id': tenant_id},
                provide_context=True
            )
            
            # Load
            load = PythonOperator(
                task_id=f'load_tenant_{tenant_id}',
                python_callable=load_data,
                op_kwargs={'tenant_id': tenant_id},
                provide_context=True
            )
            
            # Define dependencies within task group
            extract >> transform >> load
        
        tenant_tasks.append(tenant_group)
    
    # Validation task
    validate = PythonOperator(
        task_id='validate_pipeline',
        python_callable=validate_pipeline,
        provide_context=True,
        trigger_rule='all_done'  # Run even if some tasks fail
    )
    
    # Final notification
    notify = BashOperator(
        task_id='send_notification',
        bash_command='echo "Pipeline completed for all tenants"'
    )
    
    # Define DAG dependencies
    get_tenants >> tenant_tasks >> validate >> notify
```

**Discussion Points:**
- Dynamic task generation strategies
- XCom usage and limitations
- Task group organization
- Trigger rules and dependencies
- DAG performance optimization
- Handling failures and retries
- Backfill strategies

---

### Q7: Implement custom Airflow operators and sensors

**Expected Answer:**

```python
from airflow.models import BaseOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from typing import Optional, Dict, Any
import requests
import time
from datetime import timedelta

class HTTPAPIOperator(BaseOperator):
    """
    Custom operator to call HTTP APIs with retry logic.
    
    Features:
    - Automatic retries with exponential backoff
    - Response validation
    - Error handling
    - Authentication support
    """
    
    template_fields = ('endpoint', 'headers', 'data')
    ui_color = '#4CAF50'
    
    @apply_defaults
    def __init__(
        self,
        endpoint: str,
        method: str = 'GET',
        headers: Optional[Dict[str, str]] = None,
        data: Optional[Dict[str, Any]] = None,
        expected_status: int = 200,
        timeout: int = 30,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.endpoint = endpoint
        self.method = method.upper()
        self.headers = headers or {}
        self.data = data
        self.expected_status = expected_status
        self.timeout = timeout
    
    def execute(self, context):
        """Execute the API call."""
        self.log.info(f"Calling {self.method} {self.endpoint}")
        
        try:
            response = requests.request(
                method=self.method,
                url=self.endpoint,
                headers=self.headers,
                json=self.data,
                timeout=self.timeout
            )
            
            self.log.info(f"Response status: {response.status_code}")
            
            if response.status_code != self.expected_status:
                raise ValueError(
                    f"Unexpected status code: {response.status_code}. "
                    f"Expected: {self.expected_status}"
                )
            
            result = response.json()
            
            # Push result to XCom
            context['task_instance'].xcom_push(
                key='api_response',
                value=result
            )
            
            return result
            
        except requests.exceptions.Timeout:
            self.log.error(f"Request timeout after {self.timeout}s")
            raise
        except requests.exceptions.RequestException as e:
            self.log.error(f"Request failed: {str(e)}")
            raise
        except ValueError as e:
            self.log.error(str(e))
            raise

class S3FileSensor(BaseSensorOperator):
    """
    Custom sensor to wait for S3 file availability.
    
    Pokes S3 bucket until file exists or timeout.
    """
    
    template_fields = ('bucket', 'key')
    
    @apply_defaults
    def __init__(
        self,
        bucket: str,
        key: str,
        aws_conn_id: str = 'aws_default',
        poke_interval: int = 60,
        timeout: int = 3600,
        *args, **kwargs
    ):
        super().__init__(
            poke_interval=poke_interval,
            timeout=timeout,
            *args, **kwargs
        )
        self.bucket = bucket
        self.key = key
        self.aws_conn_id = aws_conn_id
    
    def poke(self, context) -> bool:
        """
        Check if file exists in S3.
        
        Returns:
            True if file exists, False otherwise
        """
        from airflow.providers.amazon.aws.hooks.s3 import S3Hook
        
        self.log.info(f"Checking for s3://{self.bucket}/{self.key}")
        
        try:
            hook = S3Hook(aws_conn_id=self.aws_conn_id)
            exists = hook.check_for_key(key=self.key, bucket_name=self.bucket)
            
            if exists:
                self.log.info(f"File found: s3://{self.bucket}/{self.key}")
                
                # Optionally push file metadata to XCom
                metadata = hook.get_file_metadata(
                    key=self.key,
                    bucket_name=self.bucket
                )
                context['task_instance'].xcom_push(
                    key='file_metadata',
                    value=metadata
                )
                
                return True
            else:
                self.log.info(f"File not found yet: s3://{self.bucket}/{self.key}")
                return False
                
        except Exception as e:
            self.log.error(f"Error checking S3: {str(e)}")
            return False

class DataQualityOperator(BaseOperator):
    """
    Custom operator for data quality checks.
    
    Validates data against defined quality rules.
    """
    
    template_fields = ('table_name', 'checks')
    ui_color = '#FFC107'
    
    @apply_defaults
    def __init__(
        self,
        table_name: str,
        checks: List[Dict[str, Any]],
        conn_id: str = 'postgres_default',
        fail_on_error: bool = True,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.table_name = table_name
        self.checks = checks
        self.conn_id = conn_id
        self.fail_on_error = fail_on_error
    
    def execute(self, context):
        """Execute data quality checks."""
        from airflow.providers.postgres.hooks.postgres import PostgresHook
        
        hook = PostgresHook(postgres_conn_id=self.conn_id)
        failed_checks = []
        
        for check in self.checks:
            check_name = check['name']
            check_sql = check['sql']
            expected = check.get('expected', 0)
            
            self.log.info(f"Running check: {check_name}")
            
            result = hook.get_first(check_sql)
            actual = result[0] if result else None
            
            if actual != expected:
                failed_checks.append({
                    'check': check_name,
                    'expected': expected,
                    'actual': actual
                })
                self.log.error(
                    f"Check failed: {check_name}. "
                    f"Expected: {expected}, Got: {actual}"
                )
        
        if failed_checks:
            if self.fail_on_error:
                raise ValueError(f"Data quality checks failed: {failed_checks}")
            else:
                self.log.warning(f"Data quality issues: {failed_checks}")
        
        # Push results to XCom
        context['task_instance'].xcom_push(
            key='quality_check_results',
            value={
                'total_checks': len(self.checks),
                'failed_checks': len(failed_checks),
                'failures': failed_checks
            }
        )

# Usage in DAG
with DAG('data_quality_pipeline', **default_args) as dag:
    
    # Wait for source file
    wait_for_file = S3FileSensor(
        task_id='wait_for_source_data',
        bucket='data-lake',
        key='raw/{{ ds }}/data.parquet',
        poke_interval=300,  # Check every 5 minutes
        timeout=3600  # 1 hour timeout
    )
    
    # Call external API
    fetch_metadata = HTTPAPIOperator(
        task_id='fetch_metadata',
        endpoint='https://api.example.com/metadata',
        method='GET',
        headers={'Authorization': 'Bearer {{ var.value.api_token }}'},
        expected_status=200
    )
    
    # Data quality checks
    quality_checks = DataQualityOperator(
        task_id='validate_data_quality',
        table_name='processed_data',
        checks=[
            {
                'name': 'row_count',
                'sql': 'SELECT COUNT(*) FROM processed_data WHERE date = "{{ ds }}"',
                'expected': 1000
            },
            {
                'name': 'null_check',
                'sql': 'SELECT COUNT(*) FROM processed_data WHERE email IS NULL',
                'expected': 0
            },
            {
                'name': 'duplicate_check',
                'sql': '''
                    SELECT COUNT(*) - COUNT(DISTINCT email) 
                    FROM processed_data 
                    WHERE date = "{{ ds }}"
                ''',
                'expected': 0
            }
        ]
    )
    
    # Dependencies
    wait_for_file >> fetch_metadata >> quality_checks
```

**Discussion Points:**
- Custom operator vs PythonOperator
- When to create custom sensors
- XCom usage patterns and limitations
- Operator idempotency
- Testing custom operators
- Packaging and distributing custom operators

---

### Q8: Design an Airflow architecture for a large-scale data platform

**Expected Answer:**

**Architecture Components:**

```python
"""
Scalable Airflow Architecture Design

1. INFRASTRUCTURE LAYER:
   - Kubernetes-based deployment (Helm charts)
   - Autoscaling workers (KEDA for Kubernetes)
   - Separate executor for different workload types
   - Multi-region deployment for DR

2. AIRFLOW COMPONENTS:
   - Webserver: 3+ replicas behind load balancer
   - Scheduler: 2 replicas (HA mode with database locking)
   - Workers: Auto-scaling (Celery or Kubernetes executor)
   - Database: PostgreSQL (RDS with Multi-AZ)
   - Message Broker: Redis or RabbitMQ (for Celery)
   - Storage: S3/GCS for logs and XComs

3. EXECUTION LAYER:
   - CeleryExecutor: For general tasks
   - KubernetesExecutor: For resource-intensive tasks
   - Pool-based resource management
   - Task queues for priority handling

4. MONITORING & OBSERVABILITY:
   - Prometheus + Grafana for metrics
   - ELK/DataDog for centralized logging
   - StatsD for custom metrics
   - Alerting via PagerDuty/Slack

5. DAG ORGANIZATION:
   - DAGs stored in Git repository
   - CI/CD pipeline for DAG deployment
   - DAG validation before deployment
   - Separate folders per domain/team
"""

# Configuration as code
from airflow.models import Variable
from airflow.models.pool import Pool

# Environment-specific configuration
class AirflowConfig:
    """Centralized configuration management."""
    
    # Executor configuration
    EXECUTOR_CONFIGS = {
        'development': {
            'executor': 'LocalExecutor',
            'parallelism': 32,
            'dag_concurrency': 16
        },
        'production': {
            'executor': 'CeleryExecutor',
            'parallelism': 512,
            'dag_concurrency': 64,
            'worker_autoscale': '16,4'
        }
    }
    
    # Pool definitions
    POOLS = [
        {'pool': 'etl_pool', 'slots': 100, 'description': 'ETL tasks'},
        {'pool': 'ml_pool', 'slots': 20, 'description': 'ML training'},
        {'pool': 'api_pool', 'slots': 50, 'description': 'API calls'},
    ]
    
    # SLA configurations
    SLA_CONFIGS = {
        'critical': timedelta(hours=1),
        'high': timedelta(hours=4),
        'medium': timedelta(hours=12),
        'low': timedelta(days=1)
    }

# Resource management with pools
def create_pools():
    """Create resource pools programmatically."""
    from airflow.utils.session import create_session
    
    with create_session() as session:
        for pool_config in AirflowConfig.POOLS:
            pool = Pool(
                pool=pool_config['pool'],
                slots=pool_config['slots'],
                description=pool_config['description']
            )
            session.merge(pool)
        session.commit()

# DAG factory pattern
def create_etl_dag(
    dag_id: str,
    source_system: str,
    schedule: str,
    pool: str = 'etl_pool'
) -> DAG:
    """
    Factory function to create standardized ETL DAGs.
    
    Args:
        dag_id: Unique DAG identifier
        source_system: Source system name
        schedule: Cron schedule
        pool: Resource pool to use
        
    Returns:
        Configured DAG instance
    """
    dag = DAG(
        dag_id=dag_id,
        default_args={
            'owner': 'data-engineering',
            'depends_on_past': False,
            'start_date': datetime(2025, 1, 1),
            'email_on_failure': True,
            'email': ['data-alerts@company.com'],
            'retries': 3,
            'retry_delay': timedelta(minutes=5),
            'pool': pool,
            'sla': AirflowConfig.SLA_CONFIGS['medium']
        },
        schedule_interval=schedule,
        catchup=False,
        max_active_runs=1,
        tags=['etl', source_system]
    )
    
    return dag

# Usage
salesforce_dag = create_etl_dag(
    dag_id='salesforce_daily_sync',
    source_system='salesforce',
    schedule='0 2 * * *'
)
```

**Architecture Decisions to Discuss:**
- **Executor Choice**: CeleryExecutor vs KubernetesExecutor
  - Celery: Better for many small tasks
  - Kubernetes: Better for resource-intensive, isolated tasks
  
- **Database Selection**: PostgreSQL vs MySQL
  - PostgreSQL recommended for production
  - Proper indexing on task_instance table
  
- **Scalability Strategies**:
  - Horizontal scaling of workers
  - Multiple schedulers for HA
  - Database connection pooling
  - XCom backend (S3 for large data)
  
- **High Availability**:
  - Multiple scheduler replicas
  - Database replication
  - Redis/RabbitMQ clustering
  - Health checks and auto-recovery

---

### Q9: Implement Airflow DAG testing and CI/CD

**Expected Answer:**

```python
import pytest
from airflow.models import DagBag, TaskInstance
from airflow.utils.state import State
from datetime import datetime
import os

# Test DAG integrity
class TestDAGIntegrity:
    """Test suite for DAG validation."""
    
    DAGBAG_FOLDER = os.path.join(os.path.dirname(__file__), '../dags')
    
    def test_dagbag_import(self):
        """Test that all DAGs load without errors."""
        dagbag = DagBag(dag_folder=self.DAGBAG_FOLDER, include_examples=False)
        
        assert len(dagbag.import_errors) == 0, \
            f"DAG import failures: {dagbag.import_errors}"
    
    def test_dag_tags(self):
        """Test that all DAGs have required tags."""
        dagbag = DagBag(dag_folder=self.DAGBAG_FOLDER, include_examples=False)
        
        for dag_id, dag in dagbag.dags.items():
            assert dag.tags, f"DAG {dag_id} must have tags"
            # Ensure at least one of: team, domain, or env tag
            tag_categories = ['team-', 'domain-', 'env-']
            has_category = any(
                any(tag.startswith(cat) for tag in dag.tags)
                for cat in tag_categories
            )
            assert has_category, \
                f"DAG {dag_id} must have team, domain, or env tag"
    
    def test_dag_ownership(self):
        """Test that all DAGs have owner defined."""
        dagbag = DagBag(dag_folder=self.DAGBAG_FOLDER, include_examples=False)
        
        for dag_id, dag in dagbag.dags.items():
            assert dag.default_args.get('owner'), \
                f"DAG {dag_id} must have owner defined"
    
    def test_dag_retries(self):
        """Test that all DAGs have retry configured."""
        dagbag = DagBag(dag_folder=self.DAGBAG_FOLDER, include_examples=False)
        
        for dag_id, dag in dagbag.dags.items():
            retries = dag.default_args.get('retries', 0)
            assert retries > 0, \
                f"DAG {dag_id} must have retries > 0"
    
    def test_task_count(self):
        """Test that DAGs have reasonable number of tasks."""
        dagbag = DagBag(dag_folder=self.DAGBAG_FOLDER, include_examples=False)
        
        MAX_TASKS = 100
        for dag_id, dag in dagbag.dags.items():
            task_count = len(dag.tasks)
            assert task_count <= MAX_TASKS, \
                f"DAG {dag_id} has too many tasks: {task_count}"
    
    def test_no_cycles(self):
        """Test that DAGs have no circular dependencies."""
        dagbag = DagBag(dag_folder=self.DAGBAG_FOLDER, include_examples=False)
        
        for dag_id, dag in dagbag.dags.items():
            # This will raise if there are cycles
            dag.test_cycle()

# Test specific DAG execution
class TestETLDAG:
    """Test specific DAG logic."""
    
    @pytest.fixture
    def dagbag(self):
        return DagBag(dag_folder=self.DAGBAG_FOLDER, include_examples=False)
    
    def test_etl_dag_structure(self, dagbag):
        """Test ETL DAG has required tasks."""
        dag = dagbag.get_dag('multi_tenant_data_pipeline')
        assert dag is not None
        
        required_tasks = ['get_tenants', 'validate_pipeline']
        for task_id in required_tasks:
            assert task_id in dag.task_ids
    
    def test_task_dependencies(self, dagbag):
        """Test task dependencies are correct."""
        dag = dagbag.get_dag('multi_tenant_data_pipeline')
        
        # Test specific dependency
        validate_task = dag.get_task('validate_pipeline')
        upstream_task_ids = [t.task_id for t in validate_task.upstream_list]
        
        # Validate should come after all processing
        assert any('process_tenant' in task_id for task_id in upstream_task_ids)
    
    def test_task_execution(self, dagbag):
        """Test individual task execution."""
        dag = dagbag.get_dag('multi_tenant_data_pipeline')
        task = dag.get_task('get_tenants')
        
        # Create task instance
        execution_date = datetime(2025, 1, 1)
        ti = TaskInstance(task=task, execution_date=execution_date)
        
        # Mock context
        context = ti.get_template_context()
        
        # Execute task
        result = task.execute(context)
        
        # Assert result
        assert result is not None
        assert isinstance(result, list)

# CI/CD Pipeline (.github/workflows/airflow-ci.yml)
CICD_YAML = """
name: Airflow DAG CI/CD

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'dags/**'
      - 'plugins/**'
  pull_request:
    branches: [ main ]
    paths:
      - 'dags/**'

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install apache-airflow==2.8.0
        pip install pytest pytest-cov
        pip install -r requirements.txt
    
    - name: Lint DAGs
      run: |
        pip install flake8
        flake8 dags/ --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Test DAG integrity
      run: |
        python -m pytest tests/test_dag_integrity.py -v
    
    - name: Test DAG logic
      run: |
        python -m pytest tests/test_dags/ -v --cov=dags
    
    - name: Static analysis
      run: |
        pip install pylint
        pylint dags/ --disable=C,R
    
  deploy-dev:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Dev Airflow
      run: |
        # Copy DAGs to Airflow DAG folder (via rsync, S3, or git-sync)
        aws s3 sync dags/ s3://airflow-dev-dags/dags/
        
        # Trigger DAG refresh
        curl -X POST https://airflow-dev/api/v1/dags/refresh
    
  deploy-prod:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to Prod Airflow
      run: |
        # Deploy with approval gate
        aws s3 sync dags/ s3://airflow-prod-dags/dags/
        
        # Create deployment record
        echo "Deployed at $(date)" > deployment.log
"""
```

**Discussion Points:**
- DAG deployment strategies (git-sync vs S3 sync)
- Environment isolation (dev, staging, prod)
- DAG versioning and rollback
- Blue-green deployment for DAGs
- Automated testing strategies
- Performance testing for DAGs

---

## System Design & Scalability

### Q10: Design a highly scalable microservices architecture using FastAPI

**Expected Answer:**

```python
"""
Microservices Architecture Design

ARCHITECTURE COMPONENTS:

1. API GATEWAY (FastAPI)
   - Request routing
   - Authentication/Authorization
   - Rate limiting
   - Request/Response transformation
   - Circuit breaker pattern

2. SERVICE MESH
   - Service discovery (Consul/Eureka)
   - Load balancing
   - Health checks
   - Distributed tracing

3. SERVICES (FastAPI-based)
   - User Service
   - Product Service
   - Order Service
   - Payment Service
   - Notification Service

4. DATA LAYER
   - Database per service (database isolation)
   - Event store for event sourcing
   - Caching layer (Redis)
   - Message queue (RabbitMQ/Kafka)

5. OBSERVABILITY
   - Distributed tracing (Jaeger/Zipkin)
   - Centralized logging (ELK)
   - Metrics (Prometheus)
   - APM (DataDog/New Relic)
"""

# API Gateway Implementation
from fastapi import FastAPI, Request, HTTPException
from httpx import AsyncClient, Timeout
from typing import Dict, Optional
import hashlib
import time

class ServiceRegistry:
    """Service discovery and health checking."""
    
    def __init__(self):
        self.services: Dict[str, Dict] = {
            'user-service': {
                'url': 'http://user-service:8001',
                'health_check': '/health',
                'circuit_breaker': CircuitBreaker()
            },
            'product-service': {
                'url': 'http://product-service:8002',
                'health_check': '/health',
                'circuit_breaker': CircuitBreaker()
            },
            'order-service': {
                'url': 'http://order-service:8003',
                'health_check': '/health',
                'circuit_breaker': CircuitBreaker()
            }
        }
    
    def get_service_url(self, service_name: str) -> Optional[str]:
        """Get service URL with load balancing."""
        service = self.services.get(service_name)
        if not service:
            return None
        
        # Check circuit breaker
        if not service['circuit_breaker'].is_available():
            return None
        
        return service['url']

class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    States: CLOSED -> OPEN -> HALF_OPEN -> CLOSED
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        expected_exception: type = Exception
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def is_available(self) -> bool:
        """Check if service is available."""
        if self.state == "CLOSED":
            return True
        
        if self.state == "OPEN":
            # Check if timeout period has passed
            if time.time() - self.last_failure_time >= self.timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        
        # HALF_OPEN state
        return True
    
    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

# API Gateway
class APIGateway:
    """Central API Gateway for microservices."""
    
    def __init__(self):
        self.registry = ServiceRegistry()
        self.client = AsyncClient(timeout=Timeout(30.0))
    
    async def route_request(
        self,
        service_name: str,
        path: str,
        method: str = "GET",
        **kwargs
    ) -> Dict:
        """Route request to appropriate microservice."""
        service_url = self.registry.get_service_url(service_name)
        
        if not service_url:
            raise HTTPException(
                status_code=503,
                detail=f"Service {service_name} unavailable"
            )
        
        circuit_breaker = self.registry.services[service_name]['circuit_breaker']
        
        try:
            # Make request
            url = f"{service_url}{path}"
            response = await self.client.request(method, url, **kwargs)
            
            # Record success
            circuit_breaker.record_success()
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            # Record failure
            circuit_breaker.record_failure()
            raise HTTPException(
                status_code=503,
                detail=f"Service {service_name} error: {str(e)}"
            )

# Gateway app
gateway = FastAPI(title="API Gateway")
api_gateway = APIGateway()

@gateway.get("/users/{user_id}")
async def get_user(user_id: int, request: Request):
    """Proxy to user service."""
    return await api_gateway.route_request(
        service_name="user-service",
        path=f"/users/{user_id}",
        headers=dict(request.headers)
    )

@gateway.post("/orders")
async def create_order(order_data: dict, request: Request):
    """Proxy to order service."""
    return await api_gateway.route_request(
        service_name="order-service",
        path="/orders",
        method="POST",
        json=order_data,
        headers=dict(request.headers)
    )

# Distributed tracing
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

def setup_tracing(app: FastAPI, service_name: str):
    """Set up distributed tracing."""
    trace.set_tracer_provider(TracerProvider())
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger",
        agent_port=6831,
    )
    
    trace.get_tracer_provider().add_span_processor(
        BatchSpanProcessor(jaeger_exporter)
    )
    
    FastAPIInstrumentor.instrument_app(app)

# Setup tracing for gateway
setup_tracing(gateway, "api-gateway")
```

**Discussion Points:**
- Service mesh vs API Gateway
- Inter-service communication (gRPC vs REST)
- Event-driven vs request-response
- Saga pattern for distributed transactions
- Eventual consistency handling
- Service versioning strategies
- Monitoring and observability

---

## Performance & Optimization

### Q11: Optimize FastAPI application for high throughput

**Expected Answer:**

```python
from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from contextlib import asynccontextmanager
import asyncpg
import aioredis
from typing import List
import orjson

# Use ORJSONResponse for faster JSON serialization
app = FastAPI(default_response_class=ORJSONResponse)

# Connection pooling
class DatabasePool:
    """PostgreSQL connection pool manager."""
    
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None
    
    async def create_pool(self):
        """Create connection pool."""
        self.pool = await asyncpg.create_pool(
            host='localhost',
            database='mydb',
            user='user',
            password='password',
            min_size=10,
            max_size=50,
            max_queries=50000,
            max_inactive_connection_lifetime=300,
            command_timeout=60
        )
    
    async def close_pool(self):
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
    
    async def fetch_one(self, query: str, *args):
        """Execute query and fetch one result."""
        async with self.pool.acquire() as connection:
            return await connection.fetchrow(query, *args)
    
    async def fetch_many(self, query: str, *args):
        """Execute query and fetch multiple results."""
        async with self.pool.acquire() as connection:
            return await connection.fetch(query, *args)

# Cache layer
class CacheLayer:
    """Redis caching layer."""
    
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Connect to Redis."""
        self.redis = await aioredis.from_url(
            "redis://localhost",
            encoding="utf-8",
            decode_responses=True,
            max_connections=50
        )
    
    async def get_or_set(
        self,
        key: str,
        factory_func,
        expire: int = 3600
    ):
        """Get from cache or compute and cache."""
        # Try cache first
        cached = await self.redis.get(key)
        if cached:
            return orjson.loads(cached)
        
        # Compute value
        value = await factory_func()
        
        # Cache it
        await self.redis.setex(
            key,
            expire,
            orjson.dumps(value)
        )
        
        return value

# Global instances
db_pool = DatabasePool()
cache = CacheLayer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await db_pool.create_pool()
    await cache.connect()
    yield
    # Shutdown
    await db_pool.close_pool()
    if cache.redis:
        await cache.redis.close()

app = FastAPI(lifespan=lifespan, default_response_class=ORJSONResponse)

# Optimized endpoint with caching
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Optimized user retrieval with caching."""
    cache_key = f"user:{user_id}"
    
    async def fetch_from_db():
        """Fetch user from database."""
        query = "SELECT id, name, email FROM users WHERE id = $1"
        row = await db_pool.fetch_one(query, user_id)
        if not row:
            return None
        return dict(row)
    
    # Get from cache or database
    user = await cache.get_or_set(cache_key, fetch_from_db, expire=300)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

# Batch endpoint for reducing round trips
@app.post("/users/batch")
async def get_users_batch(user_ids: List[int]):
    """Fetch multiple users in one request."""
    # Use asyncio.gather for concurrent database queries
    async def fetch_user(user_id: int):
        query = "SELECT id, name, email FROM users WHERE id = $1"
        row = await db_pool.fetch_one(query, user_id)
        return dict(row) if row else None
    
    users = await asyncio.gather(*[fetch_user(uid) for uid in user_ids])
    return [u for u in users if u is not None]

# Response streaming for large datasets
from fastapi.responses import StreamingResponse
import asyncio

@app.get("/users/export")
async def export_users():
    """Stream large dataset as CSV."""
    async def generate_csv():
        """Generate CSV data in chunks."""
        yield "id,name,email\n"
        
        # Stream from database in chunks
        query = "SELECT id, name, email FROM users ORDER BY id"
        async with db_pool.pool.acquire() as connection:
            async with connection.transaction():
                async for row in connection.cursor(query):
                    yield f"{row['id']},{row['name']},{row['email']}\n"
                    await asyncio.sleep(0)  # Allow other tasks to run
    
    return StreamingResponse(
        generate_csv(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=users.csv"}
    )

# Pagination for large datasets
from pydantic import BaseModel

class PaginatedResponse(BaseModel):
    """Standardized pagination response."""
    items: List[Dict]
    total: int
    page: int
    page_size: int
    has_next: bool

@app.get("/users", response_model=PaginatedResponse)
async def list_users(page: int = 1, page_size: int = 100):
    """Paginated user listing."""
    offset = (page - 1) * page_size
    
    # Count total (cached)
    cache_key = "users:total_count"
    
    async def get_total():
        query = "SELECT COUNT(*) FROM users"
        row = await db_pool.fetch_one(query)
        return row['count']
    
    total = await cache.get_or_set(cache_key, get_total, expire=60)
    
    # Fetch page
    query = """
        SELECT id, name, email 
        FROM users 
        ORDER BY id 
        LIMIT $1 OFFSET $2
    """
    rows = await db_pool.fetch_many(query, page_size, offset)
    
    return PaginatedResponse(
        items=[dict(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        has_next=offset + page_size < total
    )
```

**Optimization Techniques:**
1. **Connection Pooling**: Reuse database connections
2. **Caching**: Redis for frequently accessed data
3. **Async I/O**: Non-blocking operations
4. **Batch Operations**: Reduce round trips
5. **Response Streaming**: Handle large responses
6. **Pagination**: Limit response size
7. **ORJson**: Faster JSON serialization
8. **Database Indexing**: Optimize queries
9. **CDN**: Static content delivery
10. **Load Balancing**: Distribute traffic

---

## Security & Best Practices

### Q12: Implement OAuth2 with JWT in FastAPI

**Expected Answer:**

```python
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel

# Configuration
SECRET_KEY = "your-secret-key-here"  # Use environment variable in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None
    scopes: List[str] = []

class User(BaseModel):
    username: str
    email: str
    full_name: Optional[str] = None
    disabled: bool = False
    roles: List[str] = []

class UserInDB(User):
    hashed_password: str

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash password."""
    return pwd_context.hash(password)

# Token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def create_refresh_token(data: dict) -> str:
    """Create JWT refresh token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    Validate JWT token and get current user.
    
    Args:
        token: JWT access token
        
    Returns:
        Current user object
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")
        
        if username is None or token_type != "access":
            raise credentials_exception
        
        token_data = TokenData(
            username=username,
            scopes=payload.get("scopes", [])
        )
        
    except JWTError:
        raise credentials_exception
    
    # Fetch user from database
    user = await get_user_from_db(username)
    if user is None:
        raise credentials_exception
    
    if user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    
    return user

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Role-based access control
class RoleChecker:
    """Dependency for role-based access control."""
    
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: User = Depends(get_current_user)):
        """Check if user has required role."""
        if not any(role in user.roles for role in self.allowed_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return user

# Protected endpoints
app = FastAPI()

@app.post("/token", response_model=Token)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint to get access token."""
    user = await authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create tokens
    access_token = create_access_token(
        data={"sub": user.username, "scopes": user.roles}
    )
    refresh_token = create_refresh_token(
        data={"sub": user.username}
    )
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )

@app.get("/users/me")
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user info."""
    return current_user

@app.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    current_user: User = Depends(RoleChecker(["admin", "superuser"]))
):
    """Delete user (admin only)."""
    # Delete logic here
    return {"message": f"User {user_id} deleted"}
```

**Discussion Points:**
- JWT vs session-based authentication
- Token refresh strategies
- Token revocation mechanisms
- RBAC vs ABAC (Attribute-Based Access Control)
- OAuth2 flows (authorization code, client credentials)
- Security headers (CORS, CSP, HSTS)
- Input sanitization and SQL injection prevention

---

## Behavioral & System Design Questions

### Q13: How would you migrate a monolithic application to microservices?

**Expected Steps:**

1. **Assessment Phase**
   - Identify bounded contexts
   - Map dependencies
   - Analyze data flow
   - Identify integration points

2. **Strategy**
   - Strangler Fig pattern (gradual migration)
   - Start with least dependent modules
   - Maintain backward compatibility

3. **Implementation**
```python
"""
Migration Strategy:

Phase 1: Extract Authentication Service
- Create new FastAPI auth service
- Dual-write to old and new system
- Route new requests to new service
- Verify data consistency

Phase 2: Extract User Service
- Create user management service
- Migrate user data
- Update authentication service integration
- Decommission old user module

Phase 3: Extract Business Services
- Identify service boundaries
- Create new services
- Implement event-driven communication
- Migrate traffic gradually

Phase 4: Data Migration
- Implement database per service
- Use event sourcing for data sync
- Eventual consistency handling
"""
```

4. **Challenges & Solutions**
   - **Data consistency**: Event sourcing, saga pattern
   - **Distributed transactions**: 2PC, compensating transactions
   - **Service discovery**: Consul, Kubernetes DNS
   - **Configuration management**: Centralized config service
   - **Monitoring**: Distributed tracing, centralized logging

### Q14: Design a fault-tolerant data pipeline with Airflow

**Architecture:**

```python
"""
Fault-Tolerant Data Pipeline Architecture:

1. REDUNDANCY:
   - Multiple scheduler replicas
   - Auto-scaling workers
   - Database replication (master-slave)
   - Message broker clustering

2. ERROR HANDLING:
   - Task retries with exponential backoff
   - Dead letter queues for failed tasks
   - Alerting on failures
   - Automated recovery procedures

3. DATA INTEGRITY:
   - Idempotent tasks
   - Checkpointing
   - Transaction boundaries
   - Data validation at each stage

4. MONITORING:
   - Task duration metrics
   - Failure rate tracking
   - Resource utilization
   - SLA monitoring

5. DISASTER RECOVERY:
   - Regular backups
   - Multi-region deployment
   - Automated failover
   - Recovery runbooks
"""

# Implementation example
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.trigger_rule import TriggerRule

def process_with_checkpointing(batch_id: str, **context):
    """
    Process data with checkpoint support.
    
    Allows resuming from failure point.
    """
    from airflow.models import Variable
    
    # Get last checkpoint
    checkpoint_key = f"checkpoint:{batch_id}"
    last_processed = Variable.get(checkpoint_key, default_var=0)
    
    try:
        # Process from checkpoint
        for i in range(last_processed, total_records):
            # Process record
            process_record(i)
            
            # Update checkpoint every 1000 records
            if i % 1000 == 0:
                Variable.set(checkpoint_key, i)
        
        # Clear checkpoint on success
        Variable.delete(checkpoint_key)
        
    except Exception as e:
        # Checkpoint is preserved for retry
        raise

with DAG('fault_tolerant_pipeline', **default_args) as dag:
    
    # Task with retries and callbacks
    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_with_checkpointing,
        retries=5,
        retry_delay=timedelta(minutes=5),
        retry_exponential_backoff=True,
        on_failure_callback=send_failure_alert,
        on_retry_callback=send_retry_alert,
        on_success_callback=send_success_alert
    )
    
    # Cleanup task that always runs
    cleanup = BashOperator(
        task_id='cleanup',
        bash_command='rm -f /tmp/processing/*',
        trigger_rule=TriggerRule.ALL_DONE  # Run regardless of upstream status
    )
    
    process_task >> cleanup
```

---

## Additional Topics for Architects

### System Design Scenarios

**Q15: Design a real-time analytics platform**
- Lambda architecture (batch + stream processing)
- FastAPI for real-time queries
- Airflow for batch processing
- Kafka for streaming
- ClickHouse/Druid for OLAP

**Q16: Design a multi-cloud data platform**
- Cloud-agnostic architecture
- Terraform for infrastructure
- Airflow for orchestration across clouds
- Data replication strategies
- Cost optimization

**Q17: Design an ML pipeline with Airflow**
- Feature engineering DAGs
- Model training workflows
- Model serving with FastAPI
- A/B testing infrastructure
- Model monitoring and retraining

---

## 🎯 Preparation Tips for Architects

1. **Focus on Trade-offs**
   - Every architectural decision has trade-offs
   - Be prepared to justify your choices
   - Consider scalability, maintainability, cost

2. **Draw Diagrams**
   - System architecture diagrams
   - Sequence diagrams for interactions
   - Data flow diagrams

3. **Know the Numbers**
   - Throughput requirements (RPS)
   - Latency requirements (p50, p95, p99)
   - Data volume and growth
   - Cost implications

4. **Real-World Experience**
   - Be ready to discuss past projects
   - Challenges faced and solved
   - Lessons learned
   - Metrics and improvements

5. **Stay Current**
   - Latest FastAPI features
   - Airflow 2.x improvements
   - Cloud-native patterns
   - Industry best practices

---

## 📚 Additional Resources

- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Apache Airflow Docs**: https://airflow.apache.org/docs/
- **System Design Primer**: https://github.com/donnemartin/system-design-primer
- **Microservices Patterns**: https://microservices.io/patterns/
- **12-Factor App**: https://12factor.net/

---

**Good luck with your architect interview!** 🚀🏗️

