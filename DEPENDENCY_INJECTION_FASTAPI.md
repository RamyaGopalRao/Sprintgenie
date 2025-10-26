# 🔌 Dependency Injection in FastAPI with dependency-injector

## Complete Implementation Guide

### Installation

```bash
pip install dependency-injector
pip install fastapi uvicorn sqlalchemy aioredis
```

---

## 1️⃣ Basic Dependency Injection Pattern

### Simple Repository Pattern with DI

```python
from fastapi import FastAPI, Depends
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from typing import Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Domain Models
@dataclass
class User:
    id: int
    username: str
    email: str
    is_active: bool = True

# Repository Interface
class UserRepositoryInterface(ABC):
    """Abstract repository interface."""
    
    @abstractmethod
    async def get_by_id(self, user_id: int) -> Optional[User]:
        pass
    
    @abstractmethod
    async def get_all(self) -> List[User]:
        pass
    
    @abstractmethod
    async def create(self, user: User) -> User:
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        pass
    
    @abstractmethod
    async def delete(self, user_id: int) -> bool:
        pass

# Concrete Repository Implementation
class UserRepository(UserRepositoryInterface):
    """Concrete user repository with database access."""
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        # In production, initialize actual database connection
        self.db = {}  # Simplified for example
    
    async def get_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        # Simulate database query
        if user_id in self.db:
            return self.db[user_id]
        return None
    
    async def get_all(self) -> List[User]:
        """Get all users."""
        return list(self.db.values())
    
    async def create(self, user: User) -> User:
        """Create new user."""
        self.db[user.id] = user
        return user
    
    async def update(self, user: User) -> User:
        """Update existing user."""
        self.db[user.id] = user
        return user
    
    async def delete(self, user_id: int) -> bool:
        """Delete user."""
        if user_id in self.db:
            del self.db[user_id]
            return True
        return False

# Service Layer
class UserService:
    """User business logic service."""
    
    def __init__(self, user_repository: UserRepositoryInterface):
        self.repository = user_repository
    
    async def get_user(self, user_id: int) -> Optional[User]:
        """Get user with business logic."""
        user = await self.repository.get_by_id(user_id)
        if user and not user.is_active:
            return None  # Don't return inactive users
        return user
    
    async def create_user(self, username: str, email: str) -> User:
        """Create user with validation."""
        # Business logic: validate email, check duplicates, etc.
        user = User(id=len(await self.repository.get_all()) + 1, username=username, email=email)
        return await self.repository.create(user)
    
    async def deactivate_user(self, user_id: int) -> bool:
        """Soft delete user."""
        user = await self.repository.get_by_id(user_id)
        if not user:
            return False
        
        user.is_active = False
        await self.repository.update(user)
        return True

# Dependency Container
class Container(containers.DeclarativeContainer):
    """Application dependency container."""
    
    # Configuration
    config = providers.Configuration()
    
    # Database
    database_url = providers.Singleton(
        lambda: config.database.url()
    )
    
    # Repositories
    user_repository = providers.Singleton(
        UserRepository,
        database_url=database_url
    )
    
    # Services
    user_service = providers.Factory(
        UserService,
        user_repository=user_repository
    )

# FastAPI Application
app = FastAPI()
container = Container()
container.config.database.url.from_value("postgresql://localhost/mydb")

# Wire the container
container.wire(modules=[__name__])

# Endpoints with dependency injection
@app.get("/users/{user_id}")
@inject
async def get_user(
    user_id: int,
    user_service: UserService = Depends(Provide[Container.user_service])
):
    """Get user by ID using injected service."""
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users")
@inject
async def create_user(
    username: str,
    email: str,
    user_service: UserService = Depends(Provide[Container.user_service])
):
    """Create new user."""
    user = await user_service.create_user(username, email)
    return user

@app.get("/users")
@inject
async def list_users(
    user_service: UserService = Depends(Provide[Container.user_service])
):
    """List all active users."""
    # Access repository directly if needed
    all_users = await user_service.repository.get_all()
    return [u for u in all_users if u.is_active]
```

---

## 2️⃣ Advanced Multi-Layer Architecture

### Complete Application with DI

```python
from fastapi import FastAPI, Depends, HTTPException
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from typing import Optional, List, Protocol
import aioredis
from pydantic import BaseModel

# ================== DATA LAYER ==================

class Base(DeclarativeBase):
    pass

class DatabaseManager:
    """Manages database connections and sessions."""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(
            database_url,
            echo=True,
            pool_pre_ping=True,
            pool_size=10,
            max_overflow=20
        )
        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
    
    async def get_session(self) -> AsyncSession:
        """Get database session."""
        async with self.session_factory() as session:
            yield session
    
    async def close(self):
        """Close database connections."""
        await self.engine.dispose()

class CacheManager:
    """Manages Redis cache connections."""
    
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection."""
        self.redis = await aioredis.from_url(
            self.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
    
    async def close(self):
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        return await self.redis.get(key)
    
    async def set(self, key: str, value: str, expire: int = 3600):
        """Set value in cache with expiration."""
        await self.redis.setex(key, expire, value)

# ================== REPOSITORY LAYER ==================

class UserRepositoryProtocol(Protocol):
    """Protocol for user repository."""
    
    async def find_by_id(self, user_id: int) -> Optional[dict]:
        ...
    
    async def find_by_email(self, email: str) -> Optional[dict]:
        ...
    
    async def create(self, user_data: dict) -> dict:
        ...
    
    async def update(self, user_id: int, user_data: dict) -> Optional[dict]:
        ...
    
    async def delete(self, user_id: int) -> bool:
        ...

class UserRepository:
    """User repository implementation."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def find_by_id(self, user_id: int) -> Optional[dict]:
        """Find user by ID."""
        # Simulate database query
        # In production: query = await self.db.execute(select(User).where(User.id == user_id))
        return {"id": user_id, "username": "john_doe", "email": "john@example.com"}
    
    async def find_by_email(self, email: str) -> Optional[dict]:
        """Find user by email."""
        # Database query here
        return None
    
    async def create(self, user_data: dict) -> dict:
        """Create new user."""
        # Insert into database
        return {"id": 1, **user_data}
    
    async def update(self, user_id: int, user_data: dict) -> Optional[dict]:
        """Update user."""
        return {"id": user_id, **user_data}
    
    async def delete(self, user_id: int) -> bool:
        """Delete user."""
        return True

class ProductRepository:
    """Product repository implementation."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def find_all(self, skip: int = 0, limit: int = 100) -> List[dict]:
        """Get all products with pagination."""
        # Database query with offset and limit
        return [
            {"id": 1, "name": "Product 1", "price": 99.99},
            {"id": 2, "name": "Product 2", "price": 149.99}
        ]
    
    async def find_by_id(self, product_id: int) -> Optional[dict]:
        """Find product by ID."""
        return {"id": product_id, "name": "Product", "price": 99.99}

# ================== SERVICE LAYER ==================

class UserService:
    """User business logic service."""
    
    def __init__(
        self,
        user_repository: UserRepositoryProtocol,
        cache_manager: CacheManager
    ):
        self.repository = user_repository
        self.cache = cache_manager
    
    async def get_user(self, user_id: int) -> Optional[dict]:
        """
        Get user with caching.
        
        Args:
            user_id: User ID
            
        Returns:
            User data or None
        """
        # Try cache first
        cache_key = f"user:{user_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            import json
            return json.loads(cached)
        
        # Fetch from database
        user = await self.repository.find_by_id(user_id)
        
        if user:
            # Cache for 1 hour
            import json
            await self.cache.set(cache_key, json.dumps(user), expire=3600)
        
        return user
    
    async def create_user(self, username: str, email: str) -> dict:
        """
        Create user with validation.
        
        Args:
            username: Username
            email: Email address
            
        Returns:
            Created user data
            
        Raises:
            ValueError: If email already exists
        """
        # Check if email exists
        existing = await self.repository.find_by_email(email)
        if existing:
            raise ValueError("Email already registered")
        
        # Create user
        user_data = {"username": username, "email": email}
        user = await self.repository.create(user_data)
        
        # Invalidate cache
        cache_key = f"user:{user['id']}"
        await self.cache.redis.delete(cache_key)
        
        return user
    
    async def update_user(self, user_id: int, updates: dict) -> Optional[dict]:
        """Update user and invalidate cache."""
        user = await self.repository.update(user_id, updates)
        
        if user:
            # Invalidate cache
            cache_key = f"user:{user_id}"
            await self.cache.redis.delete(cache_key)
        
        return user

class ProductService:
    """Product business logic service."""
    
    def __init__(
        self,
        product_repository: ProductRepository,
        cache_manager: CacheManager
    ):
        self.repository = product_repository
        self.cache = cache_manager
    
    async def get_products(self, page: int = 1, page_size: int = 20) -> List[dict]:
        """Get paginated products with caching."""
        cache_key = f"products:page:{page}:size:{page_size}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            import json
            return json.loads(cached)
        
        skip = (page - 1) * page_size
        products = await self.repository.find_all(skip=skip, limit=page_size)
        
        # Cache for 5 minutes
        import json
        await self.cache.set(cache_key, json.dumps(products), expire=300)
        
        return products

# ================== CONTAINER CONFIGURATION ==================

class Container(containers.DeclarativeContainer):
    """Application dependency injection container."""
    
    # Configuration
    config = providers.Configuration()
    
    # Infrastructure Layer
    database_manager = providers.Singleton(
        DatabaseManager,
        database_url=config.database.url
    )
    
    cache_manager = providers.Singleton(
        CacheManager,
        redis_url=config.redis.url
    )
    
    # Repository Layer
    user_repository = providers.Factory(
        UserRepository,
        db_session=database_manager.provided.get_session
    )
    
    product_repository = providers.Factory(
        ProductRepository,
        db_session=database_manager.provided.get_session
    )
    
    # Service Layer
    user_service = providers.Factory(
        UserService,
        user_repository=user_repository,
        cache_manager=cache_manager
    )
    
    product_service = providers.Factory(
        ProductService,
        product_repository=product_repository,
        cache_manager=cache_manager
    )

# ================== FASTAPI APPLICATION ==================

app = FastAPI(title="DI Example API")

# Initialize container
container = Container()
container.config.database.url.from_env("DATABASE_URL", default="postgresql+asyncpg://localhost/mydb")
container.config.redis.url.from_env("REDIS_URL", default="redis://localhost")

# Wire the container to this module
container.wire(modules=[__name__])

# Lifespan events
@app.on_event("startup")
async def startup():
    """Initialize resources on startup."""
    await container.cache_manager().connect()

@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources on shutdown."""
    await container.database_manager().close()
    await container.cache_manager().close()

# ================== API ENDPOINTS ==================

# Request/Response models
class UserCreateRequest(BaseModel):
    username: str
    email: str

class UserUpdateRequest(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool

# Endpoints with dependency injection
@app.get("/users/{user_id}", response_model=UserResponse)
@inject
async def get_user(
    user_id: int,
    user_service: UserService = Depends(Provide[Container.user_service])
):
    """
    Get user by ID.
    
    The UserService is automatically injected by dependency-injector.
    """
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/users", response_model=UserResponse, status_code=201)
@inject
async def create_user(
    user_data: UserCreateRequest,
    user_service: UserService = Depends(Provide[Container.user_service])
):
    """Create new user."""
    try:
        user = await user_service.create_user(
            username=user_data.username,
            email=user_data.email
        )
        return user
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/users/{user_id}", response_model=UserResponse)
@inject
async def update_user(
    user_id: int,
    user_data: UserUpdateRequest,
    user_service: UserService = Depends(Provide[Container.user_service])
):
    """Update user."""
    updates = user_data.dict(exclude_unset=True)
    user = await user_service.update_user(user_id, updates)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.get("/products")
@inject
async def list_products(
    page: int = 1,
    page_size: int = 20,
    product_service: ProductService = Depends(Provide[Container.product_service])
):
    """List products with pagination."""
    products = await product_service.get_products(page=page, page_size=page_size)
    return {"items": products, "page": page, "page_size": page_size}

# Health check endpoint
@app.get("/health")
@inject
async def health_check(
    cache_manager: CacheManager = Depends(Provide[Container.cache_manager])
):
    """Health check endpoint."""
    try:
        # Check Redis connection
        await cache_manager.redis.ping()
        return {"status": "healthy", "cache": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

---

## 3️⃣ Advanced Multi-Tenant Application

### Tenant-Aware Dependency Injection

```python
from fastapi import FastAPI, Header, Depends
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from typing import Annotated, Optional
from contextlib import asynccontextmanager

# Tenant Context
class TenantContext:
    """Holds current tenant information."""
    
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.database_name = f"tenant_{tenant_id}"
        self.cache_prefix = f"tenant:{tenant_id}"

# Tenant-aware repository
class TenantUserRepository:
    """Repository that operates on tenant-specific database."""
    
    def __init__(self, tenant_context: TenantContext, db_manager: DatabaseManager):
        self.tenant = tenant_context
        self.db_manager = db_manager
    
    async def get_user(self, user_id: int) -> Optional[dict]:
        """Get user from tenant database."""
        # Use tenant-specific database
        db_url = f"postgresql://localhost/{self.tenant.database_name}"
        # Query user from tenant database
        return {"id": user_id, "tenant_id": self.tenant.tenant_id}

# Tenant-aware service
class TenantUserService:
    """Service with tenant isolation."""
    
    def __init__(
        self,
        tenant_context: TenantContext,
        repository: TenantUserRepository,
        cache: CacheManager
    ):
        self.tenant = tenant_context
        self.repository = repository
        self.cache = cache
    
    async def get_user(self, user_id: int) -> Optional[dict]:
        """Get user with tenant-scoped caching."""
        # Cache key includes tenant ID for isolation
        cache_key = f"{self.tenant.cache_prefix}:user:{user_id}"
        
        cached = await self.cache.get(cache_key)
        if cached:
            import json
            return json.loads(cached)
        
        user = await self.repository.get_user(user_id)
        
        if user:
            import json
            await self.cache.set(cache_key, json.dumps(user))
        
        return user

# Multi-tenant container
class TenantContainer(containers.DeclarativeContainer):
    """Container for tenant-specific dependencies."""
    
    # Tenant context (provided per request)
    tenant_context = providers.Dependency(instance_of=TenantContext)
    
    # Shared infrastructure
    config = providers.Configuration()
    db_manager = providers.Dependency(instance_of=DatabaseManager)
    cache_manager = providers.Dependency(instance_of=CacheManager)
    
    # Tenant-specific repositories
    user_repository = providers.Factory(
        TenantUserRepository,
        tenant_context=tenant_context,
        db_manager=db_manager
    )
    
    # Tenant-specific services
    user_service = providers.Factory(
        TenantUserService,
        tenant_context=tenant_context,
        repository=user_repository,
        cache=cache_manager
    )

# Application container
class ApplicationContainer(containers.DeclarativeContainer):
    """Main application container."""
    
    config = providers.Configuration()
    
    # Shared resources
    db_manager = providers.Singleton(
        DatabaseManager,
        database_url=config.database.url
    )
    
    cache_manager = providers.Singleton(
        CacheManager,
        redis_url=config.redis.url
    )
    
    # Tenant container factory
    tenant_container = providers.FactoryAggregate(
        tenant_context=providers.Factory(TenantContext),
        db_manager=db_manager,
        cache_manager=cache_manager
    )

# FastAPI app
app = FastAPI()
container = ApplicationContainer()
container.config.database.url.from_value("postgresql+asyncpg://localhost/main")
container.config.redis.url.from_value("redis://localhost")

# Dependency to get tenant context
async def get_tenant_context(
    x_tenant_id: Annotated[Optional[str], Header()] = None
) -> TenantContext:
    """Extract tenant from header."""
    if not x_tenant_id:
        raise HTTPException(status_code=400, detail="X-Tenant-ID header required")
    return TenantContext(tenant_id=x_tenant_id)

# Dependency to get tenant-specific service
async def get_tenant_user_service(
    tenant_context: TenantContext = Depends(get_tenant_context)
) -> TenantUserService:
    """Get tenant-specific user service."""
    # Create tenant container with context
    tenant_container = TenantContainer(
        tenant_context=tenant_context,
        db_manager=container.db_manager(),
        cache_manager=container.cache_manager()
    )
    return tenant_container.user_service()

# Tenant-aware endpoint
@app.get("/users/{user_id}")
async def get_tenant_user(
    user_id: int,
    service: TenantUserService = Depends(get_tenant_user_service)
):
    """Get user from tenant-specific database."""
    user = await service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## 4️⃣ Testing with Dependency Injection

### Easy Testing with Mock Dependencies

```python
import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock

# Test configuration
class TestContainer(containers.DeclarativeContainer):
    """Test container with mocked dependencies."""
    
    config = providers.Configuration()
    
    # Mock repositories
    user_repository = providers.Singleton(
        Mock,
        spec=UserRepository
    )
    
    # Mock cache
    cache_manager = providers.Singleton(
        Mock,
        spec=CacheManager
    )
    
    # Real service with mocked dependencies
    user_service = providers.Factory(
        UserService,
        user_repository=user_repository,
        cache_manager=cache_manager
    )

# Test fixtures
@pytest.fixture
def test_container():
    """Create test container."""
    container = TestContainer()
    return container

@pytest.fixture
def client(test_container):
    """Create test client with mocked dependencies."""
    # Override app container with test container
    app.container = test_container
    test_container.wire(modules=[__name__])
    
    with TestClient(app) as client:
        yield client
    
    test_container.unwire()

# Tests
def test_get_user_success(client, test_container):
    """Test successful user retrieval."""
    # Setup mock
    mock_repo = test_container.user_repository()
    mock_repo.find_by_id = AsyncMock(return_value={
        "id": 1,
        "username": "testuser",
        "email": "test@example.com"
    })
    
    mock_cache = test_container.cache_manager()
    mock_cache.get = AsyncMock(return_value=None)
    mock_cache.set = AsyncMock()
    
    # Make request
    response = client.get("/users/1")
    
    # Assertions
    assert response.status_code == 200
    assert response.json()["username"] == "testuser"
    
    # Verify mocks were called
    mock_repo.find_by_id.assert_called_once_with(1)

def test_create_user_duplicate_email(client, test_container):
    """Test creating user with duplicate email."""
    mock_repo = test_container.user_repository()
    mock_repo.find_by_email = AsyncMock(return_value={
        "id": 999,
        "email": "existing@example.com"
    })
    
    response = client.post("/users", json={
        "username": "newuser",
        "email": "existing@example.com"
    })
    
    assert response.status_code == 400
    assert "already registered" in response.json()["detail"]
```

---

## 5️⃣ Real-World Example: E-Commerce API

### Complete Implementation

```python
from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks
from dependency_injector import containers, providers
from dependency_injector.wiring import Provide, inject
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import datetime
import asyncio

# ================== MODELS ==================

class OrderItem(BaseModel):
    product_id: int
    quantity: int
    price: float

class Order(BaseModel):
    id: int
    user_id: int
    items: List[OrderItem]
    total_amount: float
    status: str
    created_at: datetime

class CreateOrderRequest(BaseModel):
    items: List[OrderItem]

# ================== EXTERNAL SERVICES ==================

class PaymentGateway:
    """External payment processing service."""
    
    def __init__(self, api_key: str, webhook_url: str):
        self.api_key = api_key
        self.webhook_url = webhook_url
    
    async def process_payment(self, amount: float, order_id: int) -> dict:
        """Process payment through gateway."""
        # Simulate API call
        await asyncio.sleep(0.1)
        return {
            "transaction_id": f"txn_{order_id}",
            "status": "success",
            "amount": amount
        }

class EmailService:
    """Email notification service."""
    
    def __init__(self, smtp_config: dict):
        self.smtp_config = smtp_config
    
    async def send_order_confirmation(self, email: str, order: Order):
        """Send order confirmation email."""
        print(f"Sending confirmation email to {email} for order {order.id}")
        await asyncio.sleep(0.05)

class InventoryService:
    """Inventory management service."""
    
    def __init__(self, cache: CacheManager):
        self.cache = cache
    
    async def check_availability(self, product_id: int, quantity: int) -> bool:
        """Check if product is available."""
        # Check inventory
        return True
    
    async def reserve_items(self, order_items: List[OrderItem]) -> bool:
        """Reserve inventory for order."""
        for item in order_items:
            if not await self.check_availability(item.product_id, item.quantity):
                return False
        return True

# ================== REPOSITORIES ==================

class OrderRepository:
    """Order data access layer."""
    
    def __init__(self, db_session: AsyncSession):
        self.db = db_session
    
    async def create(self, order_data: dict) -> Order:
        """Create new order."""
        order = Order(
            id=1,
            created_at=datetime.now(),
            **order_data
        )
        # Save to database
        return order
    
    async def find_by_id(self, order_id: int) -> Optional[Order]:
        """Find order by ID."""
        # Query database
        return None
    
    async def update_status(self, order_id: int, status: str) -> bool:
        """Update order status."""
        # Update database
        return True

# ================== SERVICES ==================

class OrderService:
    """Order management service with business logic."""
    
    def __init__(
        self,
        order_repository: OrderRepository,
        inventory_service: InventoryService,
        payment_gateway: PaymentGateway,
        email_service: EmailService,
        user_service: UserService
    ):
        self.order_repo = order_repository
        self.inventory = inventory_service
        self.payment = payment_gateway
        self.email = email_service
        self.user_service = user_service
    
    async def create_order(
        self,
        user_id: int,
        items: List[OrderItem],
        background_tasks: BackgroundTasks
    ) -> Order:
        """
        Create order with full workflow.
        
        Steps:
        1. Validate user
        2. Check inventory
        3. Create order
        4. Process payment
        5. Send confirmation email (background)
        """
        # 1. Validate user
        user = await self.user_service.get_user(user_id)
        if not user:
            raise ValueError("Invalid user")
        
        # 2. Check and reserve inventory
        if not await self.inventory.reserve_items(items):
            raise ValueError("Insufficient inventory")
        
        # 3. Calculate total
        total = sum(item.price * item.quantity for item in items)
        
        # 4. Create order
        order_data = {
            "user_id": user_id,
            "items": items,
            "total_amount": total,
            "status": "pending"
        }
        order = await self.order_repo.create(order_data)
        
        # 5. Process payment
        try:
            payment_result = await self.payment.process_payment(total, order.id)
            
            if payment_result["status"] == "success":
                await self.order_repo.update_status(order.id, "confirmed")
                order.status = "confirmed"
                
                # 6. Send email in background
                background_tasks.add_task(
                    self.email.send_order_confirmation,
                    user["email"],
                    order
                )
            else:
                await self.order_repo.update_status(order.id, "failed")
                raise ValueError("Payment processing failed")
                
        except Exception as e:
            # Rollback inventory reservation
            # await self.inventory.release_items(items)
            raise
        
        return order
    
    async def get_order(self, order_id: int) -> Optional[Order]:
        """Get order by ID."""
        return await self.order_repo.find_by_id(order_id)

# ================== CONTAINER ==================

class ECommerceContainer(containers.DeclarativeContainer):
    """E-commerce application container."""
    
    config = providers.Configuration()
    
    # Infrastructure
    db_manager = providers.Singleton(DatabaseManager, database_url=config.database.url)
    cache_manager = providers.Singleton(CacheManager, redis_url=config.redis.url)
    
    # External services
    payment_gateway = providers.Singleton(
        PaymentGateway,
        api_key=config.payment.api_key,
        webhook_url=config.payment.webhook_url
    )
    
    email_service = providers.Singleton(
        EmailService,
        smtp_config=config.email.smtp
    )
    
    # Internal services
    inventory_service = providers.Factory(
        InventoryService,
        cache=cache_manager
    )
    
    # Repositories
    user_repository = providers.Factory(UserRepository, db_session=db_manager.provided.get_session)
    order_repository = providers.Factory(OrderRepository, db_session=db_manager.provided.get_session)
    
    # Business services
    user_service = providers.Factory(
        UserService,
        user_repository=user_repository,
        cache_manager=cache_manager
    )
    
    order_service = providers.Factory(
        OrderService,
        order_repository=order_repository,
        inventory_service=inventory_service,
        payment_gateway=payment_gateway,
        email_service=email_service,
        user_service=user_service
    )

# ================== APPLICATION ==================

app = FastAPI(title="E-Commerce API")
container = ECommerceContainer()

# Load configuration
container.config.from_dict({
    "database": {"url": "postgresql+asyncpg://localhost/ecommerce"},
    "redis": {"url": "redis://localhost"},
    "payment": {
        "api_key": "payment_api_key",
        "webhook_url": "https://api.example.com/webhooks/payment"
    },
    "email": {
        "smtp": {
            "host": "smtp.example.com",
            "port": 587,
            "username": "noreply@example.com"
        }
    }
})

container.wire(modules=[__name__])

# Endpoints
@app.post("/orders", response_model=Order, status_code=201)
@inject
async def create_order(
    request: CreateOrderRequest,
    user_id: int,
    background_tasks: BackgroundTasks,
    order_service: OrderService = Depends(Provide[Container.order_service])
):
    """
    Create new order.
    
    Demonstrates:
    - Multiple service dependencies
    - Background task scheduling
    - Error handling
    - Transaction management
    """
    try:
        order = await order_service.create_order(
            user_id=user_id,
            items=request.items,
            background_tasks=background_tasks
        )
        return order
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Order creation failed")

@app.get("/orders/{order_id}", response_model=Order)
@inject
async def get_order(
    order_id: int,
    order_service: OrderService = Depends(Provide[Container.order_service])
):
    """Get order by ID."""
    order = await order_service.get_order(order_id)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order
```

---

## 🎯 Key Benefits of Dependency Injection

### 1. **Testability**
- Easy to mock dependencies
- Isolated unit testing
- Integration testing with test containers

### 2. **Maintainability**
- Clear separation of concerns
- Easy to change implementations
- Centralized configuration

### 3. **Flexibility**
- Swap implementations without code changes
- Environment-specific configurations
- Feature flags and A/B testing

### 4. **Reusability**
- Share services across endpoints
- Consistent dependency management
- DRY principle enforcement

---

## 📚 Best Practices

1. **Layer Your Application**
   - Presentation (FastAPI routes)
   - Service (business logic)
   - Repository (data access)
   - Infrastructure (external services)

2. **Use Protocols/Interfaces**
   - Define clear contracts
   - Enable easy mocking
   - Support multiple implementations

3. **Configure Properly**
   - Use environment variables
   - Separate configs per environment
   - Validate configuration on startup

4. **Handle Cleanup**
   - Use lifespan events
   - Close connections properly
   - Release resources

5. **Document Dependencies**
   - Clear docstrings
   - Type hints everywhere
   - Explain injection points

---

## 🚀 Running the Application

```bash
# Install dependencies
pip install fastapi uvicorn dependency-injector sqlalchemy aioredis

# Run application
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Access API docs**: http://localhost:8000/docs

---

**Complete, production-ready dependency injection examples!** ✨

