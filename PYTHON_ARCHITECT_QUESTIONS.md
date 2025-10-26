# 🏗️ Python Architect Interview Questions & Answers

## Complete Guide for Senior Python Architect Position

---

## 📚 Table of Contents

1. [Architecture & Design Patterns](#architecture--design-patterns)
2. [System Design & Scalability](#system-design--scalability)
3. [Database Architecture](#database-architecture)
4. [Microservices & APIs](#microservices--apis)
5. [Performance & Optimization](#performance--optimization)
6. [Security Architecture](#security-architecture)
7. [Cloud & DevOps](#cloud--devops)
8. [Team Leadership & Process](#team-leadership--process)

---

## Architecture & Design Patterns

### Q1: Design a pluggable architecture for a Python application

**Expected Answer:**

A pluggable architecture allows adding/removing features without modifying core code. Here's a comprehensive implementation:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Type, Any, Optional
import importlib
import inspect
from dataclasses import dataclass

# ================== PLUGIN INTERFACE ==================

class Plugin(ABC):
    """Base plugin interface that all plugins must implement."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name."""
        pass
    
    @property
    @abstractmethod
    def version(self) -> str:
        """Plugin version."""
        pass
    
    @abstractmethod
    async def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize plugin with configuration.
        
        Args:
            config: Plugin-specific configuration
        """
        pass
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute plugin logic.
        
        Args:
            context: Execution context with input data
            
        Returns:
            Result dictionary
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """Cleanup resources when plugin is unloaded."""
        pass

# Plugin metadata
@dataclass
class PluginMetadata:
    """Plugin metadata information."""
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str]
    enabled: bool = True

# ================== PLUGIN MANAGER ==================

class PluginManager:
    """
    Manages plugin lifecycle and execution.
    
    Features:
    - Dynamic plugin loading
    - Dependency resolution
    - Plugin ordering
    - Error isolation
    """
    
    def __init__(self):
        self.plugins: Dict[str, Plugin] = {}
        self.metadata: Dict[str, PluginMetadata] = {}
        self.execution_order: List[str] = []
    
    async def register_plugin(
        self,
        plugin_class: Type[Plugin],
        config: Dict[str, Any] = None
    ) -> None:
        """
        Register and initialize a plugin.
        
        Args:
            plugin_class: Plugin class to register
            config: Plugin configuration
        """
        plugin = plugin_class()
        plugin_name = plugin.name
        
        # Check if plugin already registered
        if plugin_name in self.plugins:
            raise ValueError(f"Plugin {plugin_name} already registered")
        
        # Initialize plugin
        try:
            await plugin.initialize(config or {})
            self.plugins[plugin_name] = plugin
            print(f"Plugin registered: {plugin_name} v{plugin.version}")
        except Exception as e:
            print(f"Failed to initialize plugin {plugin_name}: {e}")
            raise
    
    async def unregister_plugin(self, plugin_name: str) -> None:
        """Unregister and cleanup a plugin."""
        if plugin_name in self.plugins:
            plugin = self.plugins[plugin_name]
            await plugin.cleanup()
            del self.plugins[plugin_name]
            print(f"Plugin unregistered: {plugin_name}")
    
    async def discover_plugins(self, plugin_directory: str) -> None:
        """
        Automatically discover and load plugins from directory.
        
        Args:
            plugin_directory: Path to plugin directory
        """
        import os
        import sys
        
        # Add plugin directory to path
        sys.path.insert(0, plugin_directory)
        
        # Find all Python files
        for filename in os.listdir(plugin_directory):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]
                
                try:
                    # Import module
                    module = importlib.import_module(module_name)
                    
                    # Find Plugin subclasses
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if issubclass(obj, Plugin) and obj != Plugin:
                            await self.register_plugin(obj)
                            
                except Exception as e:
                    print(f"Failed to load plugin from {filename}: {e}")
    
    async def execute_plugins(
        self,
        context: Dict[str, Any],
        plugin_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute plugins in order.
        
        Args:
            context: Shared execution context
            plugin_names: Specific plugins to execute (or all if None)
            
        Returns:
            Final context after all plugins
        """
        plugins_to_execute = plugin_names or list(self.plugins.keys())
        
        result_context = context.copy()
        
        for plugin_name in plugins_to_execute:
            if plugin_name not in self.plugins:
                print(f"Warning: Plugin {plugin_name} not found")
                continue
            
            plugin = self.plugins[plugin_name]
            
            try:
                # Execute plugin
                plugin_result = await plugin.execute(result_context)
                
                # Merge results into context
                result_context.update(plugin_result)
                
            except Exception as e:
                print(f"Error executing plugin {plugin_name}: {e}")
                # Continue with other plugins (error isolation)
        
        return result_context
    
    def get_plugin_info(self) -> List[Dict[str, Any]]:
        """Get information about all registered plugins."""
        return [
            {
                "name": plugin.name,
                "version": plugin.version,
                "class": plugin.__class__.__name__
            }
            for plugin in self.plugins.values()
        ]

# ================== EXAMPLE PLUGINS ==================

class AuthenticationPlugin(Plugin):
    """Plugin for user authentication."""
    
    @property
    def name(self) -> str:
        return "authentication"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize authentication system."""
        self.jwt_secret = config.get("jwt_secret", "default_secret")
        self.token_expiry = config.get("token_expiry", 3600)
        print(f"Authentication plugin initialized")
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate authentication token."""
        token = context.get("auth_token")
        
        if not token:
            return {"authenticated": False, "user": None}
        
        # Validate token (simplified)
        # In production: decode JWT, verify signature, check expiry
        user = {"id": 1, "username": "john_doe", "role": "admin"}
        
        return {
            "authenticated": True,
            "user": user
        }
    
    async def cleanup(self) -> None:
        """Cleanup authentication resources."""
        print("Authentication plugin cleaned up")

class LoggingPlugin(Plugin):
    """Plugin for request/response logging."""
    
    @property
    def name(self) -> str:
        return "logging"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize logging system."""
        self.log_level = config.get("level", "INFO")
        self.log_file = config.get("file", "app.log")
        print(f"Logging plugin initialized: {self.log_file}")
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Log request information."""
        import logging
        
        logger = logging.getLogger(__name__)
        logger.info(f"Request: {context.get('path')} - User: {context.get('user', {}).get('username')}")
        
        return {"logged": True}
    
    async def cleanup(self) -> None:
        """Cleanup logging resources."""
        print("Logging plugin cleaned up")

class RateLimitPlugin(Plugin):
    """Plugin for rate limiting."""
    
    @property
    def name(self) -> str:
        return "rate_limit"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    async def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize rate limiting."""
        self.max_requests = config.get("max_requests", 100)
        self.time_window = config.get("time_window", 60)
        self.request_counts = {}
        print(f"Rate limit plugin initialized: {self.max_requests} req/{self.time_window}s")
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Check rate limit."""
        user_id = context.get("user", {}).get("id", "anonymous")
        
        # Simplified rate limiting
        current_count = self.request_counts.get(user_id, 0)
        
        if current_count >= self.max_requests:
            return {
                "rate_limited": True,
                "retry_after": self.time_window
            }
        
        self.request_counts[user_id] = current_count + 1
        
        return {
            "rate_limited": False,
            "requests_remaining": self.max_requests - current_count - 1
        }
    
    async def cleanup(self) -> None:
        """Cleanup rate limiting resources."""
        self.request_counts.clear()
        print("Rate limit plugin cleaned up")

# ================== FASTAPI INTEGRATION ==================

from fastapi import FastAPI, Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class PluginMiddleware(BaseHTTPMiddleware):
    """Middleware to execute plugins on each request."""
    
    def __init__(self, app, plugin_manager: PluginManager):
        super().__init__(app)
        self.plugin_manager = plugin_manager
    
    async def dispatch(self, request: Request, call_next):
        """Execute plugins before request processing."""
        # Build context
        context = {
            "path": request.url.path,
            "method": request.method,
            "auth_token": request.headers.get("Authorization", "").replace("Bearer ", ""),
            "client_ip": request.client.host
        }
        
        # Execute plugins
        result_context = await self.plugin_manager.execute_plugins(context)
        
        # Check authentication
        if not result_context.get("authenticated", False):
            raise HTTPException(status_code=401, detail="Unauthorized")
        
        # Check rate limiting
        if result_context.get("rate_limited", False):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Retry after {result_context.get('retry_after')}s"
            )
        
        # Store user in request state
        request.state.user = result_context.get("user")
        request.state.plugin_context = result_context
        
        # Continue with request
        response = await call_next(request)
        
        return response

# Application setup
app = FastAPI(title="Pluggable Architecture Demo")
plugin_manager = PluginManager()

@app.on_event("startup")
async def startup():
    """Initialize plugins on startup."""
    # Register plugins
    await plugin_manager.register_plugin(
        AuthenticationPlugin,
        config={"jwt_secret": "my_secret", "token_expiry": 3600}
    )
    await plugin_manager.register_plugin(
        LoggingPlugin,
        config={"level": "INFO", "file": "app.log"}
    )
    await plugin_manager.register_plugin(
        RateLimitPlugin,
        config={"max_requests": 100, "time_window": 60}
    )
    
    # Add middleware
    app.add_middleware(PluginMiddleware, plugin_manager=plugin_manager)

@app.on_event("shutdown")
async def shutdown():
    """Cleanup plugins on shutdown."""
    for plugin_name in list(plugin_manager.plugins.keys()):
        await plugin_manager.unregister_plugin(plugin_name)

@app.get("/api/data")
async def get_data(request: Request):
    """Protected endpoint using plugins."""
    user = request.state.user
    return {
        "message": "Success",
        "user": user,
        "data": ["item1", "item2", "item3"]
    }

@app.get("/plugins")
async def list_plugins():
    """List all registered plugins."""
    return {"plugins": plugin_manager.get_plugin_info()}
```

**Key Architectural Decisions:**
- **Plugin Interface**: ABC ensures all plugins implement required methods
- **Plugin Manager**: Centralized plugin lifecycle management
- **Middleware Integration**: Seamless FastAPI integration
- **Error Isolation**: Plugin failures don't crash entire application
- **Dynamic Loading**: Plugins can be loaded at runtime
- **Configuration**: Plugin-specific configuration support

---

### Q2: Implement a CQRS (Command Query Responsibility Segregation) pattern

**Expected Answer:**

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generic, TypeVar
from dataclasses import dataclass
from datetime import datetime
import asyncio
from enum import Enum

# ================== DOMAIN MODELS ==================

@dataclass
class User:
    id: int
    username: str
    email: str
    created_at: datetime
    updated_at: datetime

class EventType(Enum):
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"
    USER_DELETED = "user.deleted"

@dataclass
class DomainEvent:
    """Domain event for event sourcing."""
    event_id: str
    event_type: EventType
    aggregate_id: int
    payload: Dict[str, Any]
    timestamp: datetime
    version: int

# ================== COMMAND SIDE (WRITE) ==================

# Commands
@dataclass
class CreateUserCommand:
    """Command to create a new user."""
    username: str
    email: str
    password: str

@dataclass
class UpdateUserCommand:
    """Command to update existing user."""
    user_id: int
    username: Optional[str] = None
    email: Optional[str] = None

@dataclass
class DeleteUserCommand:
    """Command to delete user."""
    user_id: int

# Command Handler Interface
T = TypeVar('T')

class CommandHandler(ABC, Generic[T]):
    """Base command handler."""
    
    @abstractmethod
    async def handle(self, command: T) -> Any:
        """Handle command execution."""
        pass

# Command Handlers
class CreateUserCommandHandler(CommandHandler[CreateUserCommand]):
    """Handles user creation commands."""
    
    def __init__(self, event_store, write_repository):
        self.event_store = event_store
        self.write_repository = write_repository
    
    async def handle(self, command: CreateUserCommand) -> int:
        """
        Handle user creation.
        
        Args:
            command: CreateUserCommand
            
        Returns:
            Created user ID
        """
        # 1. Validate command
        if await self.write_repository.email_exists(command.email):
            raise ValueError("Email already exists")
        
        # 2. Create user in write database
        user_id = await self.write_repository.create_user({
            "username": command.username,
            "email": command.email,
            "password_hash": hash_password(command.password)
        })
        
        # 3. Publish domain event
        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_CREATED,
            aggregate_id=user_id,
            payload={
                "username": command.username,
                "email": command.email
            },
            timestamp=datetime.now(),
            version=1
        )
        
        await self.event_store.append(event)
        
        return user_id

class UpdateUserCommandHandler(CommandHandler[UpdateUserCommand]):
    """Handles user update commands."""
    
    def __init__(self, event_store, write_repository):
        self.event_store = event_store
        self.write_repository = write_repository
    
    async def handle(self, command: UpdateUserCommand) -> bool:
        """Handle user update."""
        # Update write database
        updated = await self.write_repository.update_user(
            command.user_id,
            {
                "username": command.username,
                "email": command.email
            }
        )
        
        if not updated:
            return False
        
        # Publish event
        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.USER_UPDATED,
            aggregate_id=command.user_id,
            payload={
                "username": command.username,
                "email": command.email
            },
            timestamp=datetime.now(),
            version=2
        )
        
        await self.event_store.append(event)
        return True

# ================== QUERY SIDE (READ) ==================

# Queries
@dataclass
class GetUserQuery:
    """Query to get user by ID."""
    user_id: int

@dataclass
class ListUsersQuery:
    """Query to list users."""
    page: int = 1
    page_size: int = 20
    filter_active: bool = True

# Query Handler Interface
class QueryHandler(ABC, Generic[T]):
    """Base query handler."""
    
    @abstractmethod
    async def handle(self, query: T) -> Any:
        """Handle query execution."""
        pass

# Query Handlers (read from optimized read database/cache)
class GetUserQueryHandler(QueryHandler[GetUserQuery]):
    """Handles user retrieval queries."""
    
    def __init__(self, read_repository, cache):
        self.read_repository = read_repository
        self.cache = cache
    
    async def handle(self, query: GetUserQuery) -> Optional[Dict]:
        """
        Handle user retrieval with caching.
        
        Reads from optimized read model.
        """
        # Check cache first
        cache_key = f"user:{query.user_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            import json
            return json.loads(cached)
        
        # Read from read database (denormalized, optimized for queries)
        user = await self.read_repository.get_user(query.user_id)
        
        if user:
            import json
            await self.cache.set(cache_key, json.dumps(user))
        
        return user

class ListUsersQueryHandler(QueryHandler[ListUsersQuery]):
    """Handles user listing queries."""
    
    def __init__(self, read_repository):
        self.read_repository = read_repository
    
    async def handle(self, query: ListUsersQuery) -> Dict:
        """
        Handle user listing.
        
        Uses denormalized read model for fast queries.
        """
        users = await self.read_repository.list_users(
            page=query.page,
            page_size=query.page_size,
            filter_active=query.filter_active
        )
        
        total = await self.read_repository.count_users(query.filter_active)
        
        return {
            "items": users,
            "total": total,
            "page": query.page,
            "page_size": query.page_size
        }

# ================== EVENT STORE ==================

class EventStore:
    """Stores all domain events for event sourcing."""
    
    def __init__(self):
        self.events: List[DomainEvent] = []
        self.subscribers: List[Callable] = []
    
    async def append(self, event: DomainEvent) -> None:
        """Append event to store and notify subscribers."""
        self.events.append(event)
        
        # Notify all subscribers asynchronously
        await asyncio.gather(*[
            subscriber(event)
            for subscriber in self.subscribers
        ])
    
    def subscribe(self, handler: Callable) -> None:
        """Subscribe to domain events."""
        self.subscribers.append(handler)
    
    async def get_events_for_aggregate(
        self,
        aggregate_id: int
    ) -> List[DomainEvent]:
        """Get all events for an aggregate."""
        return [e for e in self.events if e.aggregate_id == aggregate_id]

# ================== READ MODEL PROJECTIONS ==================

class ReadModelProjection:
    """
    Updates read model based on domain events.
    
    This keeps the read database in sync with write database.
    """
    
    def __init__(self, read_repository):
        self.read_repository = read_repository
    
    async def handle_event(self, event: DomainEvent) -> None:
        """Process domain event and update read model."""
        if event.event_type == EventType.USER_CREATED:
            await self.read_repository.insert_user({
                "id": event.aggregate_id,
                **event.payload,
                "created_at": event.timestamp
            })
        
        elif event.event_type == EventType.USER_UPDATED:
            await self.read_repository.update_user(
                event.aggregate_id,
                event.payload
            )
        
        elif event.event_type == EventType.USER_DELETED:
            await self.read_repository.delete_user(event.aggregate_id)

# ================== COMMAND/QUERY BUS ==================

class CommandBus:
    """Routes commands to appropriate handlers."""
    
    def __init__(self):
        self.handlers: Dict[Type, CommandHandler] = {}
    
    def register(self, command_type: Type, handler: CommandHandler) -> None:
        """Register command handler."""
        self.handlers[command_type] = handler
    
    async def execute(self, command: Any) -> Any:
        """Execute command using registered handler."""
        command_type = type(command)
        
        if command_type not in self.handlers:
            raise ValueError(f"No handler registered for {command_type}")
        
        handler = self.handlers[command_type]
        return await handler.handle(command)

class QueryBus:
    """Routes queries to appropriate handlers."""
    
    def __init__(self):
        self.handlers: Dict[Type, QueryHandler] = {}
    
    def register(self, query_type: Type, handler: QueryHandler) -> None:
        """Register query handler."""
        self.handlers[query_type] = handler
    
    async def execute(self, query: Any) -> Any:
        """Execute query using registered handler."""
        query_type = type(query)
        
        if query_type not in self.handlers:
            raise ValueError(f"No handler registered for {query_type}")
        
        handler = self.handlers[query_type]
        return await handler.handle(query)

# ================== APPLICATION SETUP ==================

# Initialize components
event_store = EventStore()
command_bus = CommandBus()
query_bus = QueryBus()

# Setup (simplified - in production, use DI container)
write_repo = MockWriteRepository()
read_repo = MockReadRepository()
cache = MockCache()

# Register command handlers
command_bus.register(
    CreateUserCommand,
    CreateUserCommandHandler(event_store, write_repo)
)
command_bus.register(
    UpdateUserCommand,
    UpdateUserCommandHandler(event_store, write_repo)
)

# Register query handlers
query_bus.register(
    GetUserQuery,
    GetUserQueryHandler(read_repo, cache)
)
query_bus.register(
    ListUsersQuery,
    ListUsersQueryHandler(read_repo)
)

# Setup read model projection
projection = ReadModelProjection(read_repo)
event_store.subscribe(projection.handle_event)

# ================== FASTAPI ENDPOINTS ==================

app = FastAPI(title="CQRS Example")

# Command endpoints (writes)
@app.post("/users", status_code=201)
async def create_user(username: str, email: str, password: str):
    """Create user using command."""
    command = CreateUserCommand(
        username=username,
        email=email,
        password=password
    )
    
    try:
        user_id = await command_bus.execute(command)
        return {"id": user_id, "message": "User created"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.put("/users/{user_id}")
async def update_user(user_id: int, username: str = None, email: str = None):
    """Update user using command."""
    command = UpdateUserCommand(
        user_id=user_id,
        username=username,
        email=email
    )
    
    success = await command_bus.execute(command)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {"message": "User updated"}

# Query endpoints (reads)
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user using query."""
    query = GetUserQuery(user_id=user_id)
    user = await query_bus.execute(query)
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.get("/users")
async def list_users(page: int = 1, page_size: int = 20):
    """List users using query."""
    query = ListUsersQuery(page=page, page_size=page_size)
    result = await query_bus.execute(query)
    return result
```

**CQRS Benefits:**
- **Scalability**: Scale read and write independently
- **Performance**: Optimize read models for queries
- **Flexibility**: Different databases for read/write
- **Event Sourcing**: Complete audit trail
- **Eventual Consistency**: Acceptable for many use cases

**When to Use CQRS:**
- ✅ High read-to-write ratio
- ✅ Complex domain logic
- ✅ Need for audit trail
- ✅ Different query patterns than write
- ❌ Simple CRUD applications
- ❌ Strong consistency required

---

### Q3: Design a clean architecture (Hexagonal/Onion) in Python

**Expected Answer:**

```python
"""
Clean Architecture / Hexagonal Architecture

Layers (from inside out):
1. Domain Layer (Entities, Value Objects)
2. Use Cases Layer (Business Logic)
3. Interface Adapters (Controllers, Presenters, Gateways)
4. Infrastructure (Frameworks, Drivers, External Services)

Dependency Rule: Dependencies point inward
- Outer layers depend on inner layers
- Inner layers know nothing about outer layers
"""

# ================== 1. DOMAIN LAYER (Core) ==================

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime

# Entities
@dataclass
class User:
    """
    Domain entity - represents core business concept.
    
    Contains business logic and invariants.
    """
    id: Optional[int] = None
    username: str = ""
    email: str = ""
    password_hash: str = ""
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    
    def activate(self) -> None:
        """Business logic: activate user."""
        self.is_active = True
    
    def deactivate(self) -> None:
        """Business logic: deactivate user."""
        if not self.can_deactivate():
            raise ValueError("User cannot be deactivated")
        self.is_active = False
    
    def can_deactivate(self) -> bool:
        """Business rule: check if user can be deactivated."""
        # Example business rule
        return not self.is_admin()
    
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return "admin" in getattr(self, 'roles', [])

# Value Objects
@dataclass(frozen=True)
class Email:
    """Value object for email address with validation."""
    value: str
    
    def __post_init__(self):
        """Validate email on creation."""
        if '@' not in self.value:
            raise ValueError("Invalid email address")
    
    def domain(self) -> str:
        """Get email domain."""
        return self.value.split('@')[1]

@dataclass(frozen=True)
class Money:
    """Value object for money with currency."""
    amount: float
    currency: str = "USD"
    
    def add(self, other: 'Money') -> 'Money':
        """Add money (must be same currency)."""
        if self.currency != other.currency:
            raise ValueError("Cannot add different currencies")
        return Money(self.amount + other.amount, self.currency)
    
    def multiply(self, factor: float) -> 'Money':
        """Multiply money by factor."""
        return Money(self.amount * factor, self.currency)

# Repository Interface (defined in domain, implemented in infrastructure)
class UserRepositoryInterface(ABC):
    """
    Repository interface - domain layer defines contract.
    
    Infrastructure layer implements actual data access.
    """
    
    @abstractmethod
    async def find_by_id(self, user_id: int) -> Optional[User]:
        """Find user by ID."""
        pass
    
    @abstractmethod
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        pass
    
    @abstractmethod
    async def save(self, user: User) -> User:
        """Save user."""
        pass
    
    @abstractmethod
    async def delete(self, user_id: int) -> bool:
        """Delete user."""
        pass

# ================== 2. USE CASES LAYER (Application) ==================

class CreateUserUseCase:
    """
    Use case for creating a user.
    
    Contains application-specific business logic.
    Orchestrates domain entities and repository.
    """
    
    def __init__(self, user_repository: UserRepositoryInterface):
        self.user_repository = user_repository
    
    async def execute(
        self,
        username: str,
        email_str: str,
        password: str
    ) -> User:
        """
        Execute user creation use case.
        
        Args:
            username: Username
            email_str: Email address
            password: Plain text password
            
        Returns:
            Created user
            
        Raises:
            ValueError: If validation fails
        """
        # 1. Create value objects
        email = Email(email_str)
        
        # 2. Business rule: check if email already exists
        existing_user = await self.user_repository.find_by_email(email)
        if existing_user:
            raise ValueError("Email already registered")
        
        # 3. Create domain entity
        user = User(
            username=username,
            email=email.value,
            password_hash=self._hash_password(password)
        )
        
        # 4. Save through repository
        saved_user = await self.user_repository.save(user)
        
        # 5. Domain event (could trigger email, etc.)
        # await self.event_publisher.publish(UserCreatedEvent(saved_user))
        
        return saved_user
    
    def _hash_password(self, password: str) -> str:
        """Hash password securely."""
        from passlib.context import CryptContext
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        return pwd_context.hash(password)

class GetUserUseCase:
    """Use case for retrieving user."""
    
    def __init__(self, user_repository: UserRepositoryInterface):
        self.user_repository = user_repository
    
    async def execute(self, user_id: int) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User or None
        """
        user = await self.user_repository.find_by_id(user_id)
        
        # Business logic: don't return inactive users
        if user and not user.is_active:
            return None
        
        return user

class DeactivateUserUseCase:
    """Use case for deactivating user."""
    
    def __init__(self, user_repository: UserRepositoryInterface):
        self.user_repository = user_repository
    
    async def execute(self, user_id: int) -> bool:
        """
        Deactivate user.
        
        Args:
            user_id: User ID
            
        Returns:
            Success status
            
        Raises:
            ValueError: If business rules violated
        """
        # 1. Load user
        user = await self.user_repository.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        
        # 2. Apply business logic (domain method)
        user.deactivate()  # May raise ValueError
        
        # 3. Save updated user
        await self.user_repository.save(user)
        
        return True

# ================== 3. INTERFACE ADAPTERS ==================

# Controllers (FastAPI endpoints)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticModel

# Request/Response DTOs
class CreateUserRequest(PydanticModel):
    """DTO for user creation request."""
    username: str
    email: str
    password: str

class UserResponse(PydanticModel):
    """DTO for user response."""
    id: int
    username: str
    email: str
    is_active: bool
    created_at: datetime

# Controller
class UserController:
    """
    Controller - adapts HTTP requests to use cases.
    
    Part of interface adapter layer.
    """
    
    def __init__(
        self,
        create_user_use_case: CreateUserUseCase,
        get_user_use_case: GetUserUseCase,
        deactivate_user_use_case: DeactivateUserUseCase
    ):
        self.create_user = create_user_use_case
        self.get_user = get_user_use_case
        self.deactivate_user = deactivate_user_use_case
    
    async def create(self, request: CreateUserRequest) -> UserResponse:
        """Create user endpoint."""
        try:
            user = await self.create_user.execute(
                username=request.username,
                email=request.email,
                password=request.password
            )
            
            return UserResponse(
                id=user.id,
                username=user.username,
                email=user.email,
                is_active=user.is_active,
                created_at=user.created_at
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    async def get(self, user_id: int) -> UserResponse:
        """Get user endpoint."""
        user = await self.get_user.execute(user_id)
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            created_at=user.created_at
        )
    
    async def deactivate(self, user_id: int) -> Dict:
        """Deactivate user endpoint."""
        try:
            await self.deactivate_user.execute(user_id)
            return {"message": "User deactivated"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

# ================== 4. INFRASTRUCTURE LAYER ==================

# Repository Implementation
class SQLAlchemyUserRepository(UserRepositoryInterface):
    """
    Concrete repository implementation using SQLAlchemy.
    
    This is in the infrastructure layer.
    """
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def find_by_id(self, user_id: int) -> Optional[User]:
        """Find user in database."""
        # SQLAlchemy query
        # result = await self.db.execute(select(UserModel).where(UserModel.id == user_id))
        # db_user = result.scalar_one_or_none()
        
        # Convert database model to domain entity
        db_user = {"id": user_id, "username": "john", "email": "john@example.com"}
        
        if not db_user:
            return None
        
        return User(
            id=db_user["id"],
            username=db_user["username"],
            email=db_user["email"]
        )
    
    async def find_by_email(self, email: Email) -> Optional[User]:
        """Find user by email."""
        # Database query
        return None
    
    async def save(self, user: User) -> User:
        """Save user to database."""
        # Convert domain entity to database model
        # Persist to database
        return user
    
    async def delete(self, user_id: int) -> bool:
        """Delete user from database."""
        return True

# FastAPI Application (Infrastructure)
app = FastAPI(title="Clean Architecture Example")

# Dependency injection (simplified)
async def get_user_controller() -> UserController:
    """Create user controller with dependencies."""
    db_session = None  # Get from connection pool
    repository = SQLAlchemyUserRepository(db_session)
    
    create_use_case = CreateUserUseCase(repository)
    get_use_case = GetUserUseCase(repository)
    deactivate_use_case = DeactivateUserUseCase(repository)
    
    return UserController(create_use_case, get_use_case, deactivate_use_case)

# Routes
@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user_endpoint(
    request: CreateUserRequest,
    controller: UserController = Depends(get_user_controller)
):
    """Create user endpoint."""
    return await controller.create(request)

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user_endpoint(
    user_id: int,
    controller: UserController = Depends(get_user_controller)
):
    """Get user endpoint."""
    return await controller.get(user_id)

@app.delete("/users/{user_id}")
async def deactivate_user_endpoint(
    user_id: int,
    controller: UserController = Depends(get_user_controller)
):
    """Deactivate user endpoint."""
    return await controller.deactivate(user_id)
```

**Clean Architecture Benefits:**
- **Independence**: Business logic independent of frameworks
- **Testability**: Easy to test without infrastructure
- **Flexibility**: Swap databases, frameworks without changing core
- **Maintainability**: Clear separation of concerns
- **Domain Focus**: Business rules are central

**Layer Responsibilities:**
- **Domain**: Business entities, rules, logic
- **Use Cases**: Application-specific business rules
- **Interface Adapters**: Convert data between layers
- **Infrastructure**: Frameworks, databases, external services

---

## System Design & Scalability

### Q4: Design a distributed caching strategy for a Python application

**Expected Answer:**

```python
from typing import Optional, Any, Callable
import asyncio
import hashlib
import json
from datetime import timedelta
from enum import Enum

class CacheStrategy(Enum):
    """Cache invalidation strategies."""
    TIME_BASED = "time_based"
    EVENT_BASED = "event_based"
    WRITE_THROUGH = "write_through"
    WRITE_BEHIND = "write_behind"

class CacheLevel(Enum):
    """Cache hierarchy levels."""
    L1_MEMORY = "l1_memory"  # In-process cache
    L2_REDIS = "l2_redis"    # Distributed cache
    L3_CDN = "l3_cdn"        # Edge cache

class DistributedCacheManager:
    """
    Multi-level distributed cache implementation.
    
    Architecture:
    L1 (Local Memory) -> L2 (Redis) -> L3 (Database)
    
    Features:
    - Multi-level caching
    - Cache warming
    - Stampede protection
    - TTL management
    - Cache invalidation
    """
    
    def __init__(
        self,
        redis_client,
        local_cache_size: int = 1000,
        default_ttl: int = 3600
    ):
        self.redis = redis_client
        self.default_ttl = default_ttl
        
        # L1 Cache (local memory with LRU)
        from cachetools import LRUCache
        self.local_cache = LRUCache(maxsize=local_cache_size)
        
        # Lock for stampede protection
        self.locks: Dict[str, asyncio.Lock] = {}
    
    def _get_cache_key(self, key: str, namespace: str = "default") -> str:
        """Generate namespaced cache key."""
        return f"{namespace}:{key}"
    
    async def get(
        self,
        key: str,
        namespace: str = "default",
        use_local: bool = True
    ) -> Optional[Any]:
        """
        Get value from cache with multi-level lookup.
        
        Args:
            key: Cache key
            namespace: Cache namespace
            use_local: Whether to check local cache first
            
        Returns:
            Cached value or None
        """
        cache_key = self._get_cache_key(key, namespace)
        
        # L1: Check local cache
        if use_local and cache_key in self.local_cache:
            return self.local_cache[cache_key]
        
        # L2: Check Redis
        try:
            redis_value = await self.redis.get(cache_key)
            if redis_value:
                value = json.loads(redis_value)
                
                # Populate L1 cache
                if use_local:
                    self.local_cache[cache_key] = value
                
                return value
        except Exception as e:
            print(f"Redis error: {e}")
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> None:
        """
        Set value in all cache levels.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            namespace: Cache namespace
        """
        cache_key = self._get_cache_key(key, namespace)
        ttl = ttl or self.default_ttl
        
        # Store in local cache
        self.local_cache[cache_key] = value
        
        # Store in Redis
        try:
            await self.redis.setex(
                cache_key,
                ttl,
                json.dumps(value, default=str)
            )
        except Exception as e:
            print(f"Redis set error: {e}")
    
    async def delete(self, key: str, namespace: str = "default") -> None:
        """Delete from all cache levels."""
        cache_key = self._get_cache_key(key, namespace)
        
        # Remove from local cache
        self.local_cache.pop(cache_key, None)
        
        # Remove from Redis
        try:
            await self.redis.delete(cache_key)
        except Exception as e:
            print(f"Redis delete error: {e}")
    
    async def get_or_compute(
        self,
        key: str,
        factory: Callable,
        ttl: Optional[int] = None,
        namespace: str = "default"
    ) -> Any:
        """
        Get from cache or compute if not exists.
        
        Includes stampede protection.
        
        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time to live
            namespace: Cache namespace
            
        Returns:
            Cached or computed value
        """
        # Try to get from cache
        cached = await self.get(key, namespace)
        if cached is not None:
            return cached
        
        # Stampede protection: ensure only one process computes
        lock_key = f"lock:{namespace}:{key}"
        
        if lock_key not in self.locks:
            self.locks[lock_key] = asyncio.Lock()
        
        async with self.locks[lock_key]:
            # Double-check cache (another process may have computed)
            cached = await self.get(key, namespace, use_local=False)
            if cached is not None:
                return cached
            
            # Compute value
            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()
            
            # Store in cache
            await self.set(key, value, ttl, namespace)
            
            return value
    
    async def invalidate_pattern(self, pattern: str, namespace: str = "default") -> int:
        """
        Invalidate all keys matching pattern.
        
        Args:
            pattern: Key pattern (e.g., "user:*")
            namespace: Cache namespace
            
        Returns:
            Number of keys invalidated
        """
        full_pattern = self._get_cache_key(pattern, namespace)
        
        # Clear from local cache
        keys_to_remove = [k for k in self.local_cache.keys() if k.startswith(full_pattern.replace("*", ""))]
        for k in keys_to_remove:
            del self.local_cache[k]
        
        # Clear from Redis
        try:
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self.redis.scan(cursor, match=full_pattern, count=100)
                if keys:
                    await self.redis.delete(*keys)
                    deleted_count += len(keys)
                
                if cursor == 0:
                    break
            
            return deleted_count
        except Exception as e:
            print(f"Redis pattern delete error: {e}")
            return len(keys_to_remove)
    
    async def warm_cache(
        self,
        keys: List[str],
        factory: Callable[[str], Any],
        namespace: str = "default"
    ) -> None:
        """
        Warm cache with pre-computed values.
        
        Args:
            keys: List of keys to warm
            factory: Function to compute value for each key
            namespace: Cache namespace
        """
        async def warm_key(key: str):
            value = await factory(key) if asyncio.iscoroutinefunction(factory) else factory(key)
            await self.set(key, value, namespace=namespace)
        
        # Warm cache concurrently
        await asyncio.gather(*[warm_key(key) for key in keys])

# ================== CACHE ASIDE PATTERN ==================

class CacheAsideService:
    """Service implementing cache-aside pattern."""
    
    def __init__(self, cache: DistributedCacheManager, repository):
        self.cache = cache
        self.repository = repository
    
    async def get_user(self, user_id: int) -> Optional[Dict]:
        """
        Get user with cache-aside pattern.
        
        Flow:
        1. Check cache
        2. If miss, fetch from database
        3. Store in cache
        4. Return value
        """
        cache_key = f"user:{user_id}"
        
        # Use get_or_compute for automatic cache management
        async def fetch_from_db():
            return await self.repository.find_by_id(user_id)
        
        user = await self.cache.get_or_compute(
            key=cache_key,
            factory=fetch_from_db,
            ttl=3600,
            namespace="users"
        )
        
        return user
    
    async def update_user(self, user_id: int, updates: Dict) -> Dict:
        """
        Update user with write-through caching.
        
        Flow:
        1. Update database
        2. Update cache
        3. Return value
        """
        # Update database
        user = await self.repository.update(user_id, updates)
        
        # Update cache (write-through)
        cache_key = f"user:{user_id}"
        await self.cache.set(cache_key, user, namespace="users")
        
        return user
    
    async def delete_user(self, user_id: int) -> bool:
        """
        Delete user and invalidate cache.
        
        Flow:
        1. Delete from database
        2. Invalidate cache
        """
        success = await self.repository.delete(user_id)
        
        if success:
            # Invalidate cache
            cache_key = f"user:{user_id}"
            await self.cache.delete(cache_key, namespace="users")
        
        return success

# ================== CACHE WARMING STRATEGY ==================

async def warm_user_cache(cache: DistributedCacheManager):
    """Warm cache with frequently accessed users."""
    # Identify hot users (top 1000 most accessed)
    hot_user_ids = await get_hot_user_ids(limit=1000)
    
    async def fetch_user(user_id: str):
        # Fetch from database
        return await repository.find_by_id(int(user_id))
    
    # Warm cache concurrently
    await cache.warm_cache(
        keys=[str(uid) for uid in hot_user_ids],
        factory=fetch_user,
        namespace="users"
    )
```

**Caching Strategies Discussion:**

1. **Cache-Aside (Lazy Loading)**
   - Application manages cache
   - Best for: Read-heavy workloads
   - Pros: Only caches what's needed
   - Cons: Cache miss penalty

2. **Write-Through**
   - Write to cache and database simultaneously
   - Best for: Write-heavy with consistent reads
   - Pros: Cache always up-to-date
   - Cons: Write latency

3. **Write-Behind (Write-Back)**
   - Write to cache, async write to database
   - Best for: High write throughput
   - Pros: Low write latency
   - Cons: Risk of data loss

4. **Cache Warming**
   - Pre-populate cache on startup
   - Best for: Predictable access patterns
   - Pros: No cold start
   - Cons: Memory usage

---

### Q5: How would you design a zero-downtime deployment strategy?

**Expected Answer:**

```python
"""
Zero-Downtime Deployment Strategy

1. BLUE-GREEN DEPLOYMENT:
   - Two identical environments (blue/production, green/staging)
   - Deploy to green environment
   - Run health checks on green
   - Switch traffic to green
   - Keep blue as rollback option

2. ROLLING DEPLOYMENT:
   - Gradually replace instances
   - Update 10-20% at a time
   - Monitor metrics between batches
   - Rollback if issues detected

3. CANARY DEPLOYMENT:
   - Deploy to small subset (5-10%)
   - Monitor metrics closely
   - Gradually increase traffic
   - Full rollout if successful

4. FEATURE FLAGS:
   - Deploy code with features disabled
   - Enable features gradually
   - Instant rollback by toggling flag
"""

# Feature Flag Implementation
class FeatureFlagManager:
    """
    Feature flag system for gradual rollouts.
    
    Allows deploying code without activating features.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_enabled(
        self,
        feature_name: str,
        user_id: Optional[int] = None,
        default: bool = False
    ) -> bool:
        """
        Check if feature is enabled.
        
        Supports:
        - Global enable/disable
        - Percentage rollout
        - User-specific enable
        - A/B testing
        """
        # Check global flag
        global_flag = await self.redis.get(f"feature:{feature_name}")
        
        if global_flag == "disabled":
            return False
        
        if global_flag == "enabled":
            return True
        
        # Check percentage rollout
        rollout_pct = await self.redis.get(f"feature:{feature_name}:rollout_pct")
        if rollout_pct:
            rollout_pct = int(rollout_pct)
            
            if user_id:
                # Consistent hashing for stable user assignment
                user_hash = int(hashlib.md5(str(user_id).encode()).hexdigest(), 16)
                if (user_hash % 100) < rollout_pct:
                    return True
        
        # Check user-specific override
        if user_id:
            user_flag = await self.redis.get(f"feature:{feature_name}:user:{user_id}")
            if user_flag == "enabled":
                return True
        
        return default
    
    async def enable_feature(self, feature_name: str) -> None:
        """Enable feature globally."""
        await self.redis.set(f"feature:{feature_name}", "enabled")
    
    async def disable_feature(self, feature_name: str) -> None:
        """Disable feature globally."""
        await self.redis.set(f"feature:{feature_name}", "disabled")
    
    async def set_rollout_percentage(self, feature_name: str, percentage: int) -> None:
        """Set percentage rollout (0-100)."""
        if not 0 <= percentage <= 100:
            raise ValueError("Percentage must be between 0 and 100")
        
        await self.redis.set(f"feature:{feature_name}:rollout_pct", str(percentage))
    
    async def enable_for_user(self, feature_name: str, user_id: int) -> None:
        """Enable feature for specific user."""
        await self.redis.set(f"feature:{feature_name}:user:{user_id}", "enabled")

# Health Check System
from fastapi import FastAPI
from typing import Dict, List

class HealthChecker:
    """
    Comprehensive health check system.
    
    Used for deployment verification and load balancer health checks.
    """
    
    def __init__(self):
        self.checks: Dict[str, Callable] = {}
    
    def register_check(self, name: str, check_func: Callable) -> None:
        """Register a health check."""
        self.checks[name] = check_func
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """
        Run all health checks concurrently.
        
        Returns:
            Health status for all components
        """
        results = {}
        
        for name, check_func in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result
                }
            except Exception as e:
                results[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Overall status
        all_healthy = all(r["status"] == "healthy" for r in results.values())
        
        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }

# Application with health checks
app = FastAPI()
health_checker = HealthChecker()

# Register health checks
async def check_database():
    """Check database connectivity."""
    try:
        # Execute simple query
        result = await db.execute("SELECT 1")
        return True
    except:
        return False

async def check_redis():
    """Check Redis connectivity."""
    try:
        await redis.ping()
        return True
    except:
        return False

async def check_external_api():
    """Check external API availability."""
    try:
        response = await http_client.get("https://api.example.com/health")
        return response.status_code == 200
    except:
        return False

health_checker.register_check("database", check_database)
health_checker.register_check("redis", check_redis)
health_checker.register_check("external_api", check_external_api)

# Health endpoints
@app.get("/health")
async def health_check():
    """
    Detailed health check for monitoring.
    
    Used by: Load balancers, monitoring systems
    """
    health = await health_checker.run_all_checks()
    status_code = 200 if health["status"] == "healthy" else 503
    return JSONResponse(content=health, status_code=status_code)

@app.get("/health/live")
async def liveness_check():
    """
    Liveness probe for Kubernetes.
    
    Returns 200 if application is running.
    """
    return {"status": "alive"}

@app.get("/health/ready")
async def readiness_check():
    """
    Readiness probe for Kubernetes.
    
    Returns 200 if application is ready to serve traffic.
    """
    # Check critical dependencies
    try:
        await check_database()
        await check_redis()
        return {"status": "ready"}
    except:
        return JSONResponse(
            content={"status": "not ready"},
            status_code=503
        )

# Graceful shutdown
import signal

class GracefulShutdown:
    """
    Handle graceful shutdown during deployment.
    
    Ensures in-flight requests complete before shutdown.
    """
    
    def __init__(self, app: FastAPI, shutdown_timeout: int = 30):
        self.app = app
        self.shutdown_timeout = shutdown_timeout
        self.is_shutting_down = False
    
    def setup(self):
        """Setup signal handlers."""
        signal.signal(signal.SIGTERM, self.handle_sigterm)
        signal.signal(signal.SIGINT, self.handle_sigterm)
    
    def handle_sigterm(self, signum, frame):
        """Handle termination signal."""
        print("Received shutdown signal. Gracefully shutting down...")
        self.is_shutting_down = True
        
        # Stop accepting new requests
        # Wait for in-flight requests to complete
        # Close database connections
        # Flush logs
    
    @app.middleware("http")
    async def reject_during_shutdown(self, request: Request, call_next):
        """Reject new requests during shutdown."""
        if self.is_shutting_down:
            return JSONResponse(
                status_code=503,
                content={"error": "Server is shutting down"}
            )
        return await call_next(request)
```

**Deployment Architecture:**

```yaml
# Kubernetes Deployment Strategy

# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: python-app
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1        # Add 1 extra pod during update
      maxUnavailable: 0  # Never go below 3 pods
  template:
    spec:
      containers:
      - name: app
        image: myapp:v2.0
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 15
          periodSeconds: 10
        lifecycle:
          preStop:
            exec:
              command: ["/bin/sh", "-c", "sleep 15"]  # Allow load balancer to deregister
```

---

### Q6: Design a event-sourcing system with CQRS

**Expected Answer:**

```python
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

# ================== EVENT SOURCING ==================

class EventType(Enum):
    """Domain event types."""
    ORDER_CREATED = "order.created"
    ITEM_ADDED = "order.item_added"
    PAYMENT_PROCESSED = "order.payment_processed"
    ORDER_SHIPPED = "order.shipped"
    ORDER_DELIVERED = "order.delivered"
    ORDER_CANCELLED = "order.cancelled"

@dataclass
class DomainEvent:
    """
    Immutable domain event.
    
    All state changes are represented as events.
    """
    event_id: str
    event_type: EventType
    aggregate_id: str
    aggregate_type: str
    payload: Dict[str, Any]
    metadata: Dict[str, Any]
    timestamp: datetime
    version: int
    user_id: Optional[int] = None

@dataclass
class OrderAggregate:
    """
    Aggregate root reconstructed from events.
    
    Current state is derived from event history.
    """
    id: str
    items: List[Dict] = field(default_factory=list)
    total_amount: float = 0.0
    status: str = "draft"
    created_at: Optional[datetime] = None
    version: int = 0
    
    def apply_event(self, event: DomainEvent) -> None:
        """
        Apply event to aggregate to rebuild state.
        
        Args:
            event: Domain event to apply
        """
        if event.event_type == EventType.ORDER_CREATED:
            self.id = event.aggregate_id
            self.created_at = event.timestamp
            self.status = "created"
        
        elif event.event_type == EventType.ITEM_ADDED:
            self.items.append(event.payload["item"])
            self.total_amount += event.payload["item"]["price"] * event.payload["item"]["quantity"]
        
        elif event.event_type == EventType.PAYMENT_PROCESSED:
            self.status = "paid"
        
        elif event.event_type == EventType.ORDER_SHIPPED:
            self.status = "shipped"
        
        elif event.event_type == EventType.ORDER_DELIVERED:
            self.status = "delivered"
        
        elif event.event_type == EventType.ORDER_CANCELLED:
            self.status = "cancelled"
        
        self.version = event.version

class EventStore:
    """
    Persists all domain events.
    
    Acts as source of truth for system state.
    """
    
    def __init__(self, database):
        self.database = database
        self.subscribers: List[Callable] = []
    
    async def append(self, event: DomainEvent) -> None:
        """
        Append event to store.
        
        Args:
            event: Domain event to persist
        """
        # Persist to database (append-only)
        await self.database.insert_event({
            "event_id": event.event_id,
            "event_type": event.event_type.value,
            "aggregate_id": event.aggregate_id,
            "aggregate_type": event.aggregate_type,
            "payload": json.dumps(event.payload),
            "metadata": json.dumps(event.metadata),
            "timestamp": event.timestamp,
            "version": event.version,
            "user_id": event.user_id
        })
        
        # Notify subscribers (for read model updates)
        await self._notify_subscribers(event)
    
    async def get_events(
        self,
        aggregate_id: str,
        from_version: int = 0
    ) -> List[DomainEvent]:
        """
        Get all events for an aggregate.
        
        Args:
            aggregate_id: Aggregate identifier
            from_version: Start from specific version
            
        Returns:
            List of domain events
        """
        rows = await self.database.query_events(
            aggregate_id=aggregate_id,
            from_version=from_version
        )
        
        return [self._row_to_event(row) for row in rows]
    
    async def rebuild_aggregate(self, aggregate_id: str) -> OrderAggregate:
        """
        Rebuild aggregate from event history.
        
        Args:
            aggregate_id: Aggregate to rebuild
            
        Returns:
            Fully reconstructed aggregate
        """
        events = await self.get_events(aggregate_id)
        
        aggregate = OrderAggregate(id=aggregate_id)
        
        for event in events:
            aggregate.apply_event(event)
        
        return aggregate
    
    def subscribe(self, handler: Callable) -> None:
        """Subscribe to all events."""
        self.subscribers.append(handler)
    
    async def _notify_subscribers(self, event: DomainEvent) -> None:
        """Notify all subscribers of new event."""
        await asyncio.gather(*[
            handler(event) if asyncio.iscoroutinefunction(handler) 
            else asyncio.to_thread(handler, event)
            for handler in self.subscribers
        ])
    
    def _row_to_event(self, row: Dict) -> DomainEvent:
        """Convert database row to domain event."""
        return DomainEvent(
            event_id=row["event_id"],
            event_type=EventType(row["event_type"]),
            aggregate_id=row["aggregate_id"],
            aggregate_type=row["aggregate_type"],
            payload=json.loads(row["payload"]),
            metadata=json.loads(row["metadata"]),
            timestamp=row["timestamp"],
            version=row["version"],
            user_id=row["user_id"]
        )

# ================== COMMAND HANDLERS (WRITE SIDE) ==================

class CreateOrderCommandHandler:
    """Handles order creation command."""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(self, user_id: int) -> str:
        """
        Create new order.
        
        Returns:
            Order ID
        """
        import uuid
        order_id = str(uuid.uuid4())
        
        # Create event
        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ORDER_CREATED,
            aggregate_id=order_id,
            aggregate_type="Order",
            payload={},
            metadata={"source": "api"},
            timestamp=datetime.now(),
            version=1,
            user_id=user_id
        )
        
        # Persist event
        await self.event_store.append(event)
        
        return order_id

class AddItemCommandHandler:
    """Handles adding item to order."""
    
    def __init__(self, event_store: EventStore):
        self.event_store = event_store
    
    async def handle(
        self,
        order_id: str,
        product_id: int,
        quantity: int,
        price: float
    ) -> None:
        """Add item to order."""
        # Rebuild current state
        order = await self.event_store.rebuild_aggregate(order_id)
        
        # Validate business rules
        if order.status != "created":
            raise ValueError("Cannot add items to non-draft order")
        
        # Create event
        import uuid
        event = DomainEvent(
            event_id=str(uuid.uuid4()),
            event_type=EventType.ITEM_ADDED,
            aggregate_id=order_id,
            aggregate_type="Order",
            payload={
                "item": {
                    "product_id": product_id,
                    "quantity": quantity,
                    "price": price
                }
            },
            metadata={},
            timestamp=datetime.now(),
            version=order.version + 1
        )
        
        await self.event_store.append(event)

# ================== READ MODEL PROJECTIONS ==================

class OrderReadModel:
    """
    Denormalized read model for fast queries.
    
    Updated asynchronously from events.
    """
    
    def __init__(self, read_database):
        self.read_db = read_database
    
    async def handle_order_created(self, event: DomainEvent) -> None:
        """Project order created event to read model."""
        await self.read_db.insert_order({
            "order_id": event.aggregate_id,
            "status": "created",
            "items": [],
            "total_amount": 0.0,
            "created_at": event.timestamp,
            "user_id": event.user_id
        })
    
    async def handle_item_added(self, event: DomainEvent) -> None:
        """Project item added event to read model."""
        item = event.payload["item"]
        
        # Update denormalized order in read database
        await self.read_db.add_item_to_order(
            order_id=event.aggregate_id,
            item=item
        )
        
        # Update order total
        await self.read_db.update_order_total(event.aggregate_id)
    
    async def handle_event(self, event: DomainEvent) -> None:
        """Route event to appropriate handler."""
        handlers = {
            EventType.ORDER_CREATED: self.handle_order_created,
            EventType.ITEM_ADDED: self.handle_item_added,
            # ... more handlers
        }
        
        handler = handlers.get(event.event_type)
        if handler:
            await handler(event)

# Subscribe read model to events
read_model = OrderReadModel(read_database)
event_store.subscribe(read_model.handle_event)

# ================== QUERY HANDLERS (READ SIDE) ==================

class GetOrderQueryHandler:
    """Handles order retrieval from read model."""
    
    def __init__(self, read_database, cache):
        self.read_db = read_database
        self.cache = cache
    
    async def handle(self, order_id: str) -> Optional[Dict]:
        """
        Get order from optimized read model.
        
        Fast because:
        - Denormalized data
        - Cached
        - Optimized indexes
        """
        # Check cache
        cache_key = f"order:{order_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            return json.loads(cached)
        
        # Query read database
        order = await self.read_db.get_order(order_id)
        
        if order:
            await self.cache.set(cache_key, json.dumps(order), ttl=300)
        
        return order

# ================== FASTAPI INTEGRATION ==================

app = FastAPI(title="Event Sourcing + CQRS Example")

# Commands (writes)
@app.post("/orders", status_code=201)
async def create_order(user_id: int):
    """Create order - command side."""
    handler = CreateOrderCommandHandler(event_store)
    order_id = await handler.handle(user_id)
    return {"order_id": order_id}

@app.post("/orders/{order_id}/items")
async def add_item(
    order_id: str,
    product_id: int,
    quantity: int,
    price: float
):
    """Add item to order - command side."""
    handler = AddItemCommandHandler(event_store)
    await handler.handle(order_id, product_id, quantity, price)
    return {"message": "Item added"}

# Queries (reads)
@app.get("/orders/{order_id}")
async def get_order(order_id: str):
    """Get order - query side (read model)."""
    handler = GetOrderQueryHandler(read_database, cache)
    order = await handler.handle(order_id)
    
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    
    return order

# Event replay for read model rebuild
@app.post("/admin/rebuild-read-model")
async def rebuild_read_model():
    """
    Rebuild read model from events.
    
    Useful for:
    - Fixing data inconsistencies
    - Adding new projections
    - Recovery from failures
    """
    # Clear read database
    await read_database.truncate_all()
    
    # Replay all events
    all_events = await event_store.get_all_events()
    
    for event in all_events:
        await read_model.handle_event(event)
    
    return {"message": f"Rebuilt read model from {len(all_events)} events"}
```

**Event Sourcing Benefits:**
- **Complete Audit Trail**: Every change is recorded
- **Time Travel**: Reconstruct state at any point
- **Event Replay**: Rebuild read models anytime
- **Business Intelligence**: Analyze historical data
- **Debugging**: Understand exactly what happened

**Challenges:**
- **Complexity**: More complex than CRUD
- **Eventual Consistency**: Read model may lag
- **Event Schema Evolution**: Handling event versioning
- **Storage**: Events accumulate over time
- **Query Performance**: Need optimized read models

---

## Database Architecture

### Q7: Design a database sharding strategy for a Python application

**Expected Answer:**

```python
from typing import Optional, List, Dict, Any
import hashlib
from enum import Enum

class ShardingStrategy(Enum):
    """Different sharding strategies."""
    HASH_BASED = "hash_based"
    RANGE_BASED = "range_based"
    GEOGRAPHIC = "geographic"
    DIRECTORY_BASED = "directory_based"

class DatabaseShardManager:
    """
    Manages database sharding across multiple database instances.
    
    Features:
    - Multiple sharding strategies
    - Shard rebalancing
    - Cross-shard queries
    - Consistent hashing for minimal data movement
    """
    
    def __init__(self, shard_count: int = 4):
        self.shard_count = shard_count
        self.shards: Dict[int, Dict] = {}
        
        # Initialize shard connections
        for i in range(shard_count):
            self.shards[i] = {
                "id": i,
                "connection_string": f"postgresql://localhost/db_shard_{i}",
                "connection": None,  # Initialize actual connection
                "range": None  # For range-based sharding
            }
    
    def get_shard_by_hash(self, shard_key: str) -> int:
        """
        Determine shard using consistent hashing.
        
        Args:
            shard_key: Key to hash (e.g., user_id, tenant_id)
            
        Returns:
            Shard ID
        """
        # Consistent hashing
        hash_value = int(hashlib.md5(shard_key.encode()).hexdigest(), 16)
        shard_id = hash_value % self.shard_count
        return shard_id
    
    def get_shard_by_range(self, shard_key: int) -> int:
        """
        Determine shard using range-based partitioning.
        
        Example:
        - Shard 0: IDs 1-1000000
        - Shard 1: IDs 1000001-2000000
        - etc.
        """
        range_size = 1000000
        shard_id = (shard_key - 1) // range_size
        return min(shard_id, self.shard_count - 1)
    
    def get_shard_connection(self, shard_id: int):
        """Get database connection for shard."""
        shard = self.shards[shard_id]
        
        if not shard["connection"]:
            # Initialize connection (simplified)
            shard["connection"] = f"Connection to shard {shard_id}"
        
        return shard["connection"]
    
    async def execute_on_shard(
        self,
        shard_id: int,
        query: str,
        *args
    ) -> Any:
        """Execute query on specific shard."""
        connection = self.get_shard_connection(shard_id)
        # Execute query
        # result = await connection.fetch(query, *args)
        return None
    
    async def execute_on_all_shards(
        self,
        query: str,
        *args
    ) -> List[Any]:
        """
        Execute query on all shards (scatter-gather).
        
        Use for: Global queries, analytics, aggregations
        """
        tasks = [
            self.execute_on_shard(shard_id, query, *args)
            for shard_id in range(self.shard_count)
        ]
        
        results = await asyncio.gather(*tasks)
        return results

# Sharded Repository
class ShardedUserRepository:
    """Repository that uses database sharding."""
    
    def __init__(self, shard_manager: DatabaseShardManager):
        self.shard_manager = shard_manager
    
    def _get_user_shard(self, user_id: int) -> int:
        """Determine which shard contains user."""
        return self.shard_manager.get_shard_by_hash(str(user_id))
    
    async def find_by_id(self, user_id: int) -> Optional[Dict]:
        """Find user in appropriate shard."""
        shard_id = self._get_user_shard(user_id)
        
        query = "SELECT * FROM users WHERE id = $1"
        result = await self.shard_manager.execute_on_shard(
            shard_id,
            query,
            user_id
        )
        
        return result
    
    async def create(self, user_data: Dict) -> int:
        """Create user in appropriate shard."""
        user_id = user_data["id"]
        shard_id = self._get_user_shard(user_id)
        
        query = """
            INSERT INTO users (id, username, email) 
            VALUES ($1, $2, $3)
            RETURNING id
        """
        
        result = await self.shard_manager.execute_on_shard(
            shard_id,
            query,
            user_id,
            user_data["username"],
            user_data["email"]
        )
        
        return result
    
    async def count_all_users(self) -> int:
        """
        Count users across all shards.
        
        Scatter-gather query.
        """
        query = "SELECT COUNT(*) as count FROM users"
        results = await self.shard_manager.execute_on_all_shards(query)
        
        # Aggregate results from all shards
        total = sum(r[0]["count"] for r in results if r)
        return total
    
    async def find_by_email(self, email: str) -> Optional[Dict]:
        """
        Find user by email (global search).
        
        Problem: Email is not the shard key!
        Solutions:
        1. Secondary index in global directory
        2. Search all shards (slow)
        3. Denormalize email -> user_id mapping
        """
        # Solution 1: Use global directory
        user_id = await self._lookup_email_directory(email)
        if user_id:
            return await self.find_by_id(user_id)
        
        # Solution 2: Search all shards (fallback)
        query = "SELECT * FROM users WHERE email = $1"
        results = await self.shard_manager.execute_on_all_shards(query, email)
        
        for result in results:
            if result:
                return result
        
        return None
    
    async def _lookup_email_directory(self, email: str) -> Optional[int]:
        """Look up user ID from global email directory."""
        # Global directory (separate database or cache)
        # Maps email -> user_id for cross-shard lookups
        return None

# Shard rebalancing
class ShardRebalancer:
    """
    Handles shard rebalancing when adding/removing shards.
    
    Process:
    1. Add new shard
    2. Calculate new shard assignments
    3. Move data to new shards
    4. Update routing configuration
    5. Remove old shards
    """
    
    def __init__(self, shard_manager: DatabaseShardManager):
        self.shard_manager = shard_manager
    
    async def add_shard(self, new_shard_config: Dict) -> None:
        """Add new shard and rebalance."""
        new_shard_id = len(self.shard_manager.shards)
        self.shard_manager.shards[new_shard_id] = new_shard_config
        self.shard_manager.shard_count += 1
        
        # Rebalance data
        await self.rebalance_data()
    
    async def rebalance_data(self) -> None:
        """Rebalance data across shards."""
        # 1. Identify data to move
        # 2. Copy data to new shards
        # 3. Verify data integrity
        # 4. Update routing
        # 5. Delete from old shards
        pass
```

**Sharding Considerations:**
- **Shard Key Selection**: Choose wisely (can't easily change)
- **Cross-Shard Queries**: Expensive, avoid if possible
- **Data Distribution**: Ensure even distribution
- **Hotspots**: Monitor and redistribute if needed
- **Transactions**: Limited to single shard
- **Schema Changes**: Apply to all shards

---

## Microservices & APIs

### Q8: Design an API gateway with authentication, rate limiting, and circuit breaking

**See Q2 and Q3 for implementation details**

Key Features to Cover:
- Request routing
- Authentication/Authorization
- Rate limiting
- Circuit breaking
- Request/Response transformation
- Caching
- Logging and monitoring
- API versioning

---

## Performance & Optimization

### Q9: Design a high-performance data processing pipeline

**Expected Answer:**

```python
import asyncio
from typing import AsyncIterator, Callable, Any, List
from dataclasses import dataclass
import aiofiles
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# ================== ASYNC PIPELINE ==================

class AsyncPipeline:
    """
    High-performance async data processing pipeline.
    
    Features:
    - Async I/O for network/file operations
    - Concurrent processing
    - Backpressure handling
    - Error recovery
    """
    
    def __init__(self, max_concurrent: int = 100):
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_stream(
        self,
        data_source: AsyncIterator,
        processors: List[Callable],
        batch_size: int = 100
    ) -> AsyncIterator:
        """
        Process data stream through pipeline stages.
        
        Args:
            data_source: Async iterator of input data
            processors: List of processing functions
            batch_size: Number of items to process concurrently
            
        Yields:
            Processed data items
        """
        batch = []
        
        async for item in data_source:
            batch.append(item)
            
            if len(batch) >= batch_size:
                # Process batch concurrently
                results = await self._process_batch(batch, processors)
                for result in results:
                    if result is not None:
                        yield result
                
                batch = []
        
        # Process remaining items
        if batch:
            results = await self._process_batch(batch, processors)
            for result in results:
                if result is not None:
                    yield result
    
    async def _process_batch(
        self,
        batch: List[Any],
        processors: List[Callable]
    ) -> List[Any]:
        """Process batch of items through all processors."""
        async def process_item(item):
            async with self.semaphore:
                result = item
                
                for processor in processors:
                    try:
                        if asyncio.iscoroutinefunction(processor):
                            result = await processor(result)
                        else:
                            result = processor(result)
                    except Exception as e:
                        print(f"Error processing item: {e}")
                        return None
                
                return result
        
        results = await asyncio.gather(*[process_item(item) for item in batch])
        return results

# ================== CPU-BOUND PROCESSING ==================

class ParallelProcessor:
    """
    Parallel processing for CPU-intensive tasks.
    
    Uses multiprocessing to bypass GIL.
    """
    
    def __init__(self, num_workers: int = None):
        self.num_workers = num_workers or mp.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
    
    async def process_batch(
        self,
        items: List[Any],
        processor_func: Callable
    ) -> List[Any]:
        """
        Process batch using multiprocessing.
        
        Args:
            items: Items to process
            processor_func: CPU-intensive processing function
            
        Returns:
            Processed results
        """
        loop = asyncio.get_event_loop()
        
        # Run in process pool
        results = await loop.run_in_executor(
            self.executor,
            lambda: list(map(processor_func, items))
        )
        
        return results
    
    def shutdown(self):
        """Shutdown process pool."""
        self.executor.shutdown(wait=True)

# ================== EXAMPLE USAGE ==================

# Data source (async file reading)
async def read_large_file(filename: str) -> AsyncIterator[str]:
    """Read large file line by line asynchronously."""
    async with aiofiles.open(filename, 'r') as f:
        async for line in f:
            yield line.strip()

# Processing stages
async def parse_json(line: str) -> Optional[Dict]:
    """Parse JSON line."""
    try:
        import json
        return json.loads(line)
    except:
        return None

async def enrich_data(item: Dict) -> Dict:
    """Enrich data with external API call."""
    # Simulate API call
    await asyncio.sleep(0.01)
    item["enriched"] = True
    return item

def transform_data(item: Dict) -> Dict:
    """CPU-intensive transformation."""
    # Expensive computation
    item["transformed"] = item.get("value", 0) ** 2
    return item

async def save_to_database(item: Dict) -> Dict:
    """Save to database."""
    # Simulate database write
    await asyncio.sleep(0.005)
    return item

# Run pipeline
async def run_pipeline():
    """Execute complete data pipeline."""
    pipeline = AsyncPipeline(max_concurrent=100)
    
    # Data source
    source = read_large_file("large_file.jsonl")
    
    # Processing stages
    processors = [
        parse_json,
        enrich_data,
        transform_data,
        save_to_database
    ]
    
    # Process stream
    processed_count = 0
    async for result in pipeline.process_stream(source, processors, batch_size=100):
        processed_count += 1
        
        if processed_count % 1000 == 0:
            print(f"Processed {processed_count} records")
    
    print(f"Total processed: {processed_count}")

# asyncio.run(run_pipeline())
```

**Performance Optimizations:**
1. **Async I/O**: For network/file operations
2. **Multiprocessing**: For CPU-intensive work
3. **Batching**: Reduce overhead
4. **Connection Pooling**: Reuse connections
5. **Caching**: Reduce redundant computations
6. **Streaming**: Process data incrementally
7. **Backpressure**: Prevent memory overflow

---

## Leadership & Process Questions

### Q10: How do you ensure code quality across a large Python team?

**Expected Answer:**

**1. Code Standards:**
```python
# pyproject.toml
[tool.black]
line-length = 100
target-version = ['py311']

[tool.isort]
profile = "black"
line_length = 100

[tool.mypy]
python_version = "3.11"
strict = true
warn_return_any = true
warn_unused_configs = true

[tool.pylint]
max-line-length = 100
disable = ["C0111", "C0103"]

[tool.pytest.ini_options]
minversion = "7.0"
testpaths = ["tests"]
python_files = ["test_*.py"]
```

**2. Pre-commit Hooks:**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/PyCQA/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
```

**3. CI/CD Pipeline:**
```yaml
# .github/workflows/ci.yml
name: CI/CD

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install dependencies
        run: pip install -r requirements.txt
      
      - name: Lint
        run: |
          flake8 src/
          black --check src/
          mypy src/
      
      - name: Test
        run: pytest --cov=src --cov-report=xml
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

**4. Code Review Process:**
- Mandatory reviews for all PRs
- Automated checks must pass
- Architecture review for major changes
- Security review for sensitive code

**5. Documentation:**
- ADR (Architecture Decision Records)
- API documentation (OpenAPI/Swagger)
- Code comments and docstrings
- README for each service

---

## 📊 Interview Evaluation Criteria

### For Architect Position:

**Technical Skills (40%):**
- System design capabilities
- Architectural patterns knowledge
- Performance optimization
- Security awareness

**Leadership (30%):**
- Team mentoring
- Technical decision-making
- Stakeholder communication
- Conflict resolution

**Experience (20%):**
- Large-scale system design
- Production incidents handling
- Migration projects
- Technology evaluation

**Soft Skills (10%):**
- Communication clarity
- Documentation quality
- Collaboration ability
- Continuous learning

---

## 🎯 Key Topics to Master

1. **Architectural Patterns**
   - Clean Architecture
   - Hexagonal Architecture
   - CQRS + Event Sourcing
   - Microservices
   - Event-Driven Architecture

2. **Scalability**
   - Horizontal vs Vertical scaling
   - Database sharding
   - Caching strategies
   - Load balancing
   - CDN usage

3. **Performance**
   - Profiling and optimization
   - Async programming
   - Connection pooling
   - Query optimization
   - Memory management

4. **Security**
   - Authentication/Authorization
   - Encryption (at rest & in transit)
   - Input validation
   - Security headers
   - OWASP Top 10

5. **DevOps**
   - CI/CD pipelines
   - Container orchestration (K8s)
   - Infrastructure as Code
   - Monitoring and alerting
   - Incident management

---

## 📚 Resources

- **Clean Architecture**: https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html
- **Microservices Patterns**: https://microservices.io/
- **FastAPI Best Practices**: https://github.com/zhanymkanov/fastapi-best-practices
- **System Design Interview**: https://github.com/donnemartin/system-design-primer
- **Python Performance**: https://github.com/python/performance

---

**Prepare thoroughly and showcase your architectural thinking!** 🏗️🚀

