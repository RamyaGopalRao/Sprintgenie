# 🎯 Tarento Technologies - Python Architect Interview Questions
## Bhashini 2.0 Platform Architecture

### Position: Python Architect
### Focus: API Marketplace, SEM Modules, Partner Portal, Distributed Systems

---

## 📋 Interview Structure

**Round 1**: System Design & Architecture (90 mins)
**Round 2**: Python Technical Deep Dive (60 mins)
**Round 3**: Distributed Systems & Scalability (60 mins)
**Round 4**: Leadership & Stakeholder Management (45 mins)

---

## 🏗️ Architecture & Design (Critical for Bhashini 2.0)

### Q1: Design the API Marketplace Backend for Bhashini 2.0

**Scenario:**
Bhashini 2.0 needs an API marketplace where:
- Multiple AI/ML service providers can list their APIs
- Users can discover, subscribe, and consume APIs
- Real-time usage tracking and billing
- Multi-tenancy for different providers
- SLA monitoring and enforcement
- Partner onboarding and management

**Expected Answer:**

```python
"""
API Marketplace Architecture for Bhashini 2.0

SYSTEM COMPONENTS:

1. API Gateway Layer
   - Request routing to service providers
   - Authentication & Authorization
   - Rate limiting & quotas
   - Request/Response transformation
   - Analytics collection

2. Marketplace Services
   - API Catalog Service
   - Subscription Management Service
   - Usage Metering Service
   - Billing Service
   - Partner Management Service

3. Data Layer
   - PostgreSQL: Transactional data
   - MongoDB: API metadata & documentation
   - Redis: Caching & rate limiting
   - ClickHouse: Analytics & usage tracking

4. Integration Layer
   - Service Mesh (Istio/Linkerd)
   - Message Queue (Kafka/RabbitMQ)
   - Event Store for audit trail
"""

from fastapi import FastAPI, Depends, HTTPException, Header
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import asyncio

# ================== MODELS ==================

class APICategory(str, Enum):
    """Categories of APIs in marketplace."""
    TRANSLATION = "translation"
    SPEECH_TO_TEXT = "speech_to_text"
    TEXT_TO_SPEECH = "text_to_speech"
    OCR = "ocr"
    NLP = "nlp"
    CUSTOM = "custom"

class SubscriptionTier(str, Enum):
    """Subscription tiers for API access."""
    FREE = "free"
    BASIC = "basic"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"

class APIMetadata(BaseModel):
    """API metadata for marketplace listing."""
    api_id: str
    name: str
    description: str
    category: APICategory
    provider_id: str
    version: str
    endpoint_url: str
    
    # Pricing
    pricing_model: str  # "pay_per_use", "subscription", "hybrid"
    price_per_request: Optional[float] = None
    subscription_price: Optional[Dict[SubscriptionTier, float]] = None
    
    # Technical specs
    max_requests_per_second: int = 100
    average_latency_ms: int = 200
    uptime_sla: float = 99.9
    
    # Documentation
    openapi_spec_url: str
    documentation_url: str
    example_requests: List[Dict] = Field(default_factory=list)
    
    # Status
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class APISubscription(BaseModel):
    """User subscription to an API."""
    subscription_id: str
    user_id: int
    api_id: str
    tier: SubscriptionTier
    quota_limit: int  # Requests per month
    quota_used: int = 0
    start_date: datetime
    end_date: Optional[datetime] = None
    is_active: bool = True

# ================== API GATEWAY ==================

class APIMarketplaceGateway:
    """
    Central gateway for API marketplace.
    
    Responsibilities:
    - Route requests to provider APIs
    - Enforce quotas and rate limits
    - Track usage for billing
    - Handle authentication
    - Collect analytics
    """
    
    def __init__(
        self,
        subscription_service,
        usage_tracker,
        rate_limiter,
        provider_registry
    ):
        self.subscription_service = subscription_service
        self.usage_tracker = usage_tracker
        self.rate_limiter = rate_limiter
        self.provider_registry = provider_registry
    
    async def route_request(
        self,
        api_id: str,
        user_id: int,
        request_data: Dict[str, Any],
        headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Route API request through marketplace.
        
        Flow:
        1. Validate subscription
        2. Check quota
        3. Enforce rate limit
        4. Route to provider
        5. Track usage
        6. Return response
        """
        # 1. Validate subscription
        subscription = await self.subscription_service.get_active_subscription(
            user_id=user_id,
            api_id=api_id
        )
        
        if not subscription:
            raise HTTPException(
                status_code=403,
                detail="No active subscription for this API"
            )
        
        # 2. Check quota
        if subscription.quota_used >= subscription.quota_limit:
            raise HTTPException(
                status_code=429,
                detail="Quota exceeded. Please upgrade your plan."
            )
        
        # 3. Rate limiting
        is_allowed = await self.rate_limiter.check_limit(
            user_id=user_id,
            api_id=api_id,
            tier=subscription.tier
        )
        
        if not is_allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded"
            )
        
        # 4. Get provider endpoint
        provider = await self.provider_registry.get_provider(api_id)
        if not provider:
            raise HTTPException(
                status_code=503,
                detail="API provider unavailable"
            )
        
        # 5. Route request to provider with circuit breaker
        try:
            response = await self._call_provider_with_retry(
                provider_url=provider.endpoint_url,
                request_data=request_data,
                headers=headers,
                timeout=provider.timeout
            )
            
            # 6. Track usage
            await self.usage_tracker.record_usage(
                subscription_id=subscription.subscription_id,
                api_id=api_id,
                user_id=user_id,
                response_time_ms=response.get("latency_ms", 0),
                status_code=200,
                timestamp=datetime.now()
            )
            
            # Update quota
            await self.subscription_service.increment_usage(
                subscription.subscription_id
            )
            
            return response
            
        except Exception as e:
            # Track failed request
            await self.usage_tracker.record_usage(
                subscription_id=subscription.subscription_id,
                api_id=api_id,
                user_id=user_id,
                status_code=500,
                error=str(e),
                timestamp=datetime.now()
            )
            raise
    
    async def _call_provider_with_retry(
        self,
        provider_url: str,
        request_data: Dict,
        headers: Dict,
        timeout: int,
        max_retries: int = 3
    ) -> Dict:
        """
        Call provider API with retry logic and circuit breaker.
        """
        import httpx
        from tenacity import retry, stop_after_attempt, wait_exponential
        
        @retry(
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=10)
        )
        async def make_request():
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    provider_url,
                    json=request_data,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
        
        return await make_request()

# ================== SUBSCRIPTION SERVICE ==================

class SubscriptionService:
    """
    Manages API subscriptions and quotas.
    
    Multi-tenant service supporting different subscription tiers.
    """
    
    def __init__(self, repository, cache, event_bus):
        self.repository = repository
        self.cache = cache
        self.event_bus = event_bus
    
    async def create_subscription(
        self,
        user_id: int,
        api_id: str,
        tier: SubscriptionTier
    ) -> APISubscription:
        """
        Create new API subscription.
        
        Workflow:
        1. Validate user and API
        2. Check for existing subscription
        3. Calculate quota based on tier
        4. Create subscription
        5. Publish event
        """
        # Check existing subscription
        existing = await self.repository.find_subscription(
            user_id=user_id,
            api_id=api_id,
            is_active=True
        )
        
        if existing:
            raise ValueError("Active subscription already exists")
        
        # Get quota for tier
        quota_mapping = {
            SubscriptionTier.FREE: 1000,
            SubscriptionTier.BASIC: 10000,
            SubscriptionTier.PROFESSIONAL: 100000,
            SubscriptionTier.ENTERPRISE: -1  # Unlimited
        }
        
        import uuid
        subscription = APISubscription(
            subscription_id=str(uuid.uuid4()),
            user_id=user_id,
            api_id=api_id,
            tier=tier,
            quota_limit=quota_mapping[tier],
            start_date=datetime.now()
        )
        
        # Save subscription
        await self.repository.create_subscription(subscription)
        
        # Publish event
        await self.event_bus.publish({
            "event_type": "subscription.created",
            "subscription_id": subscription.subscription_id,
            "user_id": user_id,
            "api_id": api_id,
            "tier": tier.value
        })
        
        return subscription
    
    async def get_active_subscription(
        self,
        user_id: int,
        api_id: str
    ) -> Optional[APISubscription]:
        """Get active subscription with caching."""
        cache_key = f"subscription:{user_id}:{api_id}"
        
        # Check cache
        cached = await self.cache.get(cache_key)
        if cached:
            import json
            return APISubscription(**json.loads(cached))
        
        # Fetch from database
        subscription = await self.repository.find_subscription(
            user_id=user_id,
            api_id=api_id,
            is_active=True
        )
        
        if subscription:
            # Cache for 5 minutes
            import json
            await self.cache.set(
                cache_key,
                json.dumps(subscription.dict()),
                expire=300
            )
        
        return subscription
    
    async def increment_usage(self, subscription_id: str) -> None:
        """
        Increment usage count atomically.
        
        Uses Redis for real-time quota tracking.
        """
        # Increment in Redis (atomic operation)
        redis_key = f"quota:used:{subscription_id}"
        await self.cache.redis.incr(redis_key)
        
        # Sync to database periodically (every 100 requests)
        current_count = await self.cache.redis.get(redis_key)
        if int(current_count) % 100 == 0:
            await self.repository.update_quota_used(
                subscription_id,
                int(current_count)
            )

# ================== USAGE TRACKING & ANALYTICS ==================

class UsageTracker:
    """
    Tracks API usage for billing and analytics.
    
    High-throughput service using ClickHouse for analytics.
    """
    
    def __init__(self, clickhouse_client, kafka_producer):
        self.clickhouse = clickhouse_client
        self.kafka = kafka_producer
    
    async def record_usage(
        self,
        subscription_id: str,
        api_id: str,
        user_id: int,
        response_time_ms: int = 0,
        status_code: int = 200,
        error: Optional[str] = None,
        timestamp: datetime = None
    ) -> None:
        """
        Record API usage event.
        
        Uses Kafka for high-throughput ingestion.
        """
        usage_event = {
            "subscription_id": subscription_id,
            "api_id": api_id,
            "user_id": user_id,
            "response_time_ms": response_time_ms,
            "status_code": status_code,
            "error": error,
            "timestamp": (timestamp or datetime.now()).isoformat()
        }
        
        # Send to Kafka for async processing
        await self.kafka.send(
            topic="api_usage_events",
            value=usage_event
        )
    
    async def get_usage_analytics(
        self,
        api_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Get usage analytics from ClickHouse.
        
        Optimized for analytical queries.
        """
        query = """
            SELECT 
                toStartOfHour(timestamp) as hour,
                COUNT(*) as request_count,
                AVG(response_time_ms) as avg_response_time,
                quantile(0.95)(response_time_ms) as p95_response_time,
                SUM(CASE WHEN status_code >= 500 THEN 1 ELSE 0 END) as error_count
            FROM api_usage
            WHERE api_id = {api_id:String}
              AND timestamp >= {start_date:DateTime}
              AND timestamp <= {end_date:DateTime}
            GROUP BY hour
            ORDER BY hour
        """
        
        results = await self.clickhouse.query(
            query,
            api_id=api_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "api_id": api_id,
            "period": {"start": start_date, "end": end_date},
            "metrics": results
        }

# ================== PARTNER PORTAL SERVICE ==================

class PartnerPortalService:
    """
    Service for partner onboarding and management.
    
    Features:
    - Partner registration and approval workflow
    - API registration and versioning
    - Revenue sharing and settlements
    - Performance monitoring dashboard
    """
    
    def __init__(
        self,
        partner_repository,
        api_catalog_service,
        revenue_service,
        notification_service
    ):
        self.partner_repo = partner_repository
        self.api_catalog = api_catalog_service
        self.revenue = revenue_service
        self.notifications = notification_service
    
    async def register_partner(
        self,
        partner_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Register new partner with approval workflow.
        
        Workflow:
        1. Validate partner information
        2. Create pending partner record
        3. Trigger approval workflow
        4. Send notification
        """
        # Validate
        required_fields = ["company_name", "email", "contact_person", "business_type"]
        for field in required_fields:
            if field not in partner_data:
                raise ValueError(f"Missing required field: {field}")
        
        # Create partner with pending status
        import uuid
        partner_id = str(uuid.uuid4())
        
        partner = {
            "partner_id": partner_id,
            **partner_data,
            "status": "pending_approval",
            "created_at": datetime.now(),
            "api_count": 0,
            "revenue_share_percentage": 70.0  # Default 70% to partner
        }
        
        await self.partner_repo.create(partner)
        
        # Trigger approval workflow (Airflow DAG or workflow engine)
        await self.trigger_approval_workflow(partner_id)
        
        # Send notification
        await self.notifications.send_email(
            to=partner_data["email"],
            template="partner_registration_received",
            data={"partner_id": partner_id}
        )
        
        return partner
    
    async def register_api(
        self,
        partner_id: str,
        api_metadata: APIMetadata
    ) -> str:
        """
        Register API in marketplace catalog.
        
        Workflow:
        1. Validate partner is approved
        2. Validate API metadata
        3. Test API endpoint
        4. Register in catalog
        5. Generate API key for provider
        """
        # Validate partner
        partner = await self.partner_repo.find_by_id(partner_id)
        if not partner or partner["status"] != "approved":
            raise ValueError("Partner not approved")
        
        # Test API endpoint
        is_healthy = await self._test_api_endpoint(api_metadata.endpoint_url)
        if not is_healthy:
            raise ValueError("API endpoint is not responding")
        
        # Register in catalog
        api_id = await self.api_catalog.register_api(
            partner_id=partner_id,
            metadata=api_metadata
        )
        
        # Generate API key for provider authentication
        api_key = await self._generate_api_key(partner_id, api_id)
        
        # Update partner API count
        await self.partner_repo.increment_api_count(partner_id)
        
        return api_id
    
    async def _test_api_endpoint(self, endpoint_url: str) -> bool:
        """Test if API endpoint is healthy."""
        import httpx
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{endpoint_url}/health", timeout=5)
                return response.status_code == 200
        except:
            return False
    
    async def _generate_api_key(self, partner_id: str, api_id: str) -> str:
        """Generate secure API key for provider."""
        import secrets
        api_key = f"bhashini_{secrets.token_urlsafe(32)}"
        
        # Store in secure vault
        await self.store_api_key(partner_id, api_id, api_key)
        
        return api_key
    
    async def get_partner_analytics(
        self,
        partner_id: str,
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """
        Get partner performance analytics.
        
        Metrics:
        - Total requests
        - Success rate
        - Average latency
        - Revenue generated
        - Top consuming users
        """
        # Get all APIs for partner
        apis = await self.api_catalog.get_partner_apis(partner_id)
        api_ids = [api["api_id"] for api in apis]
        
        # Aggregate analytics across all APIs
        analytics = await self.usage_tracker.get_aggregate_analytics(
            api_ids=api_ids,
            start_date=start_date,
            end_date=end_date
        )
        
        # Calculate revenue
        revenue = await self.revenue.calculate_partner_revenue(
            partner_id=partner_id,
            start_date=start_date,
            end_date=end_date
        )
        
        return {
            "partner_id": partner_id,
            "period": {"start": start_date, "end": end_date},
            "total_requests": analytics["total_requests"],
            "success_rate": analytics["success_rate"],
            "avg_latency_ms": analytics["avg_latency_ms"],
            "revenue": revenue,
            "top_apis": analytics["top_apis"]
        }

# ================== RATE LIMITING (MULTI-TIER) ==================

class TieredRateLimiter:
    """
    Rate limiter with different limits per subscription tier.
    
    Uses Redis with sliding window algorithm.
    """
    
    def __init__(self, redis_client):
        self.redis = redis_client
        
        # Rate limits per tier (requests per minute)
        self.tier_limits = {
            SubscriptionTier.FREE: 10,
            SubscriptionTier.BASIC: 60,
            SubscriptionTier.PROFESSIONAL: 300,
            SubscriptionTier.ENTERPRISE: 1000
        }
    
    async def check_limit(
        self,
        user_id: int,
        api_id: str,
        tier: SubscriptionTier
    ) -> bool:
        """
        Check if request is within rate limit.
        
        Uses sliding window counter in Redis.
        """
        import time
        
        key = f"rate_limit:{user_id}:{api_id}"
        current_time = time.time()
        window_size = 60  # 1 minute
        
        # Use Redis sorted set for sliding window
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, current_time - window_size)
        
        # Count requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, window_size)
        
        results = await pipe.execute()
        request_count = results[1]
        
        # Check against tier limit
        limit = self.tier_limits[tier]
        return request_count < limit

# ================== SLA MONITORING ==================

class SLAMonitor:
    """
    Monitors and enforces SLA for APIs.
    
    Tracks:
    - Uptime
    - Response time (p50, p95, p99)
    - Error rate
    - Availability
    """
    
    def __init__(self, metrics_db, alert_service):
        self.metrics_db = metrics_db
        self.alert_service = alert_service
    
    async def check_sla_compliance(
        self,
        api_id: str,
        time_window: int = 3600
    ) -> Dict[str, Any]:
        """
        Check if API is meeting SLA requirements.
        
        Returns SLA compliance report.
        """
        # Get API SLA requirements
        api_sla = await self.get_api_sla(api_id)
        
        # Calculate actual metrics
        metrics = await self.metrics_db.get_metrics(
            api_id=api_id,
            window_seconds=time_window
        )
        
        # Calculate uptime
        total_requests = metrics["total_requests"]
        successful_requests = metrics["successful_requests"]
        uptime_percentage = (successful_requests / total_requests * 100) if total_requests > 0 else 100
        
        # Check SLA breach
        sla_breached = uptime_percentage < api_sla["uptime_sla"]
        
        if sla_breached:
            await self.handle_sla_breach(api_id, uptime_percentage)
        
        return {
            "api_id": api_id,
            "sla_target": api_sla["uptime_sla"],
            "actual_uptime": uptime_percentage,
            "p95_latency_ms": metrics["p95_latency"],
            "error_rate": metrics["error_rate"],
            "compliant": not sla_breached
        }
    
    async def handle_sla_breach(self, api_id: str, actual_uptime: float):
        """Handle SLA breach - alert and take action."""
        # Send alert to partner
        await self.alert_service.send_alert(
            severity="high",
            message=f"SLA breach detected for API {api_id}",
            details={"uptime": actual_uptime}
        )
        
        # Auto-disable API if severely degraded
        if actual_uptime < 90:
            await self.auto_disable_api(api_id)
    
    async def auto_disable_api(self, api_id: str):
        """Automatically disable severely degraded API."""
        # Implement circuit breaker at platform level
        pass

# ================== FASTAPI ROUTES ==================

app = FastAPI(
    title="Bhashini 2.0 API Marketplace",
    description="Discover and consume AI/ML APIs",
    version="2.0"
)

# Marketplace endpoints
@app.get("/marketplace/apis", response_model=List[APIMetadata])
async def list_apis(
    category: Optional[APICategory] = None,
    search: Optional[str] = None
):
    """
    List all APIs in marketplace.
    
    Supports:
    - Filtering by category
    - Search by name/description
    - Pagination
    """
    pass

@app.post("/marketplace/subscribe")
async def subscribe_to_api(
    api_id: str,
    tier: SubscriptionTier,
    user_id: int
):
    """Subscribe to an API."""
    pass

@app.post("/marketplace/api/{api_id}/invoke")
async def invoke_api(
    api_id: str,
    request_data: Dict,
    user_id: int
):
    """Invoke API through marketplace gateway."""
    pass

# Partner portal endpoints
@app.post("/partners/register")
async def register_partner(partner_data: Dict):
    """Register as API provider."""
    pass

@app.post("/partners/{partner_id}/apis")
async def register_partner_api(
    partner_id: str,
    api_metadata: APIMetadata
):
    """Register new API in marketplace."""
    pass

@app.get("/partners/{partner_id}/analytics")
async def get_partner_analytics(
    partner_id: str,
    start_date: datetime,
    end_date: datetime
):
    """Get partner performance analytics."""
    pass
```

**Key Architectural Decisions:**

1. **Multi-Tenancy Isolation**
   - Separate schemas per partner
   - Row-level security
   - API key scoping

2. **Scalability**
   - Horizontal scaling of gateway
   - Read replicas for catalog
   - Redis cluster for rate limiting
   - ClickHouse for analytics

3. **Reliability**
   - Circuit breakers per provider
   - Retry logic with exponential backoff
   - Health checks and auto-recovery
   - Graceful degradation

4. **Performance**
   - Caching at multiple levels
   - Async I/O throughout
   - Connection pooling
   - Batch processing for analytics

5. **Observability**
   - Distributed tracing (Jaeger)
   - Metrics (Prometheus)
   - Centralized logging (ELK)
   - Real-time dashboards

---

### Q2: Design the Service Execution Module (SEM) architecture

**Scenario:**
SEM needs to:
- Execute AI/ML models dynamically
- Handle multiple concurrent requests
- Support different model types (NLP, Translation, STT, TTS)
- Auto-scaling based on load
- Model versioning and A/B testing

**Expected Answer:**

```python
"""
Service Execution Module (SEM) Architecture

ARCHITECTURE:

1. REQUEST ORCHESTRATOR
   - Receives execution requests
   - Routes to appropriate model
   - Manages model lifecycle

2. MODEL REGISTRY
   - Catalog of available models
   - Version management
   - Metadata and capabilities

3. EXECUTION ENGINES
   - Python/FastAPI for serving
   - Model-specific containers
   - GPU resource management

4. SCALING LAYER
   - Kubernetes HPA (Horizontal Pod Autoscaler)
   - KEDA for event-driven scaling
   - Queue-based load balancing

5. MONITORING
   - Model performance metrics
   - Resource utilization
   - Request success rate
"""

from fastapi import FastAPI, BackgroundTasks
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import asyncio
from enum import Enum

class ModelType(str, Enum):
    """Types of AI/ML models."""
    TRANSLATION = "translation"
    STT = "speech_to_text"
    TTS = "text_to_speech"
    NER = "named_entity_recognition"
    SENTIMENT = "sentiment_analysis"

class ModelStatus(str, Enum):
    """Model lifecycle status."""
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    UNLOADING = "unloading"

class ModelMetadata(BaseModel):
    """Metadata for registered model."""
    model_id: str
    model_type: ModelType
    version: str
    provider: str
    
    # Capabilities
    source_language: Optional[str] = None
    target_language: Optional[str] = None
    max_input_length: int = 1000
    
    # Performance
    average_inference_time_ms: int = 500
    gpu_required: bool = False
    memory_required_mb: int = 2048
    
    # Configuration
    endpoint_url: str
    health_check_url: str
    timeout_seconds: int = 30

class ExecutionRequest(BaseModel):
    """Request to execute model."""
    model_id: str
    input_data: Dict[str, Any]
    options: Dict[str, Any] = Field(default_factory=dict)
    callback_url: Optional[str] = None
    priority: int = 5  # 1 (low) to 10 (high)

class ExecutionResponse(BaseModel):
    """Response from model execution."""
    request_id: str
    model_id: str
    output: Dict[str, Any]
    execution_time_ms: int
    status: str
    error: Optional[str] = None

# ================== MODEL REGISTRY ==================

class ModelRegistry:
    """
    Central registry for all AI/ML models.
    
    Features:
    - Model discovery
    - Version management
    - Capability matching
    - Health monitoring
    """
    
    def __init__(self, database, cache):
        self.db = database
        self.cache = cache
        self.models: Dict[str, ModelMetadata] = {}
        self.model_status: Dict[str, ModelStatus] = {}
    
    async def register_model(self, metadata: ModelMetadata) -> str:
        """
        Register new model in registry.
        
        Args:
            metadata: Model metadata
            
        Returns:
            Model ID
        """
        # Validate model is accessible
        if not await self._validate_model_endpoint(metadata.endpoint_url):
            raise ValueError("Model endpoint not accessible")
        
        # Store metadata
        await self.db.insert_model(metadata.dict())
        self.models[metadata.model_id] = metadata
        self.model_status[metadata.model_id] = ModelStatus.READY
        
        # Cache for fast lookup
        import json
        await self.cache.set(
            f"model:{metadata.model_id}",
            json.dumps(metadata.dict()),
            expire=3600
        )
        
        return metadata.model_id
    
    async def find_model(
        self,
        model_type: ModelType,
        source_lang: Optional[str] = None,
        target_lang: Optional[str] = None,
        version: Optional[str] = None
    ) -> Optional[ModelMetadata]:
        """
        Find suitable model based on requirements.
        
        Capability-based matching.
        """
        # Filter models by criteria
        candidates = [
            m for m in self.models.values()
            if m.model_type == model_type
            and (not source_lang or m.source_language == source_lang)
            and (not target_lang or m.target_language == target_lang)
            and (not version or m.version == version)
            and self.model_status[m.model_id] == ModelStatus.READY
        ]
        
        if not candidates:
            return None
        
        # Select best model (by performance, load, etc.)
        return await self._select_best_model(candidates)
    
    async def _select_best_model(
        self,
        candidates: List[ModelMetadata]
    ) -> ModelMetadata:
        """Select best model based on current load and performance."""
        # Check current load on each model
        loads = await asyncio.gather(*[
            self._get_model_load(m.model_id)
            for m in candidates
        ])
        
        # Select model with lowest load
        best_idx = loads.index(min(loads))
        return candidates[best_idx]
    
    async def _get_model_load(self, model_id: str) -> float:
        """Get current load on model (0.0 to 1.0)."""
        # Check active requests
        active = await self.cache.get(f"model:load:{model_id}") or "0"
        return float(active) / 100  # Normalize

# ================== EXECUTION ORCHESTRATOR ==================

class ModelExecutionOrchestrator:
    """
    Orchestrates model execution with queueing and load balancing.
    
    Features:
    - Priority-based queueing
    - Load balancing across model instances
    - Async execution with callbacks
    - Resource management
    """
    
    def __init__(
        self,
        model_registry: ModelRegistry,
        execution_queue,
        result_store
    ):
        self.registry = model_registry
        self.queue = execution_queue
        self.results = result_store
        self.active_executions: Dict[str, asyncio.Task] = {}
    
    async def execute(
        self,
        request: ExecutionRequest
    ) -> ExecutionResponse:
        """
        Execute model inference.
        
        Supports:
        - Sync execution (wait for result)
        - Async execution (callback)
        - Priority queueing
        """
        import uuid
        request_id = str(uuid.uuid4())
        
        # Get model
        model = self.models.get(request.model_id)
        if not model:
            raise ValueError(f"Model not found: {request.model_id}")
        
        # Check if async callback requested
        if request.callback_url:
            # Queue for async processing
            await self.queue.enqueue(request_id, request)
            
            return ExecutionResponse(
                request_id=request_id,
                model_id=request.model_id,
                output={},
                execution_time_ms=0,
                status="queued"
            )
        else:
            # Sync execution
            return await self._execute_model(request_id, request, model)
    
    async def _execute_model(
        self,
        request_id: str,
        request: ExecutionRequest,
        model: ModelMetadata
    ) -> ExecutionResponse:
        """Execute model inference."""
        import httpx
        import time
        
        start_time = time.time()
        
        try:
            # Call model endpoint
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{model.endpoint_url}/predict",
                    json=request.input_data,
                    timeout=model.timeout_seconds
                )
                response.raise_for_status()
                output = response.json()
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record metrics
            await self._record_execution_metrics(
                model_id=request.model_id,
                execution_time_ms=execution_time_ms,
                success=True
            )
            
            return ExecutionResponse(
                request_id=request_id,
                model_id=request.model_id,
                output=output,
                execution_time_ms=execution_time_ms,
                status="completed"
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Record failure
            await self._record_execution_metrics(
                model_id=request.model_id,
                execution_time_ms=execution_time_ms,
                success=False
            )
            
            return ExecutionResponse(
                request_id=request_id,
                model_id=request.model_id,
                output={},
                execution_time_ms=execution_time_ms,
                status="failed",
                error=str(e)
            )
    
    async def _record_execution_metrics(
        self,
        model_id: str,
        execution_time_ms: int,
        success: bool
    ):
        """Record execution metrics for monitoring."""
        # Send to metrics system (Prometheus, ClickHouse)
        pass

# ================== AUTO-SCALING ==================

class AutoScaler:
    """
    Auto-scaling logic for model instances.
    
    Scales based on:
    - Queue depth
    - CPU/GPU utilization
    - Request latency
    - Custom metrics
    """
    
    def __init__(self, kubernetes_client):
        self.k8s = kubernetes_client
        
        self.scaling_config = {
            "min_replicas": 2,
            "max_replicas": 20,
            "target_queue_depth": 100,
            "scale_up_threshold": 150,
            "scale_down_threshold": 50
        }
    
    async def check_and_scale(self, model_id: str):
        """
        Check metrics and scale if needed.
        
        Called periodically by scheduler.
        """
        # Get current metrics
        queue_depth = await self.get_queue_depth(model_id)
        current_replicas = await self.k8s.get_replica_count(model_id)
        
        # Determine if scaling needed
        if queue_depth > self.scaling_config["scale_up_threshold"]:
            # Scale up
            new_replicas = min(
                current_replicas + 2,
                self.scaling_config["max_replicas"]
            )
            await self.k8s.scale_deployment(model_id, new_replicas)
        
        elif queue_depth < self.scaling_config["scale_down_threshold"]:
            # Scale down
            new_replicas = max(
                current_replicas - 1,
                self.scaling_config["min_replicas"]
            )
            await self.k8s.scale_deployment(model_id, new_replicas)
```

**Discussion Points:**
- Model serving strategies (TensorFlow Serving, Triton, Custom)
- GPU resource management and scheduling
- Model versioning and A/B testing
- Cold start optimization
- Cost optimization (spot instances, model caching)

---

### Q3: Design the observability strategy for Bhashini 2.0

**Expected Answer:**

```python
"""
Observability Strategy for Bhashini 2.0

THREE PILLARS:

1. METRICS (Prometheus + Grafana)
   - Request rate, latency, errors
   - Resource utilization (CPU, Memory, GPU)
   - Business metrics (subscriptions, revenue)
   - Custom metrics per service

2. LOGGING (ELK Stack or Loki)
   - Structured JSON logging
   - Correlation IDs for tracing
   - Log levels and sampling
   - Log aggregation and search

3. TRACING (Jaeger or Zipkin)
   - Distributed request tracing
   - Service dependency mapping
   - Performance bottleneck identification
   - Error propagation tracking
"""

from opentelemetry import trace, metrics
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import Counter, Histogram, Gauge
import logging
import json
from datetime import datetime
from typing import Dict, Any

# ================== STRUCTURED LOGGING ==================

class StructuredLogger:
    """
    Structured logging for better log analysis.
    
    Outputs JSON logs with context.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.logger = logging.getLogger(service_name)
        
        # JSON formatter
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log(
        self,
        level: str,
        message: str,
        **kwargs
    ):
        """Log with structured context."""
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "service": self.service_name,
            "level": level,
            "message": message,
            **kwargs
        }
        
        getattr(self.logger, level.lower())(json.dumps(log_data))

class JsonFormatter(logging.Formatter):
    """JSON log formatter."""
    
    def format(self, record):
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "service": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

# ================== METRICS COLLECTION ==================

class MetricsCollector:
    """
    Collects application and business metrics.
    
    Prometheus-compatible metrics.
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        
        # API metrics
        self.request_counter = Counter(
            'api_requests_total',
            'Total API requests',
            ['method', 'endpoint', 'status_code']
        )
        
        self.request_duration = Histogram(
            'api_request_duration_seconds',
            'API request duration',
            ['method', 'endpoint']
        )
        
        self.active_requests = Gauge(
            'api_active_requests',
            'Number of active requests',
            ['endpoint']
        )
        
        # Model metrics
        self.model_inference_duration = Histogram(
            'model_inference_duration_ms',
            'Model inference time',
            ['model_id', 'model_type']
        )
        
        self.model_error_counter = Counter(
            'model_errors_total',
            'Model execution errors',
            ['model_id', 'error_type']
        )
        
        # Business metrics
        self.subscription_counter = Counter(
            'subscriptions_total',
            'Total subscriptions',
            ['api_id', 'tier']
        )
        
        self.revenue_gauge = Gauge(
            'revenue_total',
            'Total revenue',
            ['currency']
        )
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration_seconds: float
    ):
        """Record API request metrics."""
        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration_seconds)
    
    def record_model_inference(
        self,
        model_id: str,
        model_type: str,
        duration_ms: int,
        success: bool
    ):
        """Record model inference metrics."""
        self.model_inference_duration.labels(
            model_id=model_id,
            model_type=model_type
        ).observe(duration_ms)
        
        if not success:
            self.model_error_counter.labels(
                model_id=model_id,
                error_type="inference_failed"
            ).inc()

# ================== DISTRIBUTED TRACING ==================

class TracingManager:
    """
    Distributed tracing across microservices.
    
    Tracks requests across service boundaries.
    """
    
    def __init__(self, service_name: str, jaeger_host: str = "localhost"):
        # Setup tracer
        trace.set_tracer_provider(TracerProvider())
        
        jaeger_exporter = JaegerExporter(
            agent_host_name=jaeger_host,
            agent_port=6831,
        )
        
        trace.get_tracer_provider().add_span_processor(
            BatchSpanProcessor(jaeger_exporter)
        )
        
        self.tracer = trace.get_tracer(service_name)
    
    def trace_function(self, span_name: str):
        """Decorator to trace function execution."""
        def decorator(func):
            async def wrapper(*args, **kwargs):
                with self.tracer.start_as_current_span(span_name) as span:
                    # Add attributes
                    span.set_attribute("function", func.__name__)
                    
                    try:
                        result = await func(*args, **kwargs)
                        span.set_attribute("status", "success")
                        return result
                    except Exception as e:
                        span.set_attribute("status", "error")
                        span.set_attribute("error.message", str(e))
                        span.record_exception(e)
                        raise
            
            return wrapper
        return decorator

# ================== FASTAPI WITH OBSERVABILITY ==================

from fastapi import FastAPI, Request
import time

app = FastAPI(title="Bhashini 2.0 SEM")

# Initialize observability
logger = StructuredLogger("bhashini-sem")
metrics = MetricsCollector("bhashini-sem")
tracing = TracingManager("bhashini-sem")

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Middleware for metrics and logging
@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    """Add observability to all requests."""
    # Generate request ID
    import uuid
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Start timer
    start_time = time.time()
    
    # Log request
    logger.log(
        "info",
        "Request started",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host
    )
    
    # Track active requests
    metrics.active_requests.labels(endpoint=request.url.path).inc()
    
    try:
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Record metrics
        metrics.record_request(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code,
            duration_seconds=duration
        )
        
        # Log response
        logger.log(
            "info",
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
            duration_ms=int(duration * 1000)
        )
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{int(duration * 1000)}ms"
        
        return response
        
    except Exception as e:
        # Log error
        logger.log(
            "error",
            "Request failed",
            request_id=request_id,
            error=str(e),
            exc_info=True
        )
        raise
        
    finally:
        # Track active requests
        metrics.active_requests.labels(endpoint=request.url.path).dec()

# Model execution with tracing
@app.post("/models/{model_id}/execute")
@tracing.trace_function("execute_model")
async def execute_model(
    model_id: str,
    request: ExecutionRequest,
    http_request: Request
):
    """Execute model with full observability."""
    request_id = http_request.state.request_id
    
    logger.log(
        "info",
        "Model execution started",
        request_id=request_id,
        model_id=model_id
    )
    
    start_time = time.time()
    
    try:
        # Execute
        result = await orchestrator.execute(request)
        
        # Record success metrics
        execution_time_ms = int((time.time() - start_time) * 1000)
        metrics.record_model_inference(
            model_id=model_id,
            model_type="translation",  # Get from model metadata
            duration_ms=execution_time_ms,
            success=True
        )
        
        return result
        
    except Exception as e:
        # Record error metrics
        execution_time_ms = int((time.time() - start_time) * 1000)
        metrics.record_model_inference(
            model_id=model_id,
            model_type="translation",
            duration_ms=execution_time_ms,
            success=False
        )
        
        logger.log(
            "error",
            "Model execution failed",
            request_id=request_id,
            model_id=model_id,
            error=str(e)
        )
        
        raise HTTPException(status_code=500, detail=str(e))

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import generate_latest
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

---

## Tarento-Specific Technical Questions

### Q4: How would you handle the multilingual aspects of Bhashini?

**Expected Discussion:**

```python
"""
Multilingual Architecture Strategy

CHALLENGES:
1. Multiple languages (22 Indian languages + English)
2. Different scripts (Devanagari, Tamil, Telugu, etc.)
3. RTL languages (Urdu, Kashmiri)
4. Unicode handling
5. Language-specific models

SOLUTIONS:

1. UTF-8 EVERYWHERE
   - All text stored in UTF-8
   - Database collation: utf8mb4
   - API responses in UTF-8

2. LANGUAGE DETECTION
   - Auto-detect language from text
   - Store language metadata with content

3. LANGUAGE-SPECIFIC PROCESSING
   - Different models per language pair
   - Language-specific tokenization
   - Script normalization

4. I18N/L10N FRAMEWORK
   - Internationalized error messages
   - Localized UI text
   - Currency and date formatting
"""

from typing import Dict, List, Optional
from enum import Enum

class Language(str, Enum):
    """Supported languages in Bhashini."""
    HINDI = "hi"
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    URDU = "ur"
    ENGLISH = "en"
    # ... more languages

class LanguageService:
    """
    Service for language-specific operations.
    
    Handles:
    - Language detection
    - Script normalization
    - Model routing
    """
    
    def __init__(self, model_registry):
        self.model_registry = model_registry
        
        # Language metadata
        self.language_metadata = {
            Language.HINDI: {
                "script": "Devanagari",
                "direction": "ltr",
                "unicode_range": (0x0900, 0x097F)
            },
            Language.URDU: {
                "script": "Arabic",
                "direction": "rtl",
                "unicode_range": (0x0600, 0x06FF)
            },
            # ... more languages
        }
    
    async def detect_language(self, text: str) -> Language:
        """
        Detect language from text.
        
        Uses character ranges and ML-based detection.
        """
        # Method 1: Script-based detection
        for lang, metadata in self.language_metadata.items():
            if self._contains_script(text, metadata["unicode_range"]):
                return lang
        
        # Method 2: ML-based detection (fastText, langdetect)
        # detected = await self.ml_language_detector.detect(text)
        
        return Language.ENGLISH  # Default fallback
    
    def _contains_script(self, text: str, unicode_range: tuple) -> bool:
        """Check if text contains characters from Unicode range."""
        start, end = unicode_range
        return any(start <= ord(char) <= end for char in text)
    
    async def get_translation_model(
        self,
        source_lang: Language,
        target_lang: Language
    ) -> Optional[str]:
        """
        Get appropriate translation model for language pair.
        
        Returns model_id or None if not available.
        """
        model = await self.model_registry.find_model(
            model_type=ModelType.TRANSLATION,
            source_lang=source_lang.value,
            target_lang=target_lang.value
        )
        
        return model.model_id if model else None
    
    def normalize_text(self, text: str, language: Language) -> str:
        """
        Normalize text for processing.
        
        - Remove zero-width characters
        - Normalize Unicode (NFC vs NFD)
        - Handle RTL markers
        """
        import unicodedata
        
        # Unicode normalization
        normalized = unicodedata.normalize('NFC', text)
        
        # Language-specific normalization
        if language == Language.URDU:
            # RTL text handling
            normalized = self._handle_rtl(normalized)
        
        return normalized
    
    def _handle_rtl(self, text: str) -> str:
        """Handle right-to-left text."""
        # Add RTL markers if needed
        return f"\u202B{text}\u202C"  # RLE (Right-to-Left Embedding)

# ================== API ENDPOINT ==================

@app.post("/translate")
async def translate_text(
    text: str,
    source_lang: Optional[Language] = None,
    target_lang: Language = Language.ENGLISH
):
    """
    Translate text between languages.
    
    Auto-detects source language if not provided.
    """
    language_service = get_language_service()
    
    # Detect source language if not provided
    if not source_lang:
        source_lang = await language_service.detect_language(text)
    
    # Get appropriate model
    model_id = await language_service.get_translation_model(
        source_lang=source_lang,
        target_lang=target_lang
    )
    
    if not model_id:
        raise HTTPException(
            status_code=400,
            detail=f"Translation not available for {source_lang} -> {target_lang}"
        )
    
    # Normalize text
    normalized_text = language_service.normalize_text(text, source_lang)
    
    # Execute translation
    result = await model_executor.execute(
        model_id=model_id,
        input_data={"text": normalized_text}
    )
    
    return {
        "source_language": source_lang,
        "target_language": target_lang,
        "original_text": text,
        "translated_text": result["output"]["text"]
    }
```

---

### Q5: Design the Partner Portal authentication and authorization system

**Expected Answer:**

```python
"""
Partner Portal Auth Architecture

REQUIREMENTS:
1. Multi-tenant partner isolation
2. Role-based access control (RBAC)
3. API key management
4. OAuth2 for third-party integrations
5. Audit logging of all actions

ROLES:
- Partner Admin: Manage partner account, register APIs
- Partner Developer: Access analytics, manage API keys
- Bhashini Admin: Approve partners, monitor platform
- End User: Consume APIs through subscriptions
"""

from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from typing import List, Optional, Set
from pydantic import BaseModel
from datetime import datetime, timedelta
from enum import Enum

class Role(str, Enum):
    """System roles."""
    PLATFORM_ADMIN = "platform_admin"
    PARTNER_ADMIN = "partner_admin"
    PARTNER_DEVELOPER = "partner_developer"
    API_CONSUMER = "api_consumer"

class Permission(str, Enum):
    """Granular permissions."""
    # Partner permissions
    REGISTER_API = "register_api"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_API_KEYS = "manage_api_keys"
    UPDATE_API_METADATA = "update_api_metadata"
    
    # Platform permissions
    APPROVE_PARTNERS = "approve_partners"
    VIEW_ALL_ANALYTICS = "view_all_analytics"
    MANAGE_PLATFORM = "manage_platform"
    
    # Consumer permissions
    SUBSCRIBE_API = "subscribe_api"
    INVOKE_API = "invoke_api"

# Role-Permission mapping
ROLE_PERMISSIONS: Dict[Role, Set[Permission]] = {
    Role.PLATFORM_ADMIN: {
        Permission.APPROVE_PARTNERS,
        Permission.VIEW_ALL_ANALYTICS,
        Permission.MANAGE_PLATFORM
    },
    Role.PARTNER_ADMIN: {
        Permission.REGISTER_API,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_API_KEYS,
        Permission.UPDATE_API_METADATA
    },
    Role.PARTNER_DEVELOPER: {
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_API_KEYS
    },
    Role.API_CONSUMER: {
        Permission.SUBSCRIBE_API,
        Permission.INVOKE_API
    }
}

class User(BaseModel):
    """User model."""
    user_id: int
    email: str
    roles: List[Role]
    partner_id: Optional[str] = None  # For partner users
    is_active: bool = True

class TokenPayload(BaseModel):
    """JWT token payload."""
    sub: int  # user_id
    email: str
    roles: List[Role]
    partner_id: Optional[str] = None
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation

# ================== AUTHENTICATION SERVICE ==================

class AuthenticationService:
    """
    Handles authentication for partner portal.
    
    Supports:
    - JWT tokens
    - API keys
    - OAuth2
    """
    
    def __init__(self, user_repository, token_blacklist, config):
        self.user_repo = user_repository
        self.blacklist = token_blacklist
        self.secret_key = config["jwt_secret"]
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
    
    async def create_access_token(self, user: User) -> str:
        """Create JWT access token."""
        import uuid
        
        payload = TokenPayload(
            sub=user.user_id,
            email=user.email,
            roles=user.roles,
            partner_id=user.partner_id,
            exp=datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes),
            iat=datetime.utcnow(),
            jti=str(uuid.uuid4())
        )
        
        token = jwt.encode(
            payload.dict(),
            self.secret_key,
            algorithm=self.algorithm
        )
        
        return token
    
    async def verify_token(self, token: str) -> TokenPayload:
        """
        Verify JWT token.
        
        Checks:
        - Signature validity
        - Expiration
        - Blacklist (for logout/revocation)
        """
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            token_data = TokenPayload(**payload)
            
            # Check if token is blacklisted
            is_blacklisted = await self.blacklist.is_blacklisted(token_data.jti)
            if is_blacklisted:
                raise HTTPException(status_code=401, detail="Token has been revoked")
            
            return token_data
            
        except JWTError as e:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    async def revoke_token(self, token: str) -> None:
        """Revoke token (add to blacklist)."""
        token_data = await self.verify_token(token)
        
        # Add to blacklist until expiration
        ttl = int((token_data.exp - datetime.utcnow()).total_seconds())
        await self.blacklist.add(token_data.jti, ttl)

# ================== AUTHORIZATION SERVICE ==================

class AuthorizationService:
    """
    Handles fine-grained authorization.
    
    Features:
    - Role-based access control
    - Resource-level permissions
    - Partner isolation
    """
    
    def __init__(self):
        self.role_permissions = ROLE_PERMISSIONS
    
    def check_permission(
        self,
        user: User,
        required_permission: Permission,
        resource_partner_id: Optional[str] = None
    ) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user: Current user
            required_permission: Permission to check
            resource_partner_id: Partner ID of resource (for isolation)
            
        Returns:
            True if authorized
        """
        # Check if any of user's roles grant the permission
        user_permissions = set()
        for role in user.roles:
            user_permissions.update(self.role_permissions.get(role, set()))
        
        if required_permission not in user_permissions:
            return False
        
        # Check partner isolation (partners can only access their own resources)
        if resource_partner_id and user.partner_id:
            if user.partner_id != resource_partner_id:
                # Only platform admin can access other partners' resources
                return Role.PLATFORM_ADMIN in user.roles
        
        return True
    
    def require_permission(self, permission: Permission):
        """
        Dependency for requiring specific permission.
        
        Usage: @app.get("/endpoint", dependencies=[Depends(require_permission(Permission.XXX))])
        """
        async def permission_checker(current_user: User = Depends(get_current_user)):
            if not self.check_permission(current_user, permission):
                raise HTTPException(
                    status_code=403,
                    detail=f"Missing required permission: {permission.value}"
                )
            return current_user
        
        return permission_checker

# ================== DEPENDENCIES ==================

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
auth_service = AuthenticationService(user_repo, token_blacklist, config)
authz_service = AuthorizationService()

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Dependency to get current authenticated user."""
    token_payload = await auth_service.verify_token(token)
    
    # Load user from database
    user = await user_repo.find_by_id(token_payload.sub)
    
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    return user

# ================== PROTECTED ENDPOINTS ==================

# Partner Admin endpoints
@app.post("/partners/{partner_id}/apis")
async def register_api_endpoint(
    partner_id: str,
    api_metadata: APIMetadata,
    current_user: User = Depends(authz_service.require_permission(Permission.REGISTER_API))
):
    """
    Register new API (Partner Admin only).
    
    Enforces partner isolation.
    """
    # Check partner isolation
    if not authz_service.check_permission(
        current_user,
        Permission.REGISTER_API,
        resource_partner_id=partner_id
    ):
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Register API
    api_id = await partner_portal.register_api(partner_id, api_metadata)
    
    return {"api_id": api_id, "message": "API registered successfully"}

# Analytics endpoint
@app.get("/partners/{partner_id}/analytics")
async def get_analytics(
    partner_id: str,
    start_date: datetime,
    end_date: datetime,
    current_user: User = Depends(authz_service.require_permission(Permission.VIEW_ANALYTICS))
):
    """Get partner analytics."""
    # Check permission with partner isolation
    if not authz_service.check_permission(
        current_user,
        Permission.VIEW_ANALYTICS,
        resource_partner_id=partner_id
    ):
        raise HTTPException(status_code=403, detail="Access denied")
    
    analytics = await partner_portal.get_partner_analytics(
        partner_id=partner_id,
        start_date=start_date,
        end_date=end_date
    )
    
    return analytics

# Platform Admin endpoints
@app.post("/admin/partners/{partner_id}/approve")
async def approve_partner(
    partner_id: str,
    current_user: User = Depends(authz_service.require_permission(Permission.APPROVE_PARTNERS))
):
    """Approve partner registration (Platform Admin only)."""
    await partner_repo.update_status(partner_id, "approved")
    
    # Send notification
    partner = await partner_repo.find_by_id(partner_id)
    await notification_service.send_email(
        to=partner["email"],
        template="partner_approved",
        data={"partner_id": partner_id}
    )
    
    return {"message": "Partner approved"}
```

---

## CI/CD & Reliability Engineering

### Q6: Design CI/CD pipeline for Bhashini 2.0 microservices

**Expected Answer:**

```yaml
# ================== GITHUB ACTIONS WORKFLOW ==================

name: Bhashini 2.0 CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'
  REGISTRY: ghcr.io
  IMAGE_NAME: bhashini/api-marketplace

jobs:
  # ================== QUALITY CHECKS ==================
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      
      - name: Cache dependencies
        uses: actions/cache@v3
        with:
          path: ~/.cache/pip
          key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install black flake8 mypy pylint
      
      - name: Format check (Black)
        run: black --check src/
      
      - name: Lint (Flake8)
        run: flake8 src/ --count --statistics
      
      - name: Type check (MyPy)
        run: mypy src/ --strict
      
      - name: Security scan (Bandit)
        run: |
          pip install bandit
          bandit -r src/ -f json -o bandit-report.json
      
      - name: Dependency vulnerability scan
        run: |
          pip install safety
          safety check --json

  # ================== TESTING ==================
  test:
    runs-on: ubuntu-latest
    needs: code-quality
    
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: testpass
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Run unit tests
        run: |
          pytest tests/unit/ -v --cov=src --cov-report=xml
      
      - name: Run integration tests
        run: |
          pytest tests/integration/ -v
        env:
          DATABASE_URL: postgresql://postgres:testpass@localhost/test
          REDIS_URL: redis://localhost:6379
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
          fail_ci_if_error: true

  # ================== BUILD & PUSH ==================
  build-and-push:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2
      
      - name: Log in to Container Registry
        uses: docker/login-action@v2
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v4
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=sha,prefix={{branch}}-
            type=semver,pattern={{version}}
      
      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          build-args: |
            PYTHON_VERSION=${{ env.PYTHON_VERSION }}

  # ================== DEPLOY TO STAGING ==================
  deploy-staging:
    runs-on: ubuntu-latest
    needs: build-and-push
    environment: staging
    
    steps:
      - name: Deploy to staging
        run: |
          # Update Kubernetes deployment
          kubectl set image deployment/api-marketplace \
            api-marketplace=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n bhashini-staging
          
          # Wait for rollout
          kubectl rollout status deployment/api-marketplace -n bhashini-staging
      
      - name: Run smoke tests
        run: |
          python scripts/smoke_tests.py --env staging
      
      - name: Run load tests
        run: |
          k6 run tests/load/api_marketplace_load_test.js

  # ================== DEPLOY TO PRODUCTION ==================
  deploy-production:
    runs-on: ubuntu-latest
    needs: deploy-staging
    environment: production
    if: github.ref == 'refs/heads/main'
    
    steps:
      - name: Create deployment record
        run: |
          echo "Deployment $(date)" >> deployment.log
      
      - name: Deploy to production (Blue-Green)
        run: |
          # Deploy to green environment
          kubectl set image deployment/api-marketplace-green \
            api-marketplace=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ github.sha }} \
            -n bhashini-prod
          
          # Wait for green to be ready
          kubectl rollout status deployment/api-marketplace-green -n bhashini-prod
          
          # Run health checks
          python scripts/health_check.py --env production-green
          
          # Switch traffic to green
          kubectl patch service api-marketplace \
            -p '{"spec":{"selector":{"version":"green"}}}' \
            -n bhashini-prod
          
          # Keep blue for rollback (manual cleanup after verification)
      
      - name: Notify deployment
        run: |
          curl -X POST ${{ secrets.SLACK_WEBHOOK }} \
            -d '{"text":"Bhashini API Marketplace deployed to production"}'
```

**Python Code for Deployment Automation:**

```python
# deployment_automation.py

import asyncio
from kubernetes import client, config
from typing import Dict, List
import time

class DeploymentAutomation:
    """
    Automates blue-green deployment on Kubernetes.
    
    Features:
    - Zero-downtime deployment
    - Automated rollback on failure
    - Health check validation
    - Traffic shifting
    """
    
    def __init__(self, namespace: str = "bhashini-prod"):
        config.load_kube_config()
        self.apps_v1 = client.AppsV1Api()
        self.core_v1 = client.CoreV1Api()
        self.namespace = namespace
    
    async def deploy_blue_green(
        self,
        service_name: str,
        new_image: str,
        health_check_url: str
    ) -> bool:
        """
        Execute blue-green deployment.
        
        Steps:
        1. Deploy to green environment
        2. Wait for green pods to be ready
        3. Run health checks on green
        4. Switch traffic to green
        5. Keep blue for rollback
        """
        green_deployment = f"{service_name}-green"
        blue_deployment = f"{service_name}-blue"
        
        # 1. Update green deployment with new image
        print(f"Deploying {new_image} to {green_deployment}")
        
        deployment = self.apps_v1.read_namespaced_deployment(
            name=green_deployment,
            namespace=self.namespace
        )
        
        deployment.spec.template.spec.containers[0].image = new_image
        
        self.apps_v1.patch_namespaced_deployment(
            name=green_deployment,
            namespace=self.namespace,
            body=deployment
        )
        
        # 2. Wait for rollout to complete
        print("Waiting for green deployment to be ready...")
        await self._wait_for_rollout(green_deployment)
        
        # 3. Run health checks
        print("Running health checks on green...")
        is_healthy = await self._run_health_checks(health_check_url)
        
        if not is_healthy:
            print("Health checks failed! Rolling back...")
            await self.rollback(service_name)
            return False
        
        # 4. Switch traffic to green
        print("Switching traffic to green...")
        await self._switch_traffic(service_name, "green")
        
        # 5. Monitor for issues
        print("Monitoring green environment...")
        await asyncio.sleep(300)  # Monitor for 5 minutes
        
        # Check if green is stable
        is_stable = await self._check_stability(green_deployment)
        
        if not is_stable:
            print("Green environment unstable! Rolling back...")
            await self._switch_traffic(service_name, "blue")
            return False
        
        print("Deployment successful!")
        return True
    
    async def _wait_for_rollout(
        self,
        deployment_name: str,
        timeout: int = 600
    ) -> bool:
        """Wait for deployment rollout to complete."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            deployment = self.apps_v1.read_namespaced_deployment(
                name=deployment_name,
                namespace=self.namespace
            )
            
            if deployment.status.ready_replicas == deployment.spec.replicas:
                return True
            
            await asyncio.sleep(5)
        
        return False
    
    async def _run_health_checks(self, health_check_url: str) -> bool:
        """Run comprehensive health checks."""
        import httpx
        
        # Run multiple health check attempts
        success_count = 0
        total_attempts = 10
        
        async with httpx.AsyncClient() as client:
            for _ in range(total_attempts):
                try:
                    response = await client.get(health_check_url, timeout=5)
                    if response.status_code == 200:
                        success_count += 1
                except:
                    pass
                
                await asyncio.sleep(2)
        
        # Require 80% success rate
        return (success_count / total_attempts) >= 0.8
    
    async def _switch_traffic(self, service_name: str, target_version: str):
        """Switch service traffic to target version."""
        service = self.core_v1.read_namespaced_service(
            name=service_name,
            namespace=self.namespace
        )
        
        service.spec.selector["version"] = target_version
        
        self.core_v1.patch_namespaced_service(
            name=service_name,
            namespace=self.namespace,
            body=service
        )
    
    async def _check_stability(self, deployment_name: str) -> bool:
        """Check if deployment is stable (no restarts, errors)."""
        # Check pod restart count, error logs, metrics
        # Return True if stable
        return True
    
    async def rollback(self, service_name: str) -> None:
        """Rollback deployment to previous version."""
        # Switch traffic back to blue
        await self._switch_traffic(service_name, "blue")
        print("Rolled back to blue environment")
```

---

## Behavioral & Leadership Questions

### Q7: How would you govern technical standards across development teams?

**Expected Answer:**

**1. Establish Coding Standards:**
```python
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: ['--max-line-length=100']
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
```

**2. Architecture Decision Records (ADRs):**
```markdown
# ADR: Use FastAPI for all microservices

## Status
Accepted

## Context
Need to standardize web framework across Bhashini 2.0 services.

## Decision
Use FastAPI for all new Python microservices.

## Consequences
Positive:
- Async support built-in
- Automatic OpenAPI documentation
- Type safety with Pydantic
- High performance

Negative:
- Learning curve for team
- Migration effort for existing services
```

**3. Reference Implementations:**
```python
# Service Template Repository Structure

bhashini-service-template/
├── src/
│   ├── api/
│   │   ├── v1/
│   │   │   └── endpoints/
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── services/
│   ├── repositories/
│   └── models/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── docs/
│   ├── architecture.md
│   └── api.md
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

**4. Code Review Guidelines:**
- All PRs require 2 approvals (1 from architect)
- Automated checks must pass
- Architecture changes require ADR
- Performance impact assessment for critical paths
- Security review for authentication/authorization changes

**5. Technical Guilds:**
- Monthly architecture review meetings
- Tech talks and knowledge sharing
- Pair programming sessions
- Quarterly tech radar updates

---

## Real-World Problem Solving

### Q8: The API Marketplace is experiencing 5-second response times. How do you diagnose and fix?

**Systematic Approach:**

```python
"""
Performance Debugging Methodology

STEP 1: IDENTIFY BOTTLENECK
- Check application metrics (Prometheus/Grafana)
- Review APM traces (DataDog, New Relic)
- Analyze database query times
- Check external API latencies
- Review resource utilization (CPU, Memory, Network)

STEP 2: REPRODUCE ISSUE
- Identify slow endpoints
- Isolate test case
- Profile code execution
- Analyze database queries

STEP 3: IMPLEMENT FIX
- Quick wins: Add caching, optimize queries
- Medium term: Refactor code, add indexes
- Long term: Architectural changes

STEP 4: VERIFY & MONITOR
- Load testing
- Continuous monitoring
- Alerting setup
"""

# Performance Profiling
from fastapi import Request
import cProfile
import pstats
from functools import wraps
import time

class PerformanceProfiler:
    """Profile slow endpoints in production."""
    
    def __init__(self):
        self.profiles = {}
    
    def profile_endpoint(self, path: str):
        """Decorator to profile endpoint performance."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Enable profiling for 1% of requests (sampling)
                import random
                should_profile = random.random() < 0.01
                
                if should_profile:
                    profiler = cProfile.Profile()
                    profiler.enable()
                
                start_time = time.time()
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    
                    if should_profile:
                        profiler.disable()
                        
                        # Save profile
                        stats = pstats.Stats(profiler)
                        stats.sort_stats('cumulative')
                        
                        # Store for analysis
                        self.profiles[f"{path}_{int(time.time())}"] = stats
                        
                        # Log slow queries
                        if duration > 1.0:  # Threshold: 1 second
                            print(f"SLOW QUERY: {path} took {duration:.2f}s")
                            stats.print_stats(20)
            
            return wrapper
        return decorator

profiler = PerformanceProfiler()

# Apply to slow endpoint
@app.get("/marketplace/apis")
@profiler.profile_endpoint("/marketplace/apis")
async def list_apis_with_profiling():
    """Endpoint with profiling enabled."""
    pass

# Database query optimization
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

async def get_apis_optimized(db: AsyncSession) -> List[Dict]:
    """
    Optimized query with:
    - Eager loading (avoid N+1)
    - Selective columns
    - Proper indexing
    """
    # Bad: N+1 query problem
    # apis = await db.execute(select(API))
    # for api in apis:
    #     provider = await db.execute(select(Provider).where(Provider.id == api.provider_id))
    
    # Good: Eager loading
    stmt = select(API).options(
        selectinload(API.provider),
        selectinload(API.pricing)
    ).where(
        API.is_active == True
    ).limit(100)
    
    result = await db.execute(stmt)
    return [api.to_dict() for api in result.scalars()]

# Add caching layer
from functools import lru_cache
import aioredis

class CachedAPIService:
    """Service with intelligent caching."""
    
    def __init__(self, repository, cache):
        self.repo = repository
        self.cache = cache
    
    @lru_cache(maxsize=100)  # L1: Memory cache
    async def get_popular_apis(self) -> List[Dict]:
        """Get popular APIs with multi-level caching."""
        cache_key = "apis:popular"
        
        # L2: Redis cache
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # L3: Database
        apis = await self.repo.get_popular_apis(limit=10)
        
        # Cache for 5 minutes
        await self.cache.set(cache_key, json.dumps(apis), expire=300)
        
        return apis
```

**Fixes Applied:**
1. **Database Optimization**
   - Add indexes on frequently queried columns
   - Use eager loading to avoid N+1 queries
   - Implement read replicas for read-heavy queries

2. **Caching Strategy**
   - L1: Local memory cache (LRU)
   - L2: Redis distributed cache
   - Cache popular/static data aggressively

3. **Connection Pooling**
   - Increase pool size
   - Tune pool parameters
   - Monitor connection usage

4. **Async Optimization**
   - Use asyncio.gather for concurrent operations
   - Avoid blocking operations
   - Stream large responses

---

## Technical Roadmap Questions

### Q9: Create a 6-month technical roadmap for Bhashini 2.0

**Expected Answer:**

```markdown
# Bhashini 2.0 Technical Roadmap

## Q1 (Months 1-3): Foundation & Core Services

### Month 1: Infrastructure & Platform Setup
**Objectives:**
- Set up Kubernetes cluster (multi-AZ)
- Configure CI/CD pipelines
- Establish observability stack
- Set up development environments

**Deliverables:**
- K8s cluster with auto-scaling
- Complete CI/CD pipeline
- Monitoring dashboards
- Development environment templates

### Month 2: Core Services Development
**Objectives:**
- API Marketplace MVP
- Partner Portal MVP
- Basic SEM module
- Authentication & Authorization

**Deliverables:**
- API catalog and discovery
- Partner registration workflow
- Model execution engine
- OAuth2 implementation

### Month 3: Integration & Testing
**Objectives:**
- Integrate all core services
- Performance testing
- Security audit
- Documentation

**Deliverables:**
- End-to-end integration
- Load test results (10K RPS)
- Security scan reports
- API documentation

## Q2 (Months 4-6): Enhancement & Scale

### Month 4: Advanced Features
**Objectives:**
- Usage analytics and billing
- Advanced rate limiting
- Model versioning
- A/B testing framework

**Deliverables:**
- Real-time analytics dashboard
- Tiered rate limiting
- Model deployment pipeline
- A/B testing platform

### Month 5: Optimization & Reliability
**Objectives:**
- Performance optimization
- High availability setup
- Disaster recovery
- Cost optimization

**Deliverables:**
- Response time < 200ms (p95)
- 99.9% uptime SLA
- DR runbooks
- 30% cost reduction

### Month 6: Scale & Launch
**Objectives:**
- Production launch preparation
- Partner onboarding
- Training and documentation
- Post-launch monitoring

**Deliverables:**
- Production-ready platform
- 10+ partners onboarded
- Complete documentation
- 24/7 support setup

## Success Metrics
- API response time: < 200ms (p95)
- System uptime: > 99.9%
- Partner satisfaction: > 90%
- Platform cost per request: < ₹0.10
```

---

## Additional Tarento-Specific Topics

### Q10: Explain your experience with government/public sector projects

**Key Points to Cover:**
- Compliance requirements (data sovereignty, privacy)
- Security standards (CERT-In guidelines)
- Scalability for public usage (millions of users)
- Multi-language support (constitutional requirement)
- Accessibility standards (WCAG 2.1 Level AA)
- Open-source preferences
- Procurement and vendor management

### Q11: How would you ensure Bhashini platform is accessible to resource-constrained users?

**Strategy:**
- Lightweight API responses
- Progressive enhancement
- Offline capabilities
- Low-bandwidth optimization
- Regional CDN deployment
- Free tier with generous limits

---

## 🎯 Interview Preparation Checklist

**Technical Preparation:**
- [ ] Review Flask and FastAPI advanced features
- [ ] Study microservices patterns
- [ ] Understand Kubernetes deployment strategies
- [ ] Review distributed systems concepts
- [ ] Practice system design problems
- [ ] Study Bhashini 1.0 architecture (if available)

**Domain Knowledge:**
- [ ] Understand Indian language computing
- [ ] Learn about speech/translation technologies
- [ ] Research government IT initiatives
- [ ] Study digital India programs

**Soft Skills:**
- [ ] Prepare for stakeholder management scenarios
- [ ] Practice explaining complex concepts simply
- [ ] Prepare examples of technical leadership
- [ ] Review conflict resolution approaches

---

## 📚 Resources

- **Bhashini Platform**: https://bhashini.gov.in/
- **Digital India**: https://www.digitalindia.gov.in/
- **Flask Documentation**: https://flask.palletsprojects.com/
- **FastAPI Documentation**: https://fastapi.tiangolo.com/
- **Microservices Patterns**: https://microservices.io/patterns/
- **System Design Primer**: https://github.com/donnemartin/system-design-primer

---

**Good luck with your Tarento Technologies interview!** 🚀🏗️

*Remember: Focus on scalability, multilingual support, and public sector requirements specific to Bhashini 2.0*

