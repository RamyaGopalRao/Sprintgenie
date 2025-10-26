# 💻 Tarento Technologies - Python Coding Interview Questions

## Complete Coding Challenge Guide for Python Developer/Architect Roles

---

## 📚 Table of Contents

1. [Data Structures & Algorithms](#data-structures--algorithms)
2. [String Manipulation & NLP](#string-manipulation--nlp)
3. [API Design & Implementation](#api-design--implementation)
4. [Database & ORM](#database--orm)
5. [Concurrency & Async](#concurrency--async)
6. [System Design Coding](#system-design-coding)
7. [Real-World Scenarios](#real-world-scenarios)
8. [Debugging & Code Review](#debugging--code-review)

---

## Data Structures & Algorithms

### Q1: Implement a LRU Cache with O(1) operations

**Problem:**
Design and implement a data structure for Least Recently Used (LRU) cache. It should support `get` and `put` operations in O(1) time complexity.

**Solution:**

```python
from collections import OrderedDict
from typing import Optional, Any

class LRUCache:
    """
    LRU Cache implementation using OrderedDict.
    
    Time Complexity: O(1) for both get and put
    Space Complexity: O(capacity)
    """
    
    def __init__(self, capacity: int):
        """
        Initialize LRU cache.
        
        Args:
            capacity: Maximum number of items in cache
        """
        if capacity <= 0:
            raise ValueError("Capacity must be positive")
        
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache. Move to end (most recently used).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        
        Time: O(1)
        """
        if key not in self.cache:
            return None
        
        # Move to end (mark as recently used)
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key: str, value: Any) -> None:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
        
        Time: O(1)
        """
        if key in self.cache:
            # Update existing key and move to end
            self.cache.move_to_end(key)
        
        self.cache[key] = value
        
        # Evict least recently used if over capacity
        if len(self.cache) > self.capacity:
            # Remove first item (least recently used)
            self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

# Alternative: Manual implementation with dict + doubly linked list
class Node:
    """Node for doubly linked list."""
    def __init__(self, key: str, value: Any):
        self.key = key
        self.value = value
        self.prev: Optional[Node] = None
        self.next: Optional[Node] = None

class LRUCacheManual:
    """
    LRU Cache with manual doubly linked list.
    
    Demonstrates understanding of data structures.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: Dict[str, Node] = {}
        
        # Dummy head and tail for easier manipulation
        self.head = Node("", "")
        self.tail = Node("", "")
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def _add_to_end(self, node: Node) -> None:
        """Add node to end (most recent)."""
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node
    
    def _remove_node(self, node: Node) -> None:
        """Remove node from list."""
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node
    
    def get(self, key: str) -> Optional[Any]:
        """Get value and move to end."""
        if key not in self.cache:
            return None
        
        node = self.cache[key]
        
        # Move to end
        self._remove_node(node)
        self._add_to_end(node)
        
        return node.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache."""
        if key in self.cache:
            # Remove old node
            self._remove_node(self.cache[key])
        
        # Create new node
        new_node = Node(key, value)
        self.cache[key] = new_node
        self._add_to_end(new_node)
        
        # Evict if over capacity
        if len(self.cache) > self.capacity:
            # Remove from head (least recent)
            lru_node = self.head.next
            self._remove_node(lru_node)
            del self.cache[lru_node.key]

# Test
cache = LRUCache(capacity=3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
print(cache.get("a"))  # 1, moves "a" to end
cache.put("d", 4)      # Evicts "b" (least recently used)
print(cache.get("b"))  # None (evicted)
```

**Follow-up Questions:**
- How would you make this thread-safe?
- How would you implement TTL (time-to-live)?
- How would you add cache statistics (hit rate, miss rate)?

---

### Q2: Design a rate limiter using sliding window algorithm

**Problem:**
Implement a rate limiter that allows maximum N requests per time window using sliding window algorithm.

**Solution:**

```python
import time
from collections import deque
from typing import Dict, Deque
import threading

class SlidingWindowRateLimiter:
    """
    Rate limiter using sliding window algorithm.
    
    More accurate than fixed window.
    Thread-safe implementation.
    """
    
    def __init__(self, max_requests: int, window_seconds: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, Deque[float]] = {}
        self.lock = threading.Lock()
    
    def is_allowed(self, user_id: str) -> bool:
        """
        Check if request is allowed for user.
        
        Args:
            user_id: Unique user identifier
            
        Returns:
            True if allowed, False if rate limited
        
        Time Complexity: O(k) where k = requests in window
        """
        with self.lock:
            current_time = time.time()
            
            # Initialize user's request queue if needed
            if user_id not in self.requests:
                self.requests[user_id] = deque()
            
            user_requests = self.requests[user_id]
            
            # Remove requests outside current window
            while user_requests and user_requests[0] <= current_time - self.window_seconds:
                user_requests.popleft()
            
            # Check if under limit
            if len(user_requests) < self.max_requests:
                user_requests.append(current_time)
                return True
            
            return False
    
    def get_retry_after(self, user_id: str) -> int:
        """
        Get seconds until next request is allowed.
        
        Args:
            user_id: User identifier
            
        Returns:
            Seconds to wait
        """
        with self.lock:
            if user_id not in self.requests or not self.requests[user_id]:
                return 0
            
            current_time = time.time()
            oldest_request = self.requests[user_id][0]
            time_passed = current_time - oldest_request
            
            return max(0, int(self.window_seconds - time_passed))
    
    def get_remaining_requests(self, user_id: str) -> int:
        """Get remaining requests for user in current window."""
        with self.lock:
            if user_id not in self.requests:
                return self.max_requests
            
            current_time = time.time()
            user_requests = self.requests[user_id]
            
            # Count valid requests in window
            valid_count = sum(
                1 for req_time in user_requests
                if req_time > current_time - self.window_seconds
            )
            
            return max(0, self.max_requests - valid_count)

# Test
limiter = SlidingWindowRateLimiter(max_requests=5, window_seconds=10)

# Simulate requests
for i in range(7):
    allowed = limiter.is_allowed("user123")
    if allowed:
        print(f"Request {i+1}: Allowed (Remaining: {limiter.get_remaining_requests('user123')})")
    else:
        retry_after = limiter.get_retry_after("user123")
        print(f"Request {i+1}: Rate limited! Retry after {retry_after}s")
    time.sleep(1)
```

**Async Version for FastAPI:**

```python
import asyncio
from collections import defaultdict
import aioredis

class AsyncRedisRateLimiter:
    """
    Distributed rate limiter using Redis.
    
    Suitable for multi-instance deployments.
    """
    
    def __init__(self, redis: aioredis.Redis, max_requests: int = 100, window: int = 60):
        self.redis = redis
        self.max_requests = max_requests
        self.window = window
    
    async def is_allowed(self, user_id: str) -> bool:
        """
        Check rate limit using Redis sorted set.
        
        Time Complexity: O(log N + M) where M = expired entries
        """
        key = f"rate_limit:{user_id}"
        current_time = time.time()
        window_start = current_time - self.window
        
        # Use pipeline for atomic operations
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(key, 0, window_start)
        
        # Count requests in window
        pipe.zcard(key)
        
        # Add current request
        pipe.zadd(key, {str(current_time): current_time})
        
        # Set expiration
        pipe.expire(key, self.window)
        
        results = await pipe.execute()
        request_count = results[1]
        
        return request_count < self.max_requests
    
    async def get_usage_info(self, user_id: str) -> Dict[str, Any]:
        """Get usage information for user."""
        key = f"rate_limit:{user_id}"
        current_time = time.time()
        window_start = current_time - self.window
        
        # Count requests in current window
        count = await self.redis.zcount(key, window_start, current_time)
        
        return {
            "user_id": user_id,
            "requests_used": count,
            "requests_limit": self.max_requests,
            "requests_remaining": max(0, self.max_requests - count),
            "window_seconds": self.window
        }
```

---

### Q3: Implement a trie (prefix tree) for autocomplete

**Problem:**
Implement a trie data structure for autocomplete functionality. Support insertion, search, and prefix-based suggestions.

**Solution:**

```python
from typing import List, Dict, Optional, Set

class TrieNode:
    """Node in trie data structure."""
    
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_word: bool = False
        self.word: Optional[str] = None
        self.frequency: int = 0  # For ranking suggestions

class Trie:
    """
    Trie (Prefix Tree) for autocomplete.
    
    Operations:
    - Insert: O(m) where m = word length
    - Search: O(m)
    - StartsWith: O(p + n) where p = prefix length, n = number of results
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.word_count = 0
    
    def insert(self, word: str, frequency: int = 1) -> None:
        """
        Insert word into trie.
        
        Args:
            word: Word to insert
            frequency: Word frequency (for ranking)
        """
        node = self.root
        
        # Traverse/create path for each character
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        
        # Mark end of word
        node.is_end_of_word = True
        node.word = word
        node.frequency = frequency
        self.word_count += 1
    
    def search(self, word: str) -> bool:
        """
        Search for exact word.
        
        Args:
            word: Word to search
            
        Returns:
            True if word exists
        
        Time: O(m)
        """
        node = self._find_node(word.lower())
        return node is not None and node.is_end_of_word
    
    def starts_with(self, prefix: str) -> bool:
        """
        Check if any word starts with prefix.
        
        Args:
            prefix: Prefix to check
            
        Returns:
            True if prefix exists
        
        Time: O(p)
        """
        return self._find_node(prefix.lower()) is not None
    
    def autocomplete(self, prefix: str, max_results: int = 10) -> List[str]:
        """
        Get autocomplete suggestions for prefix.
        
        Args:
            prefix: Prefix to autocomplete
            max_results: Maximum number of suggestions
            
        Returns:
            List of suggested words, sorted by frequency
        
        Time: O(p + n) where n = number of words with prefix
        """
        # Find node for prefix
        node = self._find_node(prefix.lower())
        
        if not node:
            return []
        
        # Collect all words with this prefix
        suggestions = []
        self._collect_words(node, suggestions)
        
        # Sort by frequency (descending) and return top results
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [word for word, freq in suggestions[:max_results]]
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find node corresponding to prefix."""
        node = self.root
        
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        
        return node
    
    def _collect_words(
        self,
        node: TrieNode,
        results: List[tuple],
        max_depth: int = 50
    ) -> None:
        """
        Recursively collect all words from node.
        
        Args:
            node: Starting node
            results: List to collect words
            max_depth: Maximum recursion depth
        """
        if max_depth <= 0:
            return
        
        if node.is_end_of_word:
            results.append((node.word, node.frequency))
        
        for child in node.children.values():
            self._collect_words(child, results, max_depth - 1)
    
    def delete(self, word: str) -> bool:
        """
        Delete word from trie.
        
        Args:
            word: Word to delete
            
        Returns:
            True if word was deleted
        """
        def _delete_recursive(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                if not node.is_end_of_word:
                    return False
                
                node.is_end_of_word = False
                node.word = None
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            child = node.children[char]
            should_delete_child = _delete_recursive(child, word, index + 1)
            
            if should_delete_child:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end_of_word
            
            return False
        
        if _delete_recursive(self.root, word.lower(), 0):
            self.word_count -= 1
            return True
        return False

# Usage Example
trie = Trie()

# Insert words with frequencies
words = [
    ("apple", 100),
    ("application", 50),
    ("apply", 75),
    ("app", 200),
    ("apricot", 30),
    ("banana", 80)
]

for word, freq in words:
    trie.insert(word, freq)

# Autocomplete
suggestions = trie.autocomplete("app", max_results=5)
print(suggestions)  # ['app', 'apple', 'apply', 'application']

# Search
print(trie.search("apple"))  # True
print(trie.search("appl"))   # False
print(trie.starts_with("appl"))  # True
```

**Real-World Application for Bhashini:**
- API name autocomplete in marketplace
- Search suggestions for documentation
- Command completion in CLI tools

---

## String Manipulation & NLP

### Q4: Implement a text similarity checker for Indian languages

**Problem:**
Create a function to calculate similarity between two texts, supporting Unicode and Indic scripts.

**Solution:**

```python
from typing import List, Tuple
import unicodedata
from difflib import SequenceMatcher

class TextSimilarityChecker:
    """
    Text similarity for multilingual content.
    
    Supports:
    - Edit distance (Levenshtein)
    - Jaccard similarity
    - Cosine similarity
    - Character n-gram similarity
    """
    
    def levenshtein_distance(self, str1: str, str2: str) -> int:
        """
        Calculate Levenshtein (edit) distance.
        
        Minimum number of edits (insertions, deletions, substitutions)
        needed to transform str1 into str2.
        
        Time: O(m*n) where m, n are string lengths
        Space: O(m*n)
        """
        m, n = len(str1), len(str2)
        
        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize base cases
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # Deletion
                        dp[i][j-1],      # Insertion
                        dp[i-1][j-1]     # Substitution
                    )
        
        return dp[m][n]
    
    def similarity_score(self, str1: str, str2: str) -> float:
        """
        Calculate similarity score (0.0 to 1.0).
        
        Uses normalized edit distance.
        """
        if not str1 and not str2:
            return 1.0
        
        if not str1 or not str2:
            return 0.0
        
        distance = self.levenshtein_distance(str1, str2)
        max_len = max(len(str1), len(str2))
        
        return 1.0 - (distance / max_len)
    
    def jaccard_similarity(self, str1: str, str2: str) -> float:
        """
        Jaccard similarity based on character sets.
        
        J(A,B) = |A ∩ B| / |A ∪ B|
        """
        set1 = set(str1.lower())
        set2 = set(str2.lower())
        
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def ngram_similarity(self, str1: str, str2: str, n: int = 2) -> float:
        """
        N-gram based similarity.
        
        Good for catching typos and similar words.
        """
        ngrams1 = self._get_ngrams(str1.lower(), n)
        ngrams2 = self._get_ngrams(str2.lower(), n)
        
        if not ngrams1 and not ngrams2:
            return 1.0
        
        intersection = len(ngrams1 & ngrams2)
        union = len(ngrams1 | ngrams2)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_ngrams(self, text: str, n: int) -> Set[str]:
        """Generate character n-grams."""
        return {text[i:i+n] for i in range(len(text) - n + 1)}
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode text.
        
        Important for Indian languages with multiple representations.
        """
        # NFC normalization (canonical composition)
        normalized = unicodedata.normalize('NFC', text)
        
        # Remove zero-width characters
        normalized = ''.join(
            char for char in normalized
            if unicodedata.category(char) != 'Cf'
        )
        
        return normalized
    
    def combined_similarity(
        self,
        str1: str,
        str2: str,
        weights: Dict[str, float] = None
    ) -> float:
        """
        Calculate weighted combined similarity.
        
        Args:
            str1: First string
            str2: Second string
            weights: Weights for different metrics
            
        Returns:
            Combined similarity score (0.0 to 1.0)
        """
        if weights is None:
            weights = {
                "levenshtein": 0.4,
                "jaccard": 0.2,
                "ngram": 0.4
            }
        
        # Normalize text
        str1 = self.normalize_unicode(str1)
        str2 = self.normalize_unicode(str2)
        
        # Calculate individual scores
        scores = {
            "levenshtein": self.similarity_score(str1, str2),
            "jaccard": self.jaccard_similarity(str1, str2),
            "ngram": self.ngram_similarity(str1, str2, n=2)
        }
        
        # Weighted average
        combined = sum(scores[key] * weights[key] for key in scores)
        
        return combined

# Test with Hindi text
checker = TextSimilarityChecker()

text1 = "नमस्ते"  # Hello
text2 = "नमस्कार"  # Hello (formal)
text3 = "धन्यवाद"  # Thank you

print(f"Similarity (नमस्ते, नमस्कार): {checker.combined_similarity(text1, text2):.2f}")
print(f"Similarity (नमस्ते, धन्यवाद): {checker.combined_similarity(text1, text3):.2f}")
```

---

### Q5: Implement a token bucket algorithm for API throttling

**Problem:**
Implement a token bucket algorithm to smooth out API request rates while allowing bursts.

**Solution:**

```python
import time
import threading
from typing import Optional

class TokenBucket:
    """
    Token bucket algorithm for rate limiting.
    
    Allows burst traffic while maintaining average rate.
    Better for real-world API usage patterns.
    """
    
    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        initial_tokens: Optional[int] = None
    ):
        """
        Initialize token bucket.
        
        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per second
            initial_tokens: Initial token count (defaults to capacity)
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = initial_tokens if initial_tokens is not None else capacity
        self.last_refill_time = time.time()
        self.lock = threading.Lock()
    
    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens.
        
        Args:
            tokens: Number of tokens to consume
            
        Returns:
            True if tokens were consumed, False if insufficient tokens
        """
        with self.lock:
            self._refill()
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    def _refill(self) -> None:
        """Refill tokens based on time elapsed."""
        current_time = time.time()
        time_elapsed = current_time - self.last_refill_time
        
        # Calculate tokens to add
        tokens_to_add = time_elapsed * self.refill_rate
        
        # Add tokens (cap at capacity)
        self.tokens = min(self.capacity, self.tokens + tokens_to_add)
        self.last_refill_time = current_time
    
    def get_available_tokens(self) -> int:
        """Get current available tokens."""
        with self.lock:
            self._refill()
            return int(self.tokens)
    
    def wait_for_tokens(self, tokens: int = 1, timeout: Optional[float] = None) -> bool:
        """
        Wait until tokens are available.
        
        Args:
            tokens: Number of tokens needed
            timeout: Maximum wait time in seconds
            
        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            if self.consume(tokens):
                return True
            
            # Check timeout
            if timeout and (time.time() - start_time) >= timeout:
                return False
            
            # Calculate wait time
            with self.lock:
                self._refill()
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
                
            # Wait for tokens to refill
            time.sleep(min(wait_time, 0.1))

# Distributed Token Bucket with Redis
class RedisTokenBucket:
    """
    Distributed token bucket using Redis.
    
    Works across multiple application instances.
    """
    
    def __init__(
        self,
        redis_client,
        bucket_key: str,
        capacity: int,
        refill_rate: float
    ):
        self.redis = redis_client
        self.bucket_key = bucket_key
        self.capacity = capacity
        self.refill_rate = refill_rate
    
    async def consume(self, tokens: int = 1) -> bool:
        """
        Consume tokens using Redis Lua script for atomicity.
        """
        lua_script = """
        local key = KEYS[1]
        local capacity = tonumber(ARGV[1])
        local refill_rate = tonumber(ARGV[2])
        local tokens_requested = tonumber(ARGV[3])
        local current_time = tonumber(ARGV[4])
        
        -- Get current state
        local state = redis.call('HMGET', key, 'tokens', 'last_refill')
        local tokens = tonumber(state[1]) or capacity
        local last_refill = tonumber(state[2]) or current_time
        
        -- Refill tokens
        local time_elapsed = current_time - last_refill
        local tokens_to_add = time_elapsed * refill_rate
        tokens = math.min(capacity, tokens + tokens_to_add)
        
        -- Try to consume
        if tokens >= tokens_requested then
            tokens = tokens - tokens_requested
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            redis.call('EXPIRE', key, 3600)
            return 1
        else
            redis.call('HMSET', key, 'tokens', tokens, 'last_refill', current_time)
            return 0
        end
        """
        
        result = await self.redis.eval(
            lua_script,
            1,  # Number of keys
            self.bucket_key,
            self.capacity,
            self.refill_rate,
            tokens,
            time.time()
        )
        
        return result == 1

# Test
bucket = TokenBucket(capacity=10, refill_rate=1.0)  # 10 tokens, 1 token/second

# Burst: consume 5 tokens rapidly
for i in range(5):
    if bucket.consume():
        print(f"Request {i+1}: Allowed (tokens remaining: {bucket.get_available_tokens()})")

# Try to consume more than capacity
for i in range(8):
    if bucket.consume():
        print(f"Burst {i+1}: Allowed")
    else:
        print(f"Burst {i+1}: Rate limited (tokens: {bucket.get_available_tokens()})")
    time.sleep(0.5)  # Tokens refilling
```

---

## API Design & Implementation

### Q6: Design a RESTful API for multi-language content management

**Problem:**
Design and implement a FastAPI service for managing multilingual content (supporting 22 Indian languages).

**Solution:**

```python
from fastapi import FastAPI, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import DeclarativeBase, relationship
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
from datetime import datetime
from enum import Enum

# ================== MODELS ==================

class Language(str, Enum):
    """Supported languages."""
    HINDI = "hi"
    ENGLISH = "en"
    TAMIL = "ta"
    TELUGU = "te"
    BENGALI = "bn"
    MARATHI = "mr"
    GUJARATI = "gu"
    KANNADA = "kn"
    MALAYALAM = "ml"
    PUNJABI = "pa"
    URDU = "ur"
    ODIA = "or"
    ASSAMESE = "as"
    KASHMIRI = "ks"
    KONKANI = "kok"
    MANIPURI = "mni"
    NEPALI = "ne"
    BODO = "brx"
    DOGRI = "doi"
    MAITHILI = "mai"
    SANTALI = "sat"
    SINDHI = "sd"

class Base(DeclarativeBase):
    pass

class Content(Base):
    """Content model (language-agnostic)."""
    __tablename__ = "contents"
    
    id = Column(Integer, primary_key=True, index=True)
    content_key = Column(String(255), unique=True, nullable=False, index=True)
    category = Column(String(100), index=True)
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    is_active = Column(Boolean, default=True)
    
    # Relationship to translations
    translations = relationship("ContentTranslation", back_populates="content")

class ContentTranslation(Base):
    """Translation for specific language."""
    __tablename__ = "content_translations"
    
    id = Column(Integer, primary_key=True)
    content_id = Column(Integer, ForeignKey("contents.id"), nullable=False)
    language = Column(String(10), nullable=False, index=True)
    title = Column(String(500))
    body = Column(Text)
    meta_description = Column(String(500))
    keywords = Column(Text)  # JSON array
    status = Column(String(20), default="draft")  # draft, published, archived
    translated_by = Column(String(100))
    reviewed_by = Column(String(100))
    created_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now)
    
    content = relationship("Content", back_populates="translations")

# Pydantic schemas
class ContentTranslationCreate(BaseModel):
    """Schema for creating translation."""
    language: Language
    title: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1)
    meta_description: Optional[str] = Field(None, max_length=500)
    keywords: Optional[List[str]] = []
    
    @validator('body')
    def validate_body(cls, v):
        """Validate body contains actual content."""
        if len(v.strip()) < 10:
            raise ValueError("Body must be at least 10 characters")
        return v

class ContentCreate(BaseModel):
    """Schema for creating content."""
    content_key: str = Field(..., min_length=1, max_length=255)
    category: str
    translations: List[ContentTranslationCreate]
    
    @validator('translations')
    def validate_translations(cls, v):
        """Ensure at least one translation."""
        if not v:
            raise ValueError("At least one translation required")
        return v

class ContentResponse(BaseModel):
    """Response schema."""
    id: int
    content_key: str
    category: str
    translations: Dict[str, Dict]
    created_at: datetime
    
    class Config:
        from_attributes = True

# ================== SERVICE LAYER ==================

class ContentService:
    """Service for multilingual content management."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_content(
        self,
        content_key: str,
        category: str,
        translations: List[Dict]
    ) -> Content:
        """
        Create content with multiple translations.
        
        Args:
            content_key: Unique key for content
            category: Content category
            translations: List of translation data
            
        Returns:
            Created content
        """
        # Check if content_key already exists
        existing = await self.db.execute(
            select(Content).where(Content.content_key == content_key)
        )
        if existing.scalar_one_or_none():
            raise ValueError(f"Content with key '{content_key}' already exists")
        
        # Create content
        content = Content(
            content_key=content_key,
            category=category
        )
        self.db.add(content)
        await self.db.flush()  # Get content.id
        
        # Create translations
        for trans_data in translations:
            translation = ContentTranslation(
                content_id=content.id,
                **trans_data
            )
            self.db.add(translation)
        
        await self.db.commit()
        await self.db.refresh(content)
        
        return content
    
    async def get_content(
        self,
        content_key: str,
        language: Optional[Language] = None
    ) -> Optional[Dict]:
        """
        Get content in specific language.
        
        Args:
            content_key: Content key
            language: Desired language (returns all if None)
            
        Returns:
            Content with translations
        """
        from sqlalchemy import select
        from sqlalchemy.orm import selectinload
        
        # Eager load translations
        stmt = select(Content).where(
            Content.content_key == content_key,
            Content.is_active == True
        ).options(selectinload(Content.translations))
        
        result = await self.db.execute(stmt)
        content = result.scalar_one_or_none()
        
        if not content:
            return None
        
        # Filter translations by language if specified
        translations = {}
        for trans in content.translations:
            if language is None or trans.language == language.value:
                if trans.status == "published":
                    translations[trans.language] = {
                        "title": trans.title,
                        "body": trans.body,
                        "meta_description": trans.meta_description
                    }
        
        return {
            "id": content.id,
            "content_key": content.content_key,
            "category": content.category,
            "translations": translations
        }
    
    async def update_translation(
        self,
        content_id: int,
        language: Language,
        updates: Dict
    ) -> bool:
        """Update specific translation."""
        stmt = select(ContentTranslation).where(
            ContentTranslation.content_id == content_id,
            ContentTranslation.language == language.value
        )
        
        result = await self.db.execute(stmt)
        translation = result.scalar_one_or_none()
        
        if not translation:
            return False
        
        # Update fields
        for key, value in updates.items():
            if hasattr(translation, key):
                setattr(translation, key, value)
        
        translation.updated_at = datetime.now()
        await self.db.commit()
        
        return True
    
    async def search_content(
        self,
        query: str,
        language: Language,
        category: Optional[str] = None
    ) -> List[Dict]:
        """
        Full-text search in specific language.
        
        Uses PostgreSQL full-text search.
        """
        from sqlalchemy import func, or_
        
        # Build search query
        stmt = select(Content, ContentTranslation).join(
            ContentTranslation
        ).where(
            ContentTranslation.language == language.value,
            ContentTranslation.status == "published",
            or_(
                ContentTranslation.title.ilike(f"%{query}%"),
                ContentTranslation.body.ilike(f"%{query}%")
            )
        )
        
        if category:
            stmt = stmt.where(Content.category == category)
        
        result = await self.db.execute(stmt)
        rows = result.all()
        
        return [
            {
                "content_key": content.content_key,
                "title": translation.title,
                "body": translation.body[:200] + "...",  # Preview
                "language": translation.language
            }
            for content, translation in rows
        ]

# ================== FASTAPI ENDPOINTS ==================

app = FastAPI(title="Multilingual Content API")

# Database dependency
async def get_db() -> AsyncSession:
    """Get database session."""
    async with session_factory() as session:
        yield session

# Endpoints
@app.post("/content", response_model=ContentResponse, status_code=201)
async def create_content(
    content: ContentCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create multilingual content.
    
    Example:
    POST /content
    {
        "content_key": "homepage.welcome",
        "category": "marketing",
        "translations": [
            {
                "language": "en",
                "title": "Welcome to Bhashini",
                "body": "Breaking language barriers..."
            },
            {
                "language": "hi",
                "title": "भाषिणी में आपका स्वागत है",
                "body": "भाषा की बाधाओं को तोड़ना..."
            }
        ]
    }
    """
    service = ContentService(db)
    
    try:
        content = await service.create_content(
            content_key=content.content_key,
            category=content.category,
            translations=[t.dict() for t in content.translations]
        )
        
        return content
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/content/{content_key}")
async def get_content(
    content_key: str = Path(..., description="Content key"),
    lang: Optional[Language] = Query(None, description="Language code"),
    db: AsyncSession = Depends(get_db)
):
    """
    Get content in specified language.
    
    Example:
    GET /content/homepage.welcome?lang=hi
    """
    service = ContentService(db)
    content = await service.get_content(content_key, lang)
    
    if not content:
        raise HTTPException(status_code=404, detail="Content not found")
    
    # If specific language requested, check if translation exists
    if lang and lang.value not in content["translations"]:
        # Fallback to English
        if "en" in content["translations"]:
            content["translations"] = {"en": content["translations"]["en"]}
            content["fallback_used"] = True
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Translation not available for language: {lang}"
            )
    
    return content

@app.get("/content/search")
async def search_content(
    q: str = Query(..., min_length=2, description="Search query"),
    lang: Language = Query(Language.ENGLISH, description="Search language"),
    category: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """
    Search content in specific language.
    
    Example:
    GET /content/search?q=welcome&lang=hi&category=marketing
    """
    service = ContentService(db)
    results = await service.search_content(q, lang, category)
    
    return {
        "query": q,
        "language": lang,
        "results": results,
        "count": len(results)
    }

@app.put("/content/{content_id}/translations/{language}")
async def update_translation(
    content_id: int,
    language: Language,
    updates: Dict,
    db: AsyncSession = Depends(get_db)
):
    """Update translation for specific language."""
    service = ContentService(db)
    
    success = await service.update_translation(content_id, language, updates)
    
    if not success:
        raise HTTPException(status_code=404, detail="Translation not found")
    
    return {"message": "Translation updated successfully"}

# Bulk translation endpoint
@app.post("/content/{content_id}/translate")
async def auto_translate(
    content_id: int,
    source_language: Language,
    target_languages: List[Language],
    db: AsyncSession = Depends(get_db)
):
    """
    Auto-translate content to multiple languages.
    
    Uses Bhashini translation API.
    """
    # Get source content
    service = ContentService(db)
    content = await service.get_content_by_id(content_id, source_language)
    
    if not content:
        raise HTTPException(status_code=404, detail="Source content not found")
    
    # Translate to each target language
    translation_results = []
    
    for target_lang in target_languages:
        # Call translation API
        translated_title = await translate_text(
            content["title"],
            source_language,
            target_lang
        )
        
        translated_body = await translate_text(
            content["body"],
            source_language,
            target_lang
        )
        
        # Save translation
        translation_data = {
            "language": target_lang.value,
            "title": translated_title,
            "body": translated_body,
            "status": "draft",  # Requires review
            "translated_by": "auto"
        }
        
        await service.create_translation(content_id, translation_data)
        
        translation_results.append({
            "language": target_lang,
            "status": "completed"
        })
    
    return {
        "content_id": content_id,
        "translations_created": len(translation_results),
        "results": translation_results
    }
```

---

## Database & ORM

### Q7: Implement database connection pooling with retry logic

**Problem:**
Create a robust database connection pool manager with retry logic, health checks, and failover.

**Solution:**

```python
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool, QueuePool
from contextlib import asynccontextmanager
from typing import Optional, Dict, List
import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DatabaseConnectionManager:
    """
    Manages database connections with:
    - Connection pooling
    - Automatic retry
    - Health monitoring
    - Failover support
    """
    
    def __init__(
        self,
        primary_url: str,
        replica_urls: Optional[List[str]] = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 3600
    ):
        """
        Initialize database connection manager.
        
        Args:
            primary_url: Primary database URL
            replica_urls: Read replica URLs
            pool_size: Number of permanent connections
            max_overflow: Maximum overflow connections
            pool_timeout: Connection timeout
            pool_recycle: Recycle connections after seconds
        """
        self.primary_url = primary_url
        self.replica_urls = replica_urls or []
        
        # Create primary engine
        self.primary_engine = create_async_engine(
            primary_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_timeout=pool_timeout,
            pool_recycle=pool_recycle,
            pool_pre_ping=True,  # Verify connections before using
            echo=False,
            future=True
        )
        
        # Create replica engines
        self.replica_engines = [
            create_async_engine(
                url,
                poolclass=QueuePool,
                pool_size=pool_size // 2,  # Fewer connections for replicas
                max_overflow=max_overflow // 2,
                pool_pre_ping=True
            )
            for url in replica_urls
        ]
        
        # Session factories
        self.primary_session_factory = async_sessionmaker(
            self.primary_engine,
            class_=AsyncSession,
            expire_on_commit=False
        )
        
        self.replica_session_factories = [
            async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            for engine in self.replica_engines
        ]
        
        # Health status
        self.primary_healthy = True
        self.replica_health = [True] * len(self.replica_engines)
        self.last_health_check = datetime.now()
        self.health_check_interval = timedelta(seconds=30)
    
    @asynccontextmanager
    async def get_session(
        self,
        read_only: bool = False,
        retry_count: int = 3
    ) -> AsyncSession:
        """
        Get database session with retry logic.
        
        Args:
            read_only: Use read replica if available
            retry_count: Number of retry attempts
            
        Yields:
            AsyncSession
        """
        last_exception = None
        
        for attempt in range(retry_count):
            try:
                # Choose appropriate engine
                if read_only and self.replica_engines:
                    # Use healthy replica or fallback to primary
                    session_factory = await self._get_healthy_replica()
                else:
                    session_factory = self.primary_session_factory
                
                async with session_factory() as session:
                    yield session
                    return
                    
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {e}"
                )
                
                # Exponential backoff
                if attempt < retry_count - 1:
                    await asyncio.sleep(2 ** attempt)
        
        # All retries failed
        raise Exception(f"Failed to get database session after {retry_count} attempts: {last_exception}")
    
    async def _get_healthy_replica(self) -> async_sessionmaker:
        """Get session factory for healthy replica."""
        # Find first healthy replica
        for i, is_healthy in enumerate(self.replica_health):
            if is_healthy:
                return self.replica_session_factories[i]
        
        # Fallback to primary
        logger.warning("All replicas unhealthy, using primary")
        return self.primary_session_factory
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check health of all database connections.
        
        Returns:
            Health status for each database
        """
        results = {}
        
        # Check primary
        try:
            async with self.primary_engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            self.primary_healthy = True
            results["primary"] = True
        except Exception as e:
            logger.error(f"Primary database unhealthy: {e}")
            self.primary_healthy = False
            results["primary"] = False
        
        # Check replicas
        for i, engine in enumerate(self.replica_engines):
            try:
                async with engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                self.replica_health[i] = True
                results[f"replica_{i}"] = True
            except Exception as e:
                logger.error(f"Replica {i} unhealthy: {e}")
                self.replica_health[i] = False
                results[f"replica_{i}"] = False
        
        self.last_health_check = datetime.now()
        return results
    
    async def close(self):
        """Close all database connections."""
        await self.primary_engine.dispose()
        
        for engine in self.replica_engines:
            await engine.dispose()

# ================== USAGE IN FASTAPI ==================

from fastapi import FastAPI

app = FastAPI()

# Initialize database manager
db_manager = DatabaseConnectionManager(
    primary_url="postgresql+asyncpg://user:pass@primary:5432/bhashini",
    replica_urls=[
        "postgresql+asyncpg://user:pass@replica1:5432/bhashini",
        "postgresql+asyncpg://user:pass@replica2:5432/bhashini"
    ],
    pool_size=20,
    max_overflow=40
)

@app.on_event("startup")
async def startup():
    """Run health check on startup."""
    health = await db_manager.health_check()
    logger.info(f"Database health: {health}")

@app.on_event("shutdown")
async def shutdown():
    """Close connections on shutdown."""
    await db_manager.close()

# Dependency
async def get_db_session(read_only: bool = False):
    """Dependency to get database session."""
    async with db_manager.get_session(read_only=read_only) as session:
        yield session

# Endpoints
@app.get("/users/{user_id}")
async def get_user(
    user_id: int,
    db: AsyncSession = Depends(lambda: get_db_session(read_only=True))
):
    """Get user (uses read replica)."""
    from sqlalchemy import select
    
    stmt = select(User).where(User.id == user_id)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.post("/users")
async def create_user(
    user_data: Dict,
    db: AsyncSession = Depends(get_db_session)  # Uses primary
):
    """Create user (uses primary database)."""
    user = User(**user_data)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@app.get("/health/database")
async def database_health():
    """Database health check endpoint."""
    health = await db_manager.health_check()
    
    all_healthy = all(health.values())
    status_code = 200 if all_healthy else 503
    
    return JSONResponse(
        content={"status": "healthy" if all_healthy else "degraded", "databases": health},
        status_code=status_code
    )
```

---

## Concurrency & Async

### Q8: Implement a job queue with worker pool

**Problem:**
Create an async job queue system with worker pool for background task processing.

**Solution:**

```python
import asyncio
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import uuid
import logging

logger = logging.getLogger(__name__)

class JobStatus(Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Job:
    """Represents a job in the queue."""
    job_id: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: int = 5  # 1 (low) to 10 (high)
    status: JobStatus = JobStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

class AsyncJobQueue:
    """
    Async job queue with worker pool.
    
    Features:
    - Priority-based execution
    - Worker pool with concurrency control
    - Retry logic
    - Job status tracking
    - Result storage
    """
    
    def __init__(self, num_workers: int = 4, max_queue_size: int = 1000):
        """
        Initialize job queue.
        
        Args:
            num_workers: Number of concurrent workers
            max_queue_size: Maximum queued jobs
        """
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size
        
        # Priority queue (higher priority first)
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=max_queue_size)
        
        # Job tracking
        self.jobs: Dict[str, Job] = {}
        self.active_jobs: Set[str] = set()
        
        # Workers
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        
        # Statistics
        self.stats = {
            "total_jobs": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0
        }
    
    async def start(self):
        """Start worker pool."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Create workers
        for i in range(self.num_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} workers")
    
    async def stop(self, graceful: bool = True):
        """
        Stop worker pool.
        
        Args:
            graceful: Wait for current jobs to complete
        """
        self.is_running = False
        
        if graceful:
            # Wait for queue to empty
            await self.queue.join()
        
        # Cancel workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("All workers stopped")
    
    async def enqueue(
        self,
        func: Callable,
        *args,
        priority: int = 5,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Add job to queue.
        
        Args:
            func: Function to execute
            args: Positional arguments
            priority: Job priority (1-10, higher = more urgent)
            max_retries: Maximum retry attempts
            kwargs: Keyword arguments
            
        Returns:
            Job ID
        """
        job_id = str(uuid.uuid4())
        
        job = Job(
            job_id=job_id,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries
        )
        
        self.jobs[job_id] = job
        self.stats["total_jobs"] += 1
        
        # Add to priority queue (negate priority for max-heap behavior)
        await self.queue.put((-priority, job))
        
        logger.info(f"Job queued: {job_id} (priority: {priority})")
        
        return job_id
    
    async def _worker(self, worker_id: int):
        """
        Worker that processes jobs from queue.
        
        Args:
            worker_id: Worker identifier
        """
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get job from queue (with timeout to check is_running)
                try:
                    _, job = await asyncio.wait_for(
                        self.queue.get(),
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process job
                await self._process_job(job, worker_id)
                
                # Mark task as done
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_job(self, job: Job, worker_id: int):
        """
        Process single job with retry logic.
        
        Args:
            job: Job to process
            worker_id: Worker processing the job
        """
        job.status = JobStatus.RUNNING
        job.started_at = datetime.now()
        self.active_jobs.add(job.job_id)
        
        logger.info(f"Worker {worker_id} processing job {job.job_id}")
        
        try:
            # Execute job function
            if asyncio.iscoroutinefunction(job.func):
                result = await job.func(*job.args, **job.kwargs)
            else:
                # Run sync function in thread pool
                result = await asyncio.to_thread(job.func, *job.args, **job.kwargs)
            
            # Job succeeded
            job.result = result
            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now()
            self.stats["completed"] += 1
            
            logger.info(f"Job {job.job_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Job {job.job_id} failed: {e}")
            
            # Retry logic
            if job.retry_count < job.max_retries:
                job.retry_count += 1
                job.status = JobStatus.PENDING
                
                # Re-queue with exponential backoff
                await asyncio.sleep(2 ** job.retry_count)
                await self.queue.put((-job.priority, job))
                
                logger.info(f"Job {job.job_id} requeued (attempt {job.retry_count + 1})")
            else:
                # Max retries exceeded
                job.status = JobStatus.FAILED
                job.error = str(e)
                job.completed_at = datetime.now()
                self.stats["failed"] += 1
                
                logger.error(f"Job {job.job_id} failed after {job.max_retries} retries")
        
        finally:
            self.active_jobs.remove(job.job_id)
    
    async def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get job status and result.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        job = self.jobs.get(job_id)
        
        if not job:
            return None
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "result": job.result,
            "error": job.error,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "retry_count": job.retry_count
        }
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel pending job."""
        job = self.jobs.get(job_id)
        
        if not job:
            return False
        
        if job.status == JobStatus.PENDING:
            job.status = JobStatus.CANCELLED
            self.stats["cancelled"] += 1
            return True
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "total_jobs": self.stats["total_jobs"],
            "completed": self.stats["completed"],
            "failed": self.stats["failed"],
            "cancelled": self.stats["cancelled"],
            "pending": self.queue.qsize(),
            "active": len(self.active_jobs),
            "workers": len(self.workers)
        }

# ================== EXAMPLE USAGE ==================

# Background jobs
async def send_email(recipient: str, subject: str, body: str):
    """Example async job."""
    logger.info(f"Sending email to {recipient}")
    await asyncio.sleep(2)  # Simulate sending
    return {"sent": True, "recipient": recipient}

async def process_image(image_url: str):
    """Example job."""
    logger.info(f"Processing image: {image_url}")
    await asyncio.sleep(1)
    return {"processed": True, "url": image_url}

# FastAPI integration
from fastapi import FastAPI, BackgroundTasks

app = FastAPI()
job_queue = AsyncJobQueue(num_workers=4)

@app.on_event("startup")
async def startup():
    """Start job queue on app startup."""
    await job_queue.start()

@app.on_event("shutdown")
async def shutdown():
    """Stop job queue gracefully."""
    await job_queue.stop(graceful=True)

@app.post("/jobs/email")
async def queue_email(recipient: str, subject: str, body: str):
    """Queue email sending job."""
    job_id = await job_queue.enqueue(
        send_email,
        recipient,
        subject,
        body,
        priority=8
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Email job queued"
    }

@app.get("/jobs/{job_id}")
async def get_job_status_endpoint(job_id: str):
    """Get job status."""
    status = await job_queue.get_job_status(job_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return status

@app.get("/jobs/stats")
async def get_queue_stats():
    """Get queue statistics."""
    return job_queue.get_statistics()
```

---

## Real-World Scenarios

### Q9: Implement a distributed file upload service

**Problem:**
Design a file upload service that handles large files, supports resumable uploads, and stores files in object storage (S3/MinIO).

**Solution:**

```python
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from typing import Optional, Dict
import aiofiles
import hashlib
import os
from pathlib import Path
import aioboto3
from datetime import datetime, timedelta

class ChunkedFileUploader:
    """
    Handles chunked/resumable file uploads.
    
    Features:
    - Resume interrupted uploads
    - MD5 verification
    - Multi-part uploads for large files
    - Direct upload to S3
    """
    
    def __init__(
        self,
        s3_bucket: str,
        s3_endpoint: str,
        temp_dir: str = "/tmp/uploads"
    ):
        self.s3_bucket = s3_bucket
        self.s3_endpoint = s3_endpoint
        self.temp_dir = Path(temp_dir)
        self.temp_dir.mkdir(exist_ok=True)
        
        # Track upload sessions
        self.upload_sessions: Dict[str, Dict] = {}
        
        # S3 client
        self.session = aioboto3.Session()
    
    async def initiate_upload(
        self,
        filename: str,
        file_size: int,
        content_type: str,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Initiate resumable upload.
        
        Args:
            filename: Original filename
            file_size: Total file size in bytes
            content_type: MIME type
            metadata: Additional metadata
            
        Returns:
            Upload session information
        """
        upload_id = str(uuid.uuid4())
        
        # Calculate chunk size (5MB for S3 multi-part)
        chunk_size = 5 * 1024 * 1024
        total_chunks = (file_size + chunk_size - 1) // chunk_size
        
        # Create upload session
        session = {
            "upload_id": upload_id,
            "filename": filename,
            "file_size": file_size,
            "content_type": content_type,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "uploaded_chunks": set(),
            "metadata": metadata or {},
            "created_at": datetime.now(),
            "s3_key": f"uploads/{upload_id}/{filename}",
            "multipart_upload_id": None
        }
        
        # Initiate S3 multipart upload
        async with self.session.client(
            's3',
            endpoint_url=self.s3_endpoint
        ) as s3:
            response = await s3.create_multipart_upload(
                Bucket=self.s3_bucket,
                Key=session["s3_key"],
                ContentType=content_type,
                Metadata=metadata or {}
            )
            
            session["multipart_upload_id"] = response["UploadId"]
        
        self.upload_sessions[upload_id] = session
        
        return {
            "upload_id": upload_id,
            "chunk_size": chunk_size,
            "total_chunks": total_chunks,
            "expires_in_seconds": 86400  # 24 hours
        }
    
    async def upload_chunk(
        self,
        upload_id: str,
        chunk_number: int,
        chunk_data: bytes
    ) -> Dict:
        """
        Upload single chunk.
        
        Args:
            upload_id: Upload session ID
            chunk_number: Chunk number (1-indexed)
            chunk_data: Chunk bytes
            
        Returns:
            Chunk upload status
        """
        session = self.upload_sessions.get(upload_id)
        
        if not session:
            raise ValueError("Upload session not found")
        
        # Validate chunk number
        if chunk_number < 1 or chunk_number > session["total_chunks"]:
            raise ValueError(f"Invalid chunk number: {chunk_number}")
        
        # Calculate MD5 for verification
        md5_hash = hashlib.md5(chunk_data).hexdigest()
        
        # Upload chunk to S3
        async with self.session.client('s3', endpoint_url=self.s3_endpoint) as s3:
            response = await s3.upload_part(
                Bucket=self.s3_bucket,
                Key=session["s3_key"],
                PartNumber=chunk_number,
                UploadId=session["multipart_upload_id"],
                Body=chunk_data,
                ContentMD5=md5_hash
            )
            
            etag = response["ETag"]
        
        # Track uploaded chunk
        session["uploaded_chunks"].add(chunk_number)
        
        # Store ETag for completion
        if "parts" not in session:
            session["parts"] = {}
        session["parts"][chunk_number] = etag
        
        is_complete = len(session["uploaded_chunks"]) == session["total_chunks"]
        
        return {
            "upload_id": upload_id,
            "chunk_number": chunk_number,
            "uploaded": True,
            "chunks_uploaded": len(session["uploaded_chunks"]),
            "total_chunks": session["total_chunks"],
            "is_complete": is_complete,
            "etag": etag
        }
    
    async def complete_upload(self, upload_id: str) -> Dict:
        """
        Complete multi-part upload.
        
        Args:
            upload_id: Upload session ID
            
        Returns:
            Final upload information
        """
        session = self.upload_sessions.get(upload_id)
        
        if not session:
            raise ValueError("Upload session not found")
        
        # Verify all chunks uploaded
        if len(session["uploaded_chunks"]) != session["total_chunks"]:
            raise ValueError(
                f"Not all chunks uploaded: {len(session['uploaded_chunks'])}/{session['total_chunks']}"
            )
        
        # Complete S3 multipart upload
        parts = [
            {"PartNumber": part_num, "ETag": etag}
            for part_num, etag in sorted(session["parts"].items())
        ]
        
        async with self.session.client('s3', endpoint_url=self.s3_endpoint) as s3:
            response = await s3.complete_multipart_upload(
                Bucket=self.s3_bucket,
                Key=session["s3_key"],
                UploadId=session["multipart_upload_id"],
                MultipartUpload={"Parts": parts}
            )
        
        # Generate presigned URL for download
        async with self.session.client('s3', endpoint_url=self.s3_endpoint) as s3:
            download_url = await s3.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.s3_bucket, 'Key': session["s3_key"]},
                ExpiresIn=3600
            )
        
        return {
            "upload_id": upload_id,
            "filename": session["filename"],
            "file_size": session["file_size"],
            "s3_key": session["s3_key"],
            "download_url": download_url,
            "etag": response["ETag"],
            "completed_at": datetime.now().isoformat()
        }
    
    async def get_upload_status(self, upload_id: str) -> Optional[Dict]:
        """Get upload session status."""
        session = self.upload_sessions.get(upload_id)
        
        if not session:
            return None
        
        return {
            "upload_id": upload_id,
            "filename": session["filename"],
            "file_size": session["file_size"],
            "chunks_uploaded": len(session["uploaded_chunks"]),
            "total_chunks": session["total_chunks"],
            "progress_percentage": (
                len(session["uploaded_chunks"]) / session["total_chunks"] * 100
            )
        }
    
    async def cancel_upload(self, upload_id: str) -> bool:
        """Cancel upload and cleanup."""
        session = self.upload_sessions.get(upload_id)
        
        if not session:
            return False
        
        # Abort S3 multipart upload
        async with self.session.client('s3', endpoint_url=self.s3_endpoint) as s3:
            await s3.abort_multipart_upload(
                Bucket=self.s3_bucket,
                Key=session["s3_key"],
                UploadId=session["multipart_upload_id"]
            )
        
        # Remove session
        del self.upload_sessions[upload_id]
        
        return True

# ================== FASTAPI ENDPOINTS ==================

app = FastAPI()
uploader = ChunkedFileUploader(
    s3_bucket="bhashini-uploads",
    s3_endpoint="https://s3.amazonaws.com"
)

@app.post("/upload/init")
async def initiate_upload(
    filename: str,
    file_size: int,
    content_type: str = "application/octet-stream"
):
    """
    Initiate resumable file upload.
    
    Returns upload_id and chunk information.
    """
    if file_size > 5 * 1024 * 1024 * 1024:  # 5GB limit
        raise HTTPException(status_code=400, detail="File too large (max 5GB)")
    
    session = await uploader.initiate_upload(
        filename=filename,
        file_size=file_size,
        content_type=content_type
    )
    
    return session

@app.put("/upload/{upload_id}/chunk/{chunk_number}")
async def upload_chunk(
    upload_id: str,
    chunk_number: int,
    file: UploadFile = File(...),
    content_md5: Optional[str] = Header(None)
):
    """
    Upload single chunk.
    
    Supports resume from any chunk.
    """
    # Read chunk data
    chunk_data = await file.read()
    
    # Verify MD5 if provided
    if content_md5:
        actual_md5 = hashlib.md5(chunk_data).hexdigest()
        if actual_md5 != content_md5:
            raise HTTPException(status_code=400, detail="MD5 checksum mismatch")
    
    # Upload chunk
    result = await uploader.upload_chunk(upload_id, chunk_number, chunk_data)
    
    return result

@app.post("/upload/{upload_id}/complete")
async def complete_upload(upload_id: str):
    """Complete upload and get file URL."""
    try:
        result = await uploader.complete_upload(upload_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/upload/{upload_id}/status")
async def get_upload_status(upload_id: str):
    """Get upload progress."""
    status = await uploader.get_upload_status(upload_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return status

@app.delete("/upload/{upload_id}")
async def cancel_upload(upload_id: str):
    """Cancel ongoing upload."""
    success = await uploader.cancel_upload(upload_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Upload not found")
    
    return {"message": "Upload cancelled"}
```

---

## Debugging & Code Review

### Q10: Debug and fix this code

**Problem:**
Find and fix the bugs in this code:

```python
# BUGGY CODE
class UserManager:
    users = []  # Bug 1: Mutable class variable
    
    def __init__(self, db):
        self.db = db
    
    async def get_users(self):
        users = await self.db.query("SELECT * FROM users")
        for user in users:
            user['password'] = None  # Bug 2: Modifying original data
        return users
    
    def calculate_stats(self, numbers):
        total = sum(numbers)
        average = total / len(numbers)  # Bug 3: Division by zero
        return {'total': total, 'average': average}
    
    async def process_users(self):
        users = await self.get_users()
        results = []
        for user in users:
            result = await self.process_user(user)  # Bug 4: Sequential async
            results.append(result)
        return results
```

**Fixed Code:**

```python
from typing import List, Dict, Any, Optional
from copy import deepcopy
import asyncio

class UserManager:
    """
    Fixed UserManager class.
    
    Fixes:
    1. Instance variable instead of class variable
    2. Return copy to avoid mutation
    3. Handle edge cases (empty list)
    4. Concurrent async processing
    """
    
    def __init__(self, db):
        self.db = db
        self.users: List[Dict] = []  # Fix 1: Instance variable
    
    async def get_users(self) -> List[Dict]:
        """
        Get users without exposing sensitive data.
        
        Returns:
            List of user dictionaries (sanitized)
        """
        users = await self.db.query("SELECT * FROM users")
        
        # Fix 2: Create copies and sanitize
        sanitized_users = []
        for user in users:
            user_copy = deepcopy(user)  # Don't modify original
            user_copy.pop('password', None)  # Remove password
            user_copy.pop('password_hash', None)  # Remove hash too
            sanitized_users.append(user_copy)
        
        return sanitized_users
    
    def calculate_stats(self, numbers: List[float]) -> Dict[str, float]:
        """
        Calculate statistics with error handling.
        
        Args:
            numbers: List of numbers
            
        Returns:
            Statistics dictionary
        """
        # Fix 3: Handle empty list
        if not numbers:
            return {
                'total': 0.0,
                'average': 0.0,
                'count': 0
            }
        
        total = sum(numbers)
        average = total / len(numbers)
        
        return {
            'total': total,
            'average': average,
            'count': len(numbers),
            'min': min(numbers),
            'max': max(numbers)
        }
    
    async def process_users(self) -> List[Any]:
        """
        Process users concurrently.
        
        Fix 4: Use asyncio.gather for concurrent processing
        """
        users = await self.get_users()
        
        # Process all users concurrently
        results = await asyncio.gather(
            *[self.process_user(user) for user in users],
            return_exceptions=True  # Don't fail if one user fails
        )
        
        # Filter out errors
        successful_results = [
            r for r in results 
            if not isinstance(r, Exception)
        ]
        
        # Log errors
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            logger.error(f"{len(errors)} users failed processing")
        
        return successful_results
    
    async def process_user(self, user: Dict) -> Dict:
        """Process single user."""
        # Simulate processing
        await asyncio.sleep(0.1)
        return {"user_id": user["id"], "processed": True}

# Additional improvements
class ImprovedUserManager(UserManager):
    """
    Further improvements with:
    - Type hints
    - Better error handling
    - Logging
    - Caching
    - Rate limiting
    """
    
    def __init__(self, db, cache, logger):
        super().__init__(db)
        self.cache = cache
        self.logger = logger
    
    async def get_users(
        self,
        include_inactive: bool = False,
        page: int = 1,
        page_size: int = 100
    ) -> Dict[str, Any]:
        """
        Get users with pagination and caching.
        
        Args:
            include_inactive: Include inactive users
            page: Page number
            page_size: Items per page
            
        Returns:
            Paginated user list
        """
        cache_key = f"users:page:{page}:size:{page_size}:inactive:{include_inactive}"
        
        # Check cache
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # Query database with pagination
        offset = (page - 1) * page_size
        
        query = """
            SELECT id, username, email, is_active, created_at
            FROM users
            WHERE 1=1
        """
        
        if not include_inactive:
            query += " AND is_active = true"
        
        query += f" ORDER BY id LIMIT {page_size} OFFSET {offset}"
        
        try:
            users = await self.db.query(query)
            
            # Count total
            count_query = "SELECT COUNT(*) as total FROM users"
            if not include_inactive:
                count_query += " WHERE is_active = true"
            
            total_result = await self.db.query(count_query)
            total = total_result[0]["total"]
            
            result = {
                "users": users,
                "total": total,
                "page": page,
                "page_size": page_size,
                "total_pages": (total + page_size - 1) // page_size
            }
            
            # Cache for 5 minutes
            await self.cache.set(cache_key, json.dumps(result), expire=300)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error fetching users: {e}")
            raise
```

---

## Coding Best Practices to Demonstrate

### 1. Type Hints & Documentation

```python
from typing import List, Dict, Optional, Union, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')

class Repository(Generic[T]):
    """
    Generic repository pattern.
    
    Demonstrates:
    - Generics
    - Type hints
    - Comprehensive docstrings
    """
    
    def __init__(self, model_class: type[T]):
        """
        Initialize repository.
        
        Args:
            model_class: Model class this repository manages
        """
        self.model_class = model_class
    
    async def find_by_id(self, id: int) -> Optional[T]:
        """
        Find entity by ID.
        
        Args:
            id: Entity ID
            
        Returns:
            Entity instance or None if not found
            
        Raises:
            DatabaseError: If database query fails
        """
        pass
```

### 2. Error Handling

```python
class CustomException(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, error_code: str, details: Optional[Dict] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

async def robust_function(param: str) -> Dict:
    """
    Function with comprehensive error handling.
    
    Demonstrates:
    - Input validation
    - Try-except blocks
    - Specific exception handling
    - Logging
    - Cleanup
    """
    # Validate input
    if not param:
        raise ValueError("Parameter cannot be empty")
    
    resource = None
    
    try:
        # Acquire resource
        resource = await acquire_resource()
        
        # Process
        result = await process_data(param)
        
        return result
        
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    
    except ConnectionError as e:
        logger.error(f"Connection error: {e}")
        raise CustomException(
            message="Failed to connect to service",
            error_code="CONNECTION_ERROR",
            details={"original_error": str(e)}
        )
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise CustomException(
            message="Internal error occurred",
            error_code="INTERNAL_ERROR"
        )
    
    finally:
        # Always cleanup
        if resource:
            await release_resource(resource)
```

---

## 🎯 Common Coding Patterns to Know

### Pattern 1: Context Manager

```python
from contextlib import asynccontextmanager

@asynccontextmanager
async def database_transaction(db: AsyncSession):
    """
    Context manager for database transactions.
    
    Automatically commits or rolls back.
    """
    try:
        yield db
        await db.commit()
    except Exception:
        await db.rollback()
        raise
```

### Pattern 2: Decorator for Logging

```python
from functools import wraps
import time

def log_execution_time(func):
    """Decorator to log function execution time."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration = time.time() - start
            logger.info(f"{func.__name__} completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start
            logger.error(f"{func.__name__} failed after {duration:.2f}s: {e}")
            raise
    return wrapper
```

---

## 📝 Coding Interview Tips

1. **Clarify Requirements**
   - Ask about edge cases
   - Confirm input/output formats
   - Understand constraints

2. **Think Out Loud**
   - Explain your approach
   - Discuss trade-offs
   - Mention alternatives

3. **Write Clean Code**
   - Use meaningful names
   - Add comments for complex logic
   - Follow PEP 8

4. **Test Your Code**
   - Walk through test cases
   - Check edge cases
   - Verify error handling

5. **Optimize Gradually**
   - Start with working solution
   - Identify bottlenecks
   - Optimize if needed

---

## 📚 Practice Resources

- **LeetCode**: Python-specific problems
- **HackerRank**: Python certification
- **CodeSignal**: Real-world scenarios
- **GitHub**: Study open-source code
- **Real Python**: Tutorials and best practices

---

**Practice these patterns and you'll ace the Tarento coding interview!** 💻🚀

