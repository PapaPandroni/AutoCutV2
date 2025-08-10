"""Video caching system for AutoCut V2.

Thread-safe video clip caching to reduce redundant loading operations.
Extracted and enhanced from the original god module's VideoCache class.

Features:
- Thread-safe operations with proper locking
- Memory-based size tracking and LRU eviction
- Cache statistics and monitoring
- Configurable cache size limits
"""

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Dict, Any, Set

from moviepy import VideoFileClip

try:
    from core.logging_config import get_logger, log_performance
except ImportError:
    import logging
    def get_logger(name):
        return logging.getLogger(name)
    def log_performance(func):
        return func


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    video_clip: VideoFileClip
    cache_key: str
    access_count: int
    last_access_time: float
    estimated_size_mb: float
    created_time: float

    def touch(self) -> None:
        """Update access time and count."""
        self.access_count += 1
        self.last_access_time = time.time()

    @property
    def age_seconds(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.created_time

    @property
    def idle_time_seconds(self) -> float:
        """Get idle time since last access."""
        return time.time() - self.last_access_time


class VideoCache:
    """Thread-safe video clip cache with LRU eviction.

    Provides efficient caching of loaded video clips to reduce redundant
    I/O operations. Uses LRU (Least Recently Used) eviction policy with
    memory size tracking.

    Features:
    - Thread-safe operations
    - Memory-based size limits
    - LRU eviction policy
    - Access statistics
    - Cache hit/miss tracking
    """

    def __init__(
        self,
        max_size_mb: float = 2048.0,  # 2GB default
        max_entries: int = 100,
        enable_stats: bool = True,
    ):
        """Initialize video cache.

        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of cached entries
            enable_stats: Whether to collect detailed statistics
        """
        self.max_size_mb = max_size_mb
        self.max_entries = max_entries
        self.enable_stats = enable_stats

        # Thread-safe cache storage (OrderedDict maintains insertion/access order)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()  # Reentrant lock for nested calls

        # Size tracking
        self._current_size_mb = 0.0

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "puts": 0,
            "clears": 0,
            "size_evictions": 0,
            "count_evictions": 0,
            "total_access_time": 0.0,
        }

        self.logger = get_logger("autocut.video.loading.VideoCache")

        self.logger.info(
            f"Video cache initialized",
            extra={
                "max_size_mb": max_size_mb,
                "max_entries": max_entries,
                "enable_stats": enable_stats,
            },
        )

    def get(self, cache_key: str) -> Optional[VideoFileClip]:
        """Get video clip from cache.

        Args:
            cache_key: Unique cache key

        Returns:
            Cached video clip or None if not found
        """
        start_time = time.time() if self.enable_stats else 0

        with self._lock:
            entry = self._cache.get(cache_key)

            if entry is not None:
                # Cache hit - update access info and move to end (most recently used)
                entry.touch()
                self._cache.move_to_end(cache_key)

                if self.enable_stats:
                    self._stats["hits"] += 1
                    self._stats["total_access_time"] += time.time() - start_time

                self.logger.debug(
                    f"Cache hit: {cache_key}",
                    extra={
                        "access_count": entry.access_count,
                        "age_seconds": entry.age_seconds,
                    },
                )

                return entry.video_clip
            else:
                # Cache miss
                if self.enable_stats:
                    self._stats["misses"] += 1

                self.logger.debug(f"Cache miss: {cache_key}")

                return None

    def put(
        self, cache_key: str, video_clip: VideoFileClip, estimated_size_mb: float = 50.0
    ) -> bool:
        """Put video clip into cache.

        Args:
            cache_key: Unique cache key
            video_clip: Video clip to cache
            estimated_size_mb: Estimated size in MB (used for eviction)

        Returns:
            True if successfully cached, False otherwise
        """
        if not cache_key or video_clip is None:
            return False

        with self._lock:
            current_time = time.time()

            # Check if key already exists
            if cache_key in self._cache:
                # Update existing entry
                existing_entry = self._cache[cache_key]
                size_delta = estimated_size_mb - existing_entry.estimated_size_mb

                existing_entry.video_clip = video_clip
                existing_entry.estimated_size_mb = estimated_size_mb
                existing_entry.touch()

                self._current_size_mb += size_delta
                self._cache.move_to_end(cache_key)

                self.logger.debug(
                    f"Updated cache entry: {cache_key}",
                    extra={"size_delta_mb": size_delta},
                )
            else:
                # Create new entry
                entry = CacheEntry(
                    video_clip=video_clip,
                    cache_key=cache_key,
                    access_count=1,
                    last_access_time=current_time,
                    estimated_size_mb=estimated_size_mb,
                    created_time=current_time,
                )

                self._cache[cache_key] = entry
                self._current_size_mb += estimated_size_mb

                if self.enable_stats:
                    self._stats["puts"] += 1

                self.logger.debug(
                    f"Added to cache: {cache_key}",
                    extra={
                        "size_mb": estimated_size_mb,
                        "cache_size": len(self._cache),
                        "total_size_mb": self._current_size_mb,
                    },
                )

            # Perform eviction if necessary
            self._evict_if_necessary()

            return True

    def remove(self, cache_key: str) -> bool:
        """Remove specific entry from cache.

        Args:
            cache_key: Cache key to remove

        Returns:
            True if entry was removed, False if not found
        """
        with self._lock:
            entry = self._cache.pop(cache_key, None)

            if entry is not None:
                self._current_size_mb -= entry.estimated_size_mb

                # Clean up the video clip
                try:
                    entry.video_clip.close()
                except Exception as e:
                    self.logger.warning(
                        f"Error closing video clip during removal: {e}",
                        extra={"cache_key": cache_key},
                    )

                self.logger.debug(
                    f"Removed from cache: {cache_key}",
                    extra={"freed_mb": entry.estimated_size_mb},
                )

                return True
            else:
                return False

    def clear(self) -> int:
        """Clear entire cache.

        Returns:
            Number of entries removed
        """
        with self._lock:
            count = len(self._cache)

            # Clean up all video clips
            for entry in self._cache.values():
                try:
                    entry.video_clip.close()
                except Exception as e:
                    self.logger.warning(
                        f"Error closing video clip during clear: {e}",
                        extra={"cache_key": entry.cache_key},
                    )

            self._cache.clear()
            self._current_size_mb = 0.0

            if self.enable_stats:
                self._stats["clears"] += 1

            self.logger.info(f"Cache cleared", extra={"entries_removed": count})

            return count

    def _evict_if_necessary(self) -> None:
        """Evict entries if cache exceeds limits."""
        # Evict by size
        while self._current_size_mb > self.max_size_mb and self._cache:
            self._evict_lru_entry("size limit exceeded")
            if self.enable_stats:
                self._stats["size_evictions"] += 1

        # Evict by count
        while len(self._cache) > self.max_entries:
            self._evict_lru_entry("entry count limit exceeded")
            if self.enable_stats:
                self._stats["count_evictions"] += 1

    def _evict_lru_entry(self, reason: str) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Get LRU entry (first in OrderedDict)
        lru_key, lru_entry = self._cache.popitem(last=False)  # FIFO = LRU
        self._current_size_mb -= lru_entry.estimated_size_mb

        # Clean up video clip
        try:
            lru_entry.video_clip.close()
        except Exception as e:
            self.logger.warning(
                f"Error closing evicted video clip: {e}", extra={"cache_key": lru_key}
            )

        if self.enable_stats:
            self._stats["evictions"] += 1

        self.logger.debug(
            f"Evicted LRU entry: {lru_key}",
            extra={
                "reason": reason,
                "freed_mb": lru_entry.estimated_size_mb,
                "age_seconds": lru_entry.age_seconds,
                "access_count": lru_entry.access_count,
            },
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            hit_rate = 0.0
            total_requests = self._stats["hits"] + self._stats["misses"]
            if total_requests > 0:
                hit_rate = self._stats["hits"] / total_requests * 100

            avg_access_time = 0.0
            if self._stats["hits"] > 0:
                avg_access_time = (
                    self._stats["total_access_time"] / self._stats["hits"] * 1000
                )  # ms

            stats = {
                "cache_size": len(self._cache),
                "current_size_mb": round(self._current_size_mb, 2),
                "max_size_mb": self.max_size_mb,
                "max_entries": self.max_entries,
                "size_utilization_percent": round(
                    self._current_size_mb / self.max_size_mb * 100, 1
                ),
                "count_utilization_percent": round(
                    len(self._cache) / self.max_entries * 100, 1
                ),
                "hit_rate_percent": round(hit_rate, 2),
                "avg_access_time_ms": round(avg_access_time, 2),
                **self._stats.copy(),
            }

            return stats

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information.

        Returns:
            Detailed information about cache contents
        """
        with self._lock:
            entries_info = []

            for key, entry in self._cache.items():
                entries_info.append(
                    {
                        "key": key,
                        "size_mb": entry.estimated_size_mb,
                        "access_count": entry.access_count,
                        "age_seconds": round(entry.age_seconds, 1),
                        "idle_seconds": round(entry.idle_time_seconds, 1),
                    }
                )

            # Sort by most recently used
            entries_info.reverse()

            return {
                "total_entries": len(entries_info),
                "total_size_mb": round(self._current_size_mb, 2),
                "entries": entries_info,
                "statistics": self.get_stats(),
            }

    @log_performance("cache_maintenance")
    def perform_maintenance(self) -> Dict[str, Any]:
        """Perform cache maintenance operations.

        Returns:
            Maintenance report
        """
        with self._lock:
            initial_count = len(self._cache)
            initial_size_mb = self._current_size_mb

            # Find entries that haven't been accessed recently
            current_time = time.time()
            stale_threshold_seconds = 3600  # 1 hour
            stale_keys = []

            for key, entry in self._cache.items():
                if entry.idle_time_seconds > stale_threshold_seconds:
                    stale_keys.append(key)

            # Remove stale entries
            stale_removed = 0
            for key in stale_keys:
                if self.remove(key):
                    stale_removed += 1

            # Force eviction check
            self._evict_if_necessary()

            final_count = len(self._cache)
            final_size_mb = self._current_size_mb

            maintenance_report = {
                "timestamp": current_time,
                "initial_entries": initial_count,
                "final_entries": final_count,
                "initial_size_mb": round(initial_size_mb, 2),
                "final_size_mb": round(final_size_mb, 2),
                "stale_entries_removed": stale_removed,
                "size_freed_mb": round(initial_size_mb - final_size_mb, 2),
                "stale_threshold_seconds": stale_threshold_seconds,
            }

            if stale_removed > 0:
                self.logger.info(
                    f"Cache maintenance completed", extra=maintenance_report
                )

            return maintenance_report

    def configure(
        self, max_size_mb: Optional[float] = None, max_entries: Optional[int] = None
    ) -> None:
        """Update cache configuration.

        Args:
            max_size_mb: New maximum size in MB (None = no change)
            max_entries: New maximum entry count (None = no change)
        """
        with self._lock:
            changes = {}

            if max_size_mb is not None:
                old_size = self.max_size_mb
                self.max_size_mb = max_size_mb
                changes["max_size_mb"] = f"{old_size} -> {max_size_mb}"

            if max_entries is not None:
                old_count = self.max_entries
                self.max_entries = max_entries
                changes["max_entries"] = f"{old_count} -> {max_entries}"

            # Apply new limits immediately
            self._evict_if_necessary()

            if changes:
                self.logger.info(
                    "Cache configuration updated", extra={"changes": changes}
                )

    def __len__(self) -> int:
        """Get number of cached entries."""
        with self._lock:
            return len(self._cache)

    def __contains__(self, cache_key: str) -> bool:
        """Check if key is in cache."""
        with self._lock:
            return cache_key in self._cache

    def keys(self) -> Set[str]:
        """Get set of all cache keys."""
        with self._lock:
            return set(self._cache.keys())


# Convenience functions
def create_default_cache() -> VideoCache:
    """Create video cache with default settings."""
    return VideoCache(
        max_size_mb=2048.0,  # 2GB
        max_entries=100,
        enable_stats=True,
    )


def create_memory_limited_cache(max_memory_gb: float) -> VideoCache:
    """Create cache with specific memory limit.

    Args:
        max_memory_gb: Maximum memory in GB

    Returns:
        Configured video cache
    """
    return VideoCache(
        max_size_mb=max_memory_gb * 1024,
        max_entries=min(200, int(max_memory_gb * 50)),  # ~20MB per entry estimate
        enable_stats=True,
    )
