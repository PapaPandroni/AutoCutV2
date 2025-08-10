"""Resource management for video loading operations.

Handles memory monitoring, resource allocation, and cleanup to prevent
system overload during video processing. Extracted from the original
god module's scattered resource management logic.

Features:
- Real-time memory monitoring
- Resource allocation with limits
- Automatic cleanup on context exit
- System performance tracking
"""

import gc
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any, Optional, Generator, NamedTuple

from ...core.exceptions import VideoProcessingError
from ...core.logging_config import get_logger, log_performance


class ResourceAllocation(NamedTuple):
    """Resource allocation information."""

    memory_mb: float
    thread_count: int
    start_time: float


@dataclass
class SystemResources:
    """System resource snapshot."""

    total_memory_gb: float
    available_memory_gb: float
    used_memory_gb: float
    cpu_count: int
    memory_percent: float
    timestamp: float

    @property
    def memory_pressure(self) -> str:
        """Get memory pressure level."""
        if self.memory_percent < 50:
            return "low"
        elif self.memory_percent < 75:
            return "medium"
        elif self.memory_percent < 90:
            return "high"
        else:
            return "critical"


class MemoryMonitor:
    """Real-time memory monitoring for video operations.

    Provides system memory tracking with configurable thresholds
    and automatic alerts when memory usage becomes critical.
    """

    def __init__(
        self, warning_threshold: float = 75.0, critical_threshold: float = 90.0
    ):
        """Initialize memory monitor.

        Args:
            warning_threshold: Memory usage % to trigger warnings
            critical_threshold: Memory usage % to trigger errors
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.logger = get_logger("autocut.video.loading.MemoryMonitor")
        self._lock = threading.Lock()
        self._last_warning = 0.0
        self._warning_interval = 30.0  # Minimum seconds between warnings

        # Try to import psutil for detailed monitoring
        try:
            import psutil

            self._psutil = psutil
            self._has_psutil = True
        except ImportError:
            self._psutil = None
            self._has_psutil = False
            self.logger.warning(
                "psutil not available - memory monitoring will be limited"
            )

    def get_system_resources(self) -> SystemResources:
        """Get current system resource information."""
        if self._has_psutil:
            memory = self._psutil.virtual_memory()
            return SystemResources(
                total_memory_gb=memory.total / (1024**3),
                available_memory_gb=memory.available / (1024**3),
                used_memory_gb=memory.used / (1024**3),
                cpu_count=self._psutil.cpu_count(),
                memory_percent=memory.percent,
                timestamp=time.time(),
            )
        else:
            # Fallback without psutil
            return SystemResources(
                total_memory_gb=8.0,  # Assume 8GB default
                available_memory_gb=4.0,  # Conservative estimate
                used_memory_gb=4.0,
                cpu_count=os.cpu_count() or 4,
                memory_percent=50.0,  # Conservative estimate
                timestamp=time.time(),
            )

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure.

        Returns:
            True if memory usage is above critical threshold

        Raises:
            VideoProcessingError: If memory usage is critically high
        """
        resources = self.get_system_resources()

        if resources.memory_percent >= self.critical_threshold:
            raise VideoProcessingError(
                f"Critical memory pressure: {resources.memory_percent:.1f}% used",
                error_code="MEMORY_CRITICAL",
                details={
                    "memory_percent": resources.memory_percent,
                    "available_gb": resources.available_memory_gb,
                    "pressure": resources.memory_pressure,
                },
            )

        elif resources.memory_percent >= self.warning_threshold:
            # Rate-limited warnings
            current_time = time.time()
            with self._lock:
                if current_time - self._last_warning > self._warning_interval:
                    self.logger.warning(
                        f"High memory usage: {resources.memory_percent:.1f}%",
                        extra={
                            "available_gb": resources.available_memory_gb,
                            "pressure": resources.memory_pressure,
                        },
                    )
                    self._last_warning = current_time

            return True

        return False

    def get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        resources = self.get_system_resources()
        return resources.available_memory_gb

    def log_memory_snapshot(self, operation: str) -> None:
        """Log current memory state for debugging."""
        resources = self.get_system_resources()
        self.logger.debug(
            f"Memory snapshot during {operation}",
            extra={
                "operation": operation,
                "memory_percent": resources.memory_percent,
                "available_gb": resources.available_memory_gb,
                "pressure": resources.memory_pressure,
            },
        )


class VideoResourceManager:
    """Resource manager for video loading operations.

    Coordinates memory allocation, cleanup, and resource limits
    to prevent system overload during video processing.
    """

    def __init__(
        self, max_memory_gb: Optional[float] = None, max_concurrent_clips: int = 10
    ):
        """Initialize resource manager.

        Args:
            max_memory_gb: Maximum memory to allocate (None = auto-detect)
            max_concurrent_clips: Maximum clips to process simultaneously
        """
        self.memory_monitor = MemoryMonitor()
        self.logger = get_logger("autocut.video.loading.VideoResourceManager")

        # Resource limits
        system_resources = self.memory_monitor.get_system_resources()
        if max_memory_gb is None:
            # Use 50% of available memory by default
            self.max_memory_gb = system_resources.available_memory_gb * 0.5
        else:
            self.max_memory_gb = max_memory_gb

        self.max_concurrent_clips = max_concurrent_clips

        # Resource tracking
        self._lock = threading.Lock()
        self._allocated_memory_mb = 0.0
        self._active_allocations = {}
        self._allocation_id_counter = 0

        # Statistics
        self._stats = {
            "allocations": 0,
            "deallocations": 0,
            "peak_memory_mb": 0.0,
            "memory_warnings": 0,
            "cleanup_operations": 0,
        }

        self.logger.info(
            f"Resource manager initialized",
            extra={
                "max_memory_gb": self.max_memory_gb,
                "max_concurrent_clips": self.max_concurrent_clips,
                "system_memory_gb": system_resources.total_memory_gb,
            },
        )

    @contextmanager
    def allocate_resources(
        self, estimated_memory_mb: float, require_memory_check: bool = True
    ) -> Generator[ResourceAllocation, None, None]:
        """Allocate resources for video processing.

        Args:
            estimated_memory_mb: Expected memory usage in MB
            require_memory_check: Whether to enforce memory limits

        Yields:
            ResourceAllocation with details about allocated resources

        Raises:
            VideoProcessingError: If resources cannot be allocated
        """
        allocation_id = None

        try:
            # Check memory pressure before allocation
            if require_memory_check:
                self.memory_monitor.check_memory_pressure()

            # Allocate resources
            with self._lock:
                # Check if allocation would exceed limits
                projected_memory_gb = (
                    self._allocated_memory_mb + estimated_memory_mb
                ) / 1024
                if projected_memory_gb > self.max_memory_gb:
                    raise VideoProcessingError(
                        f"Memory allocation would exceed limit: "
                        f"{projected_memory_gb:.1f}GB > {self.max_memory_gb:.1f}GB",
                        error_code="MEMORY_LIMIT_EXCEEDED",
                        details={
                            "requested_mb": estimated_memory_mb,
                            "allocated_mb": self._allocated_memory_mb,
                            "limit_gb": self.max_memory_gb,
                        },
                    )

                # Check concurrent clip limit
                if len(self._active_allocations) >= self.max_concurrent_clips:
                    raise VideoProcessingError(
                        f"Too many concurrent operations: "
                        f"{len(self._active_allocations)} >= {self.max_concurrent_clips}",
                        error_code="CONCURRENCY_LIMIT_EXCEEDED",
                    )

                # Create allocation
                self._allocation_id_counter += 1
                allocation_id = self._allocation_id_counter

                allocation = ResourceAllocation(
                    memory_mb=estimated_memory_mb,
                    thread_count=1,
                    start_time=time.time(),
                )

                self._active_allocations[allocation_id] = allocation
                self._allocated_memory_mb += estimated_memory_mb
                self._stats["allocations"] += 1

                # Update peak memory tracking
                if self._allocated_memory_mb > self._stats["peak_memory_mb"]:
                    self._stats["peak_memory_mb"] = self._allocated_memory_mb

                self.logger.debug(
                    f"Allocated resources",
                    extra={
                        "allocation_id": allocation_id,
                        "memory_mb": estimated_memory_mb,
                        "total_allocated_mb": self._allocated_memory_mb,
                        "active_allocations": len(self._active_allocations),
                    },
                )

            # Yield the allocation
            yield allocation

        finally:
            # Clean up allocation
            if allocation_id is not None:
                with self._lock:
                    if allocation_id in self._active_allocations:
                        allocation = self._active_allocations[allocation_id]
                        del self._active_allocations[allocation_id]
                        self._allocated_memory_mb -= allocation.memory_mb
                        self._stats["deallocations"] += 1

                        duration = time.time() - allocation.start_time

                        self.logger.debug(
                            f"Deallocated resources",
                            extra={
                                "allocation_id": allocation_id,
                                "memory_mb": allocation.memory_mb,
                                "duration_s": f"{duration:.2f}",
                                "remaining_allocated_mb": self._allocated_memory_mb,
                            },
                        )

            # Force garbage collection for large allocations
            if estimated_memory_mb > 100:
                self._perform_cleanup()

    def _perform_cleanup(self) -> None:
        """Perform system cleanup to free memory."""
        try:
            # Force garbage collection
            collected = gc.collect()

            if collected > 0:
                self._stats["cleanup_operations"] += 1
                self.logger.debug(
                    f"Garbage collection freed {collected} objects",
                    extra={"objects_collected": collected},
                )

        except Exception as e:
            self.logger.warning(f"Cleanup operation failed: {e}")

    @log_performance("resource_health_check")
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive resource health check.

        Returns:
            Health check results with recommendations
        """
        system_resources = self.memory_monitor.get_system_resources()

        with self._lock:
            allocation_info = {
                "active_allocations": len(self._active_allocations),
                "allocated_memory_mb": self._allocated_memory_mb,
                "allocated_memory_gb": self._allocated_memory_mb / 1024,
            }

        # Determine health status
        health_issues = []
        recommendations = []

        if system_resources.memory_percent > 85:
            health_issues.append("high_system_memory")
            recommendations.append("Consider reducing concurrent operations")

        if self._allocated_memory_mb / 1024 > self.max_memory_gb * 0.8:
            health_issues.append("high_allocated_memory")
            recommendations.append("Close unused video clips")

        if len(self._active_allocations) > self.max_concurrent_clips * 0.8:
            health_issues.append("high_concurrency")
            recommendations.append("Wait for current operations to complete")

        health_status = "healthy" if not health_issues else "warning"
        if system_resources.memory_percent > 95:
            health_status = "critical"

        health_report = {
            "status": health_status,
            "timestamp": time.time(),
            "system_resources": {
                "total_memory_gb": system_resources.total_memory_gb,
                "available_memory_gb": system_resources.available_memory_gb,
                "memory_percent": system_resources.memory_percent,
                "memory_pressure": system_resources.memory_pressure,
                "cpu_count": system_resources.cpu_count,
            },
            "allocation_info": allocation_info,
            "resource_limits": {
                "max_memory_gb": self.max_memory_gb,
                "max_concurrent_clips": self.max_concurrent_clips,
            },
            "issues": health_issues,
            "recommendations": recommendations,
            "statistics": self._stats.copy(),
        }

        if health_issues:
            self.logger.warning(
                f"Resource health check found issues: {health_issues}",
                extra={"recommendations": recommendations},
            )
        else:
            self.logger.debug("Resource health check passed")

        return health_report

    def get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        return self.memory_monitor.get_available_memory_gb()

    def get_stats(self) -> Dict[str, Any]:
        """Get resource manager statistics."""
        with self._lock:
            stats = self._stats.copy()
            stats.update(
                {
                    "active_allocations": len(self._active_allocations),
                    "allocated_memory_mb": self._allocated_memory_mb,
                    "allocated_memory_gb": self._allocated_memory_mb / 1024,
                }
            )
        return stats

    def force_cleanup(self) -> None:
        """Force cleanup of all resources and memory."""
        with self._lock:
            num_allocations = len(self._active_allocations)
            self._active_allocations.clear()
            self._allocated_memory_mb = 0.0

        self._perform_cleanup()

        self.logger.info(
            f"Forced cleanup completed", extra={"cleared_allocations": num_allocations}
        )

    def configure_limits(
        self,
        max_memory_gb: Optional[float] = None,
        max_concurrent_clips: Optional[int] = None,
    ) -> None:
        """Update resource limits.

        Args:
            max_memory_gb: New memory limit (None = no change)
            max_concurrent_clips: New concurrency limit (None = no change)
        """
        changes = {}

        if max_memory_gb is not None:
            old_limit = self.max_memory_gb
            self.max_memory_gb = max_memory_gb
            changes["max_memory_gb"] = f"{old_limit:.1f} -> {max_memory_gb:.1f}"

        if max_concurrent_clips is not None:
            old_limit = self.max_concurrent_clips
            self.max_concurrent_clips = max_concurrent_clips
            changes["max_concurrent_clips"] = f"{old_limit} -> {max_concurrent_clips}"

        if changes:
            self.logger.info("Resource limits updated", extra={"changes": changes})


# Convenience functions for monitoring
def get_system_memory_info() -> Dict[str, float]:
    """Get basic system memory information."""
    monitor = MemoryMonitor()
    resources = monitor.get_system_resources()
    return {
        "total_gb": resources.total_memory_gb,
        "available_gb": resources.available_memory_gb,
        "used_gb": resources.used_memory_gb,
        "percent_used": resources.memory_percent,
        "pressure": resources.memory_pressure,
    }


def check_memory_available(required_gb: float) -> bool:
    """Check if required memory is available.

    Args:
        required_gb: Required memory in GB

    Returns:
        True if memory is available
    """
    monitor = MemoryMonitor()
    available = monitor.get_available_memory_gb()
    return available >= required_gb
