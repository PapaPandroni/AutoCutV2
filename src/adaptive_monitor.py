"""
Adaptive Worker Monitor for AutoCut - Real-Time Safety Monitoring

Provides real-time monitoring of system resources during video processing
and can scale down workers if memory usage becomes dangerous.
"""

import contextlib
import logging
import threading
import time
from typing import Callable, Optional

try:
    import psutil
except ImportError:
    psutil = None


class AdaptiveWorkerMonitor:
    """Real-time worker monitoring and adaptive scaling"""

    def __init__(self, initial_workers: int = 3):
        self.initial_workers = initial_workers
        self.current_workers = initial_workers
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Memory thresholds (percentage of total memory)
        self.warning_threshold = 85.0  # Start being concerned
        self.critical_threshold = 95.0  # Emergency scaling
        self.recovery_threshold = 70.0  # Safe to scale back up

        # Callbacks for worker adjustments
        self.scale_down_callback: Optional[Callable[[int], None]] = None
        self.scale_up_callback: Optional[Callable[[int], None]] = None

        # Monitoring state
        self.consecutive_warnings = 0
        self.last_scale_down_time = 0
        self.min_time_between_adjustments = 30  # seconds

        # Logger for error handling
        self.logger = logging.getLogger(__name__)

    def set_scale_callbacks(
        self,
        scale_down_cb: Callable[[int], None],
        scale_up_cb: Callable[[int], None],
    ):
        """Set callbacks for scaling workers up/down"""
        self.scale_down_callback = scale_down_cb
        self.scale_up_callback = scale_up_cb

    def start_monitoring(self):
        """Start real-time memory monitoring in background thread"""
        if psutil is None:
            return

        if self.monitoring:
            return

        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="AdaptiveWorkerMonitor",
        )
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
            self.monitor_thread = None

    def _monitoring_loop(self):
        """Main monitoring loop running in background thread"""
        while self.monitoring:
            if self._safe_check_and_adjust():
                time.sleep(5.0)  # Check every 5 seconds
            else:
                time.sleep(10.0)  # Wait longer on error

    def _safe_check_and_adjust(self) -> bool:
        """Safe wrapper for _check_and_adjust that handles exceptions outside the loop."""
        try:
            self._check_and_adjust()
        except Exception:
            self.logger.exception("Monitoring check failed")
            return False
        else:
            return True

    def _check_and_adjust(self):
        """Check memory usage and adjust workers if needed"""
        if not psutil:
            return

        try:
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            current_time = time.time()

            # Critical threshold - immediate action
            if memory_percent >= self.critical_threshold:
                if self.current_workers > 1:
                    new_workers = max(1, self.current_workers // 2)
                    self._scale_down(
                        new_workers,
                        f"CRITICAL memory usage: {memory_percent:.1f}%",
                    )

            # Warning threshold - gradual scaling
            elif memory_percent >= self.warning_threshold:
                self.consecutive_warnings += 1

                # Scale down after 2 consecutive warnings (10 seconds)
                if (
                    self.consecutive_warnings >= 2
                    and self.current_workers > 1
                    and current_time - self.last_scale_down_time
                    > self.min_time_between_adjustments
                ):
                    new_workers = max(1, self.current_workers - 1)
                    self._scale_down(
                        new_workers,
                        f"HIGH memory usage: {memory_percent:.1f}% (warning #{self.consecutive_warnings})",
                    )

            # Recovery threshold - scale back up gradually
            elif memory_percent <= self.recovery_threshold:
                self.consecutive_warnings = 0

                # Consider scaling up if we're below initial workers
                if (
                    self.current_workers < self.initial_workers
                    and current_time - self.last_scale_down_time
                    > self.min_time_between_adjustments * 2
                ):
                    new_workers = min(self.initial_workers, self.current_workers + 1)
                    self._scale_up(
                        new_workers,
                        f"Memory recovered: {memory_percent:.1f}%",
                    )

            # Normal range
            else:
                self.consecutive_warnings = max(0, self.consecutive_warnings - 1)

        except Exception as e:
            # Log error but continue monitoring
            self.logger.warning(f"Memory check failed: {e}")

    def _scale_down(self, new_workers: int, reason: str):
        """Scale down workers with logging"""
        if new_workers >= self.current_workers:
            return

        self.current_workers = new_workers
        self.last_scale_down_time = time.time()

        if self.scale_down_callback:
            with contextlib.suppress(Exception):
                self.scale_down_callback(new_workers)

    def _scale_up(self, new_workers: int, reason: str):
        """Scale up workers with logging"""
        if new_workers <= self.current_workers:
            return

        self.current_workers = new_workers

        if self.scale_up_callback:
            with contextlib.suppress(Exception):
                self.scale_up_callback(new_workers)

    def get_current_status(self) -> dict:
        """Get current monitoring status"""
        if not psutil:
            return {"monitoring": False, "reason": "psutil not available"}

        try:
            memory = psutil.virtual_memory()
            return {
                "monitoring": self.monitoring,
                "current_workers": self.current_workers,
                "initial_workers": self.initial_workers,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "consecutive_warnings": self.consecutive_warnings,
                "warning_threshold": self.warning_threshold,
                "critical_threshold": self.critical_threshold,
            }
        except Exception as e:
            return {"monitoring": self.monitoring, "error": str(e)}


class ThreadSafeWorkerController:
    """Thread-safe controller for managing worker scaling in ThreadPoolExecutor"""

    def __init__(self, max_workers: int):
        self.max_workers = max_workers
        self.target_workers = max_workers
        self._lock = threading.Lock()
        self.executor = None

    def set_executor(self, executor):
        """Set the executor to control"""
        with self._lock:
            self.executor = executor

    def scale_workers(self, new_count: int):
        """Request worker scaling (note: ThreadPoolExecutor can't dynamically scale)"""
        with self._lock:
            self.target_workers = min(new_count, self.max_workers)

        # Note: ThreadPoolExecutor doesn't support dynamic scaling
        # This is more of a "request" that could be honored in future implementations
        # For now, we just log the request

    def get_status(self) -> dict:
        """Get current controller status"""
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "target_workers": self.target_workers,
                "can_scale_dynamically": False,  # ThreadPoolExecutor limitation
                "executor_active": self.executor is not None,
            }
