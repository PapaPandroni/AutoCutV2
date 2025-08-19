"""
Real-time memory monitoring and adaptive processing for AutoCut V2.

Provides intelligent memory pressure detection and adaptive batch sizing
to maintain optimal performance while preventing system overload.
"""

import psutil
import logging
import time
import threading
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryPressureLevel(Enum):
    """Memory pressure levels for adaptive processing."""
    LOW = "low"           # <50% memory usage
    MODERATE = "moderate" # 50-65% memory usage  
    HIGH = "high"         # 65-80% memory usage
    CRITICAL = "critical" # >80% memory usage


@dataclass
class MemorySnapshot:
    """Memory usage snapshot at a point in time."""
    timestamp: float
    memory_percent: float
    memory_mb: int
    available_mb: int
    pressure_level: MemoryPressureLevel
    swap_percent: float = 0.0
    

class MemoryMonitor:
    """Real-time memory monitoring with adaptive processing capabilities."""
    
    def __init__(self, 
                 warning_threshold: float = 50.0,     # Reduced from 80%
                 critical_threshold: float = 65.0,    # Reduced from 88%
                 check_interval: float = 2.0):
        """
        Initialize memory monitor with conservative thresholds.
        
        Args:
            warning_threshold: Memory % to start conservative processing
            critical_threshold: Memory % to trigger aggressive optimization
            check_interval: Seconds between memory checks
        """
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.check_interval = check_interval
        
        self.current_snapshot: Optional[MemorySnapshot] = None
        self.memory_history: List[MemorySnapshot] = []
        self.max_history_size = 100
        
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: List[Callable[[MemorySnapshot], None]] = []
        
        # Adaptive processing parameters
        self.adaptive_batch_sizes = {
            MemoryPressureLevel.LOW: 10,       # Process 10 clips at once
            MemoryPressureLevel.MODERATE: 6,   # Process 6 clips at once
            MemoryPressureLevel.HIGH: 3,       # Process 3 clips at once 
            MemoryPressureLevel.CRITICAL: 1,   # Process 1 clip at a time
        }
        
        logger.info(f"Memory Monitor initialized:")
        logger.info(f"  Warning threshold: {warning_threshold}%")
        logger.info(f"  Critical threshold: {critical_threshold}%")
        logger.info(f"  Check interval: {check_interval}s")
    
    def get_current_memory_info(self) -> MemorySnapshot:
        """Get current memory usage information."""
        # Virtual memory (RAM)
        vm = psutil.virtual_memory()
        memory_percent = vm.percent
        memory_mb = vm.used // (1024 * 1024)
        available_mb = vm.available // (1024 * 1024)
        
        # Swap memory
        swap = psutil.swap_memory()
        swap_percent = swap.percent if swap.total > 0 else 0.0
        
        # Determine pressure level
        if memory_percent >= self.critical_threshold:
            pressure_level = MemoryPressureLevel.CRITICAL
        elif memory_percent >= self.warning_threshold:
            pressure_level = MemoryPressureLevel.HIGH
        elif memory_percent >= 50.0:
            pressure_level = MemoryPressureLevel.MODERATE
        else:
            pressure_level = MemoryPressureLevel.LOW
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            memory_percent=memory_percent,
            memory_mb=memory_mb,
            available_mb=available_mb,
            pressure_level=pressure_level,
            swap_percent=swap_percent
        )
        
        self.current_snapshot = snapshot
        return snapshot
    
    def start_monitoring(self) -> None:
        """Start background memory monitoring."""
        if self.monitoring_active:
            logger.warning("Memory monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Memory monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop background memory monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                snapshot = self.get_current_memory_info()
                
                # Add to history
                self.memory_history.append(snapshot)
                if len(self.memory_history) > self.max_history_size:
                    self.memory_history.pop(0)
                
                # Trigger callbacks
                for callback in self.callbacks:
                    try:
                        callback(snapshot)
                    except Exception as e:
                        logger.error(f"Memory monitor callback failed: {e}")
                
                # Log pressure changes
                if len(self.memory_history) > 1:
                    prev_level = self.memory_history[-2].pressure_level
                    curr_level = snapshot.pressure_level
                    
                    if prev_level != curr_level:
                        logger.info(f"Memory pressure changed: {prev_level.value} → {curr_level.value} "
                                  f"({snapshot.memory_percent:.1f}%)")
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def add_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """Add callback to be triggered on memory updates."""
        self.callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[MemorySnapshot], None]) -> None:
        """Remove callback from memory updates."""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_adaptive_batch_size(self, default_size: int = 5) -> int:
        """Get recommended batch size based on current memory pressure."""
        if not self.current_snapshot:
            self.get_current_memory_info()
        
        if self.current_snapshot:
            pressure_level = self.current_snapshot.pressure_level
            recommended_size = self.adaptive_batch_sizes.get(pressure_level, default_size)
            
            logger.debug(f"Adaptive batch size: {recommended_size} "
                        f"(pressure: {pressure_level.value}, memory: {self.current_snapshot.memory_percent:.1f}%)")
            return recommended_size
        
        return default_size
    
    def should_pause_processing(self) -> bool:
        """Check if processing should be paused due to memory pressure."""
        if not self.current_snapshot:
            self.get_current_memory_info()
        
        if self.current_snapshot:
            # Pause if critical memory pressure AND swap usage is high
            is_critical = self.current_snapshot.pressure_level == MemoryPressureLevel.CRITICAL
            high_swap = self.current_snapshot.swap_percent > 25.0
            
            should_pause = is_critical and high_swap
            
            if should_pause:
                logger.warning(f"Pausing processing due to memory pressure: "
                             f"{self.current_snapshot.memory_percent:.1f}% RAM, "
                             f"{self.current_snapshot.swap_percent:.1f}% swap")
            
            return should_pause
        
        return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        current = self.get_current_memory_info()
        
        stats = {
            'current_percent': current.memory_percent,
            'current_mb': current.memory_mb,
            'available_mb': current.available_mb,
            'pressure_level': current.pressure_level.value,
            'swap_percent': current.swap_percent,
            'warning_threshold': self.warning_threshold,
            'critical_threshold': self.critical_threshold,
        }
        
        # Add history statistics if available
        if len(self.memory_history) > 1:
            recent_usage = [s.memory_percent for s in self.memory_history[-10:]]
            stats.update({
                'avg_recent_percent': sum(recent_usage) / len(recent_usage),
                'max_recent_percent': max(recent_usage),
                'min_recent_percent': min(recent_usage),
                'history_samples': len(self.memory_history),
            })
        
        return stats
    
    def estimate_available_capacity(self, target_clip_size_mb: int = 100) -> int:
        """Estimate how many clips can be loaded safely."""
        current = self.get_current_memory_info()
        
        # Conservative estimation
        safety_margin_mb = 1024  # Keep 1GB free
        usable_memory_mb = current.available_mb - safety_margin_mb
        
        if usable_memory_mb <= 0:
            return 1  # Always allow at least 1 clip
        
        estimated_clips = usable_memory_mb // target_clip_size_mb
        
        # Apply pressure-based limits
        pressure_limits = {
            MemoryPressureLevel.LOW: float('inf'),
            MemoryPressureLevel.MODERATE: 20,
            MemoryPressureLevel.HIGH: 10,
            MemoryPressureLevel.CRITICAL: 3,
        }
        
        max_clips = pressure_limits.get(current.pressure_level, 5)
        final_estimate = min(estimated_clips, max_clips)
        
        logger.debug(f"Memory capacity estimate: {final_estimate} clips "
                    f"(available: {current.available_mb}MB, "
                    f"pressure: {current.pressure_level.value})")
        
        return max(1, final_estimate)
    
    def wait_for_memory_relief(self, target_threshold: float = None, 
                             timeout: float = 30.0) -> bool:
        """Wait for memory usage to drop below threshold."""
        if target_threshold is None:
            target_threshold = self.warning_threshold
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            current = self.get_current_memory_info()
            
            if current.memory_percent < target_threshold:
                logger.info(f"Memory relief achieved: {current.memory_percent:.1f}% < {target_threshold}%")
                return True
            
            logger.debug(f"Waiting for memory relief: {current.memory_percent:.1f}% >= {target_threshold}%")
            time.sleep(2.0)
        
        logger.warning(f"Memory relief timeout after {timeout}s")
        return False
    
    def force_garbage_collection(self) -> None:
        """Force garbage collection to free memory."""
        import gc
        
        before = self.get_current_memory_info()
        
        # Multiple GC passes for thorough cleanup
        for i in range(3):
            collected = gc.collect()
            logger.debug(f"GC pass {i+1}: collected {collected} objects")
        
        # Small delay for system cleanup
        time.sleep(0.5)
        
        after = self.get_current_memory_info()
        freed_mb = before.memory_mb - after.memory_mb
        
        logger.info(f"Garbage collection: freed {freed_mb}MB "
                   f"({before.memory_percent:.1f}% → {after.memory_percent:.1f}%)")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations based on current state."""
        current = self.get_current_memory_info()
        recommendations = []
        
        if current.pressure_level == MemoryPressureLevel.CRITICAL:
            recommendations.extend([
                "🔴 CRITICAL: Reduce batch size to 1-2 clips maximum",
                "🔴 Consider closing other applications",
                "🔴 Enable aggressive garbage collection",
                "🔴 Use streaming analysis only (no clip loading)",
            ])
        elif current.pressure_level == MemoryPressureLevel.HIGH:
            recommendations.extend([
                "🟡 HIGH: Reduce batch size to 3-5 clips",
                "🟡 Enable frequent garbage collection",
                "🟡 Use lower quality settings if possible",
            ])
        elif current.pressure_level == MemoryPressureLevel.MODERATE:
            recommendations.extend([
                "🟢 MODERATE: Current settings should work well",
                "🟢 Consider reducing batch size if processing 4K videos",
            ])
        else:
            recommendations.append("✅ LOW: Optimal memory conditions for processing")
        
        # Add swap-specific recommendations
        if current.swap_percent > 10:
            recommendations.append(f"⚠️ Swap usage high ({current.swap_percent:.1f}%) - consider restarting system")
        
        return recommendations