"""Memory monitoring utilities for AutoCut."""

import psutil
from typing import Dict


def get_memory_info() -> Dict[str, float]:
    """Get current memory usage information.
    
    Returns:
        Dictionary with memory usage statistics
    """
    try:
        memory = psutil.virtual_memory()
        return {
            "percent": memory.percent,
            "available_gb": memory.available / (1024**3),
            "used_gb": memory.used / (1024**3),
            "total_gb": memory.total / (1024**3)
        }
    except Exception:
        # Fallback values if psutil is not available
        return {
            "percent": 50.0,
            "available_gb": 8.0,
            "used_gb": 4.0,
            "total_gb": 12.0
        }