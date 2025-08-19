"""
Analysis result caching system for AutoCut V2.

Provides persistent storage of video analysis results to avoid reprocessing
unchanged videos, dramatically improving performance for repeated operations.
"""

import os
import json
import hashlib
import time
import pickle
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
from dataclasses import asdict, dataclass
import threading

from .streaming_analyzer import VideoAnalysisResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry metadata."""
    video_path: str
    file_hash: str
    file_size: int
    file_mtime: float
    analysis_result: VideoAnalysisResult
    cache_timestamp: float
    cache_version: str = "1.0"


class AnalysisCacheManager:
    """Manages persistent caching of video analysis results."""
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 max_cache_size_mb: int = 500,
                 max_cache_age_days: int = 30):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage (default: ~/.autocut/cache)
            max_cache_size_mb: Maximum cache size in MB
            max_cache_age_days: Maximum age of cache entries in days
        """
        self.cache_dir = Path(cache_dir) if cache_dir else self._get_default_cache_dir()
        self.max_cache_size_bytes = max_cache_size_mb * 1024 * 1024
        self.max_cache_age_seconds = max_cache_age_days * 24 * 3600
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache management
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index: Dict[str, Dict[str, Any]] = {}
        self.cache_lock = threading.RLock()
        
        # Load existing cache index
        self._load_cache_index()
        
        # Clean up old entries
        self._cleanup_cache()
        
        logger.info(f"Analysis Cache Manager initialized:")
        logger.info(f"  Cache directory: {self.cache_dir}")
        logger.info(f"  Max cache size: {max_cache_size_mb}MB")
        logger.info(f"  Max cache age: {max_cache_age_days} days")
        logger.info(f"  Current cache entries: {len(self.cache_index)}")
    
    def _get_default_cache_dir(self) -> Path:
        """Get default cache directory."""
        home = Path.home()
        if os.name == 'nt':  # Windows
            cache_base = home / "AppData" / "Local" / "AutoCut"
        elif os.name == 'posix':  # macOS/Linux
            cache_base = home / ".autocut"
        else:
            cache_base = home / ".autocut"
        
        return cache_base / "analysis_cache"
    
    def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        try:
            if self.cache_index_file.exists():
                with open(self.cache_index_file, 'r') as f:
                    self.cache_index = json.load(f)
                logger.debug(f"Loaded cache index with {len(self.cache_index)} entries")
            else:
                self.cache_index = {}
                logger.debug("Created new cache index")
        except Exception as e:
            logger.warning(f"Could not load cache index: {e}")
            self.cache_index = {}
    
    def _save_cache_index(self) -> None:
        """Save cache index to disk."""
        try:
            with open(self.cache_index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
            logger.debug("Cache index saved")
        except Exception as e:
            logger.error(f"Could not save cache index: {e}")
    
    def _get_file_info(self, video_path: str) -> Dict[str, Any]:
        """Get file metadata for cache validation."""
        try:
            stat = os.stat(video_path)
            return {
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'path': os.path.abspath(video_path)
            }
        except Exception as e:
            logger.warning(f"Could not get file info for {video_path}: {e}")
            return {}
    
    def _calculate_file_hash(self, video_path: str, sample_size: int = 1024 * 1024) -> str:
        """Calculate file hash for cache validation (using first 1MB for speed)."""
        try:
            hasher = hashlib.md5()
            
            with open(video_path, 'rb') as f:
                # Hash first part of file for speed
                chunk = f.read(sample_size)
                hasher.update(chunk)
                
                # Also hash file size and mtime for additional validation
                file_info = self._get_file_info(video_path)
                hasher.update(str(file_info.get('size', 0)).encode())
                hasher.update(str(file_info.get('mtime', 0)).encode())
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.warning(f"Could not calculate hash for {video_path}: {e}")
            return ""
    
    def _get_cache_key(self, video_path: str) -> str:
        """Generate cache key for video path."""
        # Use normalized absolute path as base
        abs_path = os.path.abspath(video_path)
        return hashlib.md5(abs_path.encode()).hexdigest()
    
    def _get_cache_file_path(self, cache_key: str) -> Path:
        """Get cache file path for cache key."""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def is_cached(self, video_path: str) -> bool:
        """Check if video analysis is cached and valid."""
        with self.cache_lock:
            cache_key = self._get_cache_key(video_path)
            
            if cache_key not in self.cache_index:
                return False
            
            cache_entry = self.cache_index[cache_key]
            
            # Check if video file still exists
            if not os.path.exists(video_path):
                return False
            
            # Check if file has been modified
            file_info = self._get_file_info(video_path)
            if (file_info.get('size') != cache_entry.get('file_size') or
                file_info.get('mtime') != cache_entry.get('file_mtime')):
                logger.debug(f"Cache invalid for {video_path}: file modified")
                return False
            
            # Check cache age
            cache_age = time.time() - cache_entry.get('cache_timestamp', 0)
            if cache_age > self.max_cache_age_seconds:
                logger.debug(f"Cache expired for {video_path}: {cache_age/3600:.1f}h old")
                return False
            
            # Check if cache file exists
            cache_file = self._get_cache_file_path(cache_key)
            if not cache_file.exists():
                logger.debug(f"Cache file missing for {video_path}")
                return False
            
            return True
    
    def get_cached_analysis(self, video_path: str) -> Optional[VideoAnalysisResult]:
        """Get cached analysis result for video."""
        with self.cache_lock:
            if not self.is_cached(video_path):
                return None
            
            cache_key = self._get_cache_key(video_path)
            cache_file = self._get_cache_file_path(cache_key)
            
            try:
                with open(cache_file, 'rb') as f:
                    cache_entry = pickle.load(f)
                
                logger.info(f"Cache hit for {Path(video_path).name}")
                return cache_entry.analysis_result
                
            except Exception as e:
                logger.warning(f"Could not load cached analysis for {video_path}: {e}")
                # Remove invalid cache entry
                self._remove_cache_entry(cache_key)
                return None
    
    def cache_analysis(self, video_path: str, analysis_result: VideoAnalysisResult) -> None:
        """Cache analysis result for video."""
        with self.cache_lock:
            try:
                cache_key = self._get_cache_key(video_path)
                cache_file = self._get_cache_file_path(cache_key)
                
                # Get file info
                file_info = self._get_file_info(video_path)
                if not file_info:
                    logger.warning(f"Cannot cache analysis: could not get file info for {video_path}")
                    return
                
                # Create cache entry
                cache_entry = CacheEntry(
                    video_path=video_path,
                    file_hash=self._calculate_file_hash(video_path),
                    file_size=file_info['size'],
                    file_mtime=file_info['mtime'],
                    analysis_result=analysis_result,
                    cache_timestamp=time.time()
                )
                
                # Save cache entry to file
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_entry, f)
                
                # Update cache index
                self.cache_index[cache_key] = {
                    'video_path': video_path,
                    'file_size': file_info['size'],
                    'file_mtime': file_info['mtime'],
                    'cache_timestamp': cache_entry.cache_timestamp,
                    'cache_file_size': cache_file.stat().st_size,
                }
                
                self._save_cache_index()
                
                logger.info(f"Cached analysis for {Path(video_path).name}")
                
                # Clean up if cache is getting too large
                self._cleanup_cache()
                
            except Exception as e:
                logger.error(f"Could not cache analysis for {video_path}: {e}")
    
    def _remove_cache_entry(self, cache_key: str) -> None:
        """Remove cache entry and file."""
        try:
            # Remove from index
            if cache_key in self.cache_index:
                del self.cache_index[cache_key]
            
            # Remove cache file
            cache_file = self._get_cache_file_path(cache_key)
            if cache_file.exists():
                cache_file.unlink()
            
            logger.debug(f"Removed cache entry: {cache_key}")
            
        except Exception as e:
            logger.warning(f"Could not remove cache entry {cache_key}: {e}")
    
    def _cleanup_cache(self) -> None:
        """Clean up old and oversized cache entries."""
        try:
            current_time = time.time()
            total_size = 0
            entries_to_remove = []
            
            # Check each cache entry
            for cache_key, cache_info in self.cache_index.items():
                cache_file = self._get_cache_file_path(cache_key)
                
                # Check if file exists
                if not cache_file.exists():
                    entries_to_remove.append(cache_key)
                    continue
                
                # Check age
                cache_age = current_time - cache_info.get('cache_timestamp', 0)
                if cache_age > self.max_cache_age_seconds:
                    entries_to_remove.append(cache_key)
                    continue
                
                # Track total size
                total_size += cache_info.get('cache_file_size', 0)
            
            # Remove expired entries
            for cache_key in entries_to_remove:
                self._remove_cache_entry(cache_key)
            
            # Check total cache size
            if total_size > self.max_cache_size_bytes:
                logger.info(f"Cache size ({total_size/1024/1024:.1f}MB) exceeds limit, cleaning up")
                self._cleanup_by_size()
            
            if entries_to_remove:
                logger.info(f"Cleaned up {len(entries_to_remove)} cache entries")
                self._save_cache_index()
                
        except Exception as e:
            logger.error(f"Cache cleanup failed: {e}")
    
    def _cleanup_by_size(self) -> None:
        """Remove oldest cache entries to stay under size limit."""
        try:
            # Sort entries by age (oldest first)
            sorted_entries = sorted(
                self.cache_index.items(),
                key=lambda x: x[1].get('cache_timestamp', 0)
            )
            
            total_size = sum(info.get('cache_file_size', 0) for _, info in sorted_entries)
            target_size = self.max_cache_size_bytes * 0.8  # Leave 20% buffer
            
            entries_removed = 0
            for cache_key, cache_info in sorted_entries:
                if total_size <= target_size:
                    break
                
                file_size = cache_info.get('cache_file_size', 0)
                self._remove_cache_entry(cache_key)
                total_size -= file_size
                entries_removed += 1
            
            if entries_removed > 0:
                logger.info(f"Removed {entries_removed} entries to reduce cache size")
                self._save_cache_index()
                
        except Exception as e:
            logger.error(f"Size-based cache cleanup failed: {e}")
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        with self.cache_lock:
            try:
                # Remove all cache files
                for cache_file in self.cache_dir.glob("*.pkl"):
                    cache_file.unlink()
                
                # Clear index
                self.cache_index = {}
                self._save_cache_index()
                
                logger.info("Cache cleared")
                
            except Exception as e:
                logger.error(f"Could not clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.cache_lock:
            total_entries = len(self.cache_index)
            total_size = sum(info.get('cache_file_size', 0) for info in self.cache_index.values())
            
            # Calculate age distribution
            current_time = time.time()
            age_buckets = {'<1h': 0, '1h-24h': 0, '1d-7d': 0, '>7d': 0}
            
            for cache_info in self.cache_index.values():
                age_hours = (current_time - cache_info.get('cache_timestamp', 0)) / 3600
                
                if age_hours < 1:
                    age_buckets['<1h'] += 1
                elif age_hours < 24:
                    age_buckets['1h-24h'] += 1
                elif age_hours < 168:  # 7 days
                    age_buckets['1d-7d'] += 1
                else:
                    age_buckets['>7d'] += 1
            
            return {
                'total_entries': total_entries,
                'total_size_mb': total_size / (1024 * 1024),
                'max_size_mb': self.max_cache_size_bytes / (1024 * 1024),
                'cache_dir': str(self.cache_dir),
                'age_distribution': age_buckets,
                'usage_percent': (total_size / self.max_cache_size_bytes) * 100,
            }
    
    def invalidate_video(self, video_path: str) -> None:
        """Invalidate cache for specific video."""
        with self.cache_lock:
            cache_key = self._get_cache_key(video_path)
            if cache_key in self.cache_index:
                self._remove_cache_entry(cache_key)
                self._save_cache_index()
                logger.info(f"Invalidated cache for {Path(video_path).name}")
    
    def preload_cache(self, video_paths: List[str]) -> Dict[str, bool]:
        """Check cache status for multiple videos."""
        results = {}
        with self.cache_lock:
            for video_path in video_paths:
                results[video_path] = self.is_cached(video_path)
        return results