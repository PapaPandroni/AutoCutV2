"""Structured logging configuration for AutoCut V2.

This module provides centralized logging configuration to replace the 200+ print()
statements throughout the codebase with structured, configurable logging.

Features:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- File and console output with automatic rotation
- Performance timing decorators
- Context-aware logging with extra fields
- Configurable formatters for different environments
"""

import functools
import logging
import logging.config
import logging.handlers
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar

# Type variables for generic decorator
F = TypeVar("F", bound=Callable[..., Any])

# Default logging configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "simple": {
            "format": "%(levelname)s | %(message)s",
        },
        "performance": {
            "format": "%(asctime)s | PERF | %(name)s | %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(name)s %(levelname)s %(funcName)s %(lineno)d %(message)s",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "autocut.log",
            "maxBytes": 10 * 1024 * 1024,  # 10MB
            "backupCount": 5,
            "level": "DEBUG",
            "formatter": "detailed",
            "encoding": "utf-8",
        },
        "performance": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "autocut_performance.log",
            "maxBytes": 5 * 1024 * 1024,  # 5MB
            "backupCount": 3,
            "level": "INFO",
            "formatter": "performance",
            "encoding": "utf-8",
        },
        "error": {
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "autocut_errors.log",
            "maxBytes": 2 * 1024 * 1024,  # 2MB
            "backupCount": 5,
            "level": "ERROR",
            "formatter": "detailed",
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "autocut": {
            "level": "DEBUG",
            "handlers": ["console", "file", "error"],
            "propagate": False,
        },
        "autocut.performance": {
            "level": "INFO",
            "handlers": ["performance"],
            "propagate": False,
        },
        "moviepy": {
            "level": "WARNING",
            "handlers": ["file"],
            "propagate": False,
        },
        "librosa": {
            "level": "WARNING",
            "handlers": ["file"],
            "propagate": False,
        },
    },
    "root": {
        "level": "INFO",
        "handlers": ["console", "file"],
    },
}


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    enable_performance_logging: bool = True,
) -> None:
    """Setup structured logging for AutoCut.

    Args:
        log_level: Root logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Custom log file path (optional)
        config: Custom logging configuration dict (optional)
        enable_performance_logging: Whether to enable performance timing logs

    Raises:
        ValueError: If log_level is invalid
        OSError: If log file cannot be created
    """
    # Validate log level
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. Must be one of {valid_levels}",
        )

    # Use custom config or default
    logging_config = config or LOGGING_CONFIG.copy()

    # Update log level
    logging_config["root"]["level"] = log_level.upper()
    logging_config["loggers"]["autocut"]["level"] = log_level.upper()

    # Update log file if provided
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging_config["handlers"]["file"]["filename"] = str(log_path)

        # Update related log files to same directory
        base_dir = log_path.parent
        logging_config["handlers"]["performance"]["filename"] = str(
            base_dir / "autocut_performance.log",
        )
        logging_config["handlers"]["error"]["filename"] = str(
            base_dir / "autocut_errors.log",
        )

    # Disable performance logging if requested
    if not enable_performance_logging:
        if "performance" in logging_config["handlers"]:
            del logging_config["handlers"]["performance"]
        if "autocut.performance" in logging_config["loggers"]:
            del logging_config["loggers"]["autocut.performance"]

    # Apply logging configuration
    try:
        logging.config.dictConfig(logging_config)

        # Log initial setup message
        logger = get_logger("autocut.setup")
        logger.info(
            "Logging configured successfully",
            extra={
                "log_level": log_level,
                "log_file": log_file or "autocut.log",
                "performance_logging": enable_performance_logging,
            },
        )

    except Exception as e:
        # Fallback to basic logging if configuration fails
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        )
        logging.exception(f"Failed to configure logging: {e}")
        raise


def get_logger(name: str) -> logging.Logger:
    """Get logger instance for module.

    Args:
        name: Logger name, typically __name__ of calling module

    Returns:
        Configured logger instance

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing started")
        >>> logger.error("Failed to process", extra={"file": "test.mp4"})
    """
    return logging.getLogger(name)


def log_performance(
    operation: Optional[str] = None,
    include_memory: bool = True,
    log_args: bool = False,
) -> Callable[[F], F]:
    """Decorator to log function execution time and optionally memory usage.

    Args:
        operation: Custom operation name (defaults to function name)
        include_memory: Whether to include memory usage statistics
        log_args: Whether to log function arguments (be careful with sensitive data)

    Returns:
        Decorator function

    Examples:
        >>> @log_performance()
        ... def process_video(video_path: str) -> None:
        ...     # Function implementation
        ...     pass

        >>> @log_performance("Custom Operation", include_memory=False)
        ... def quick_function() -> str:
        ...     return "result"
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get performance logger
            perf_logger = get_logger("autocut.performance")

            # Operation name
            op_name = operation or func.__name__

            # Start timing and memory monitoring
            start_time = time.time()
            start_memory = None
            memory_available = include_memory
            if memory_available:
                try:
                    import psutil

                    process = psutil.Process()
                    start_memory = process.memory_info().rss / 1024 / 1024  # MB
                except ImportError:
                    memory_available = False

            # Prepare log context
            log_context = {
                "operation": op_name,
                "function": func.__name__,
                "module": func.__module__,
            }

            # Add arguments if requested (be careful with sensitive data)
            if log_args and args:
                log_context["args_count"] = len(args)
            if log_args and kwargs:
                log_context["kwargs_keys"] = list(kwargs.keys())

            try:
                # Execute function
                result = func(*args, **kwargs)

                # Calculate execution time
                execution_time = time.time() - start_time

                # Calculate memory usage
                memory_info = {}
                if memory_available and start_memory is not None:
                    try:
                        end_memory = process.memory_info().rss / 1024 / 1024  # MB
                        memory_info = {
                            "memory_start_mb": round(start_memory, 2),
                            "memory_end_mb": round(end_memory, 2),
                            "memory_delta_mb": round(end_memory - start_memory, 2),
                        }
                    except Exception:
                        pass

                # Log successful execution
                log_context.update(
                    {
                        "status": "success",
                        "execution_time_s": round(execution_time, 3),
                        **memory_info,
                    },
                )

                perf_logger.info(f"{op_name} completed successfully", extra=log_context)

            except Exception as e:
                # Calculate execution time even for failures
                execution_time = time.time() - start_time

                # Log failed execution
                log_context.update(
                    {
                        "status": "error",
                        "execution_time_s": round(execution_time, 3),
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    },
                )

                perf_logger.exception(f"{op_name} failed", extra=log_context)

                # Re-raise the exception
                raise
            else:
                return result

        return wrapper

    return decorator


def log_memory_usage(operation: str) -> None:
    """Log current memory usage for debugging memory issues.

    Args:
        operation: Description of current operation for context
    """
    try:
        import psutil

        process = psutil.Process()
        memory_info = process.memory_info()

        logger = get_logger("autocut.memory")
        logger.debug(
            f"Memory usage during {operation}",
            extra={
                "operation": operation,
                "rss_mb": round(memory_info.rss / 1024 / 1024, 2),
                "vms_mb": round(memory_info.vms / 1024 / 1024, 2),
                "percent": psutil.virtual_memory().percent,
            },
        )
    except ImportError:
        # psutil not available
        pass
    except Exception as e:
        logger = get_logger("autocut.memory")
        logger.warning(f"Failed to log memory usage: {e}")


def log_system_info() -> None:
    """Log system information at startup for debugging."""
    import platform
    import sys

    logger = get_logger("autocut.system")

    try:
        import psutil

        memory_gb = round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2)
        cpu_count = psutil.cpu_count()
    except ImportError:
        memory_gb = "unknown"
        cpu_count = "unknown"

    logger.info(
        "AutoCut system information",
        extra={
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "architecture": platform.architecture()[0],
            "cpu_count": cpu_count,
            "total_memory_gb": memory_gb,
        },
    )


# Context manager for operation-level logging
class LoggingContext:
    """Context manager for operation-level logging with automatic cleanup."""

    def __init__(
        self,
        operation: str,
        logger: Optional[logging.Logger] = None,
        log_level: str = "INFO",
        **extra_context: Any,
    ):
        """Initialize logging context.

        Args:
            operation: Operation description
            logger: Logger instance (defaults to autocut logger)
            log_level: Log level for this operation
            **extra_context: Additional context fields
        """
        self.operation = operation
        self.logger = logger or get_logger("autocut")
        self.log_level = getattr(logging, log_level.upper())
        self.extra_context = extra_context
        self.start_time = None

    def __enter__(self) -> "LoggingContext":
        """Enter context and log operation start."""
        self.start_time = time.time()

        self.logger.log(
            self.log_level,
            f"Starting {self.operation}",
            extra={"operation": self.operation, **self.extra_context},
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context and log operation completion or failure."""
        execution_time = time.time() - self.start_time if self.start_time else 0

        context = {
            "operation": self.operation,
            "execution_time_s": round(execution_time, 3),
            **self.extra_context,
        }

        if exc_type is None:
            self.logger.log(
                self.log_level,
                f"Completed {self.operation}",
                extra=context,
            )
        else:
            context.update(
                {
                    "error_type": exc_type.__name__,
                    "error_message": str(exc_val),
                },
            )
            self.logger.error(
                f"Failed {self.operation}",
                extra=context,
            )

    def log(self, message: str, level: str = "INFO", **extra: Any) -> None:
        """Log a message within this operation context.

        Args:
            message: Log message
            level: Log level
            **extra: Additional context fields
        """
        log_level = getattr(logging, level.upper())
        context = {
            "operation": self.operation,
            **self.extra_context,
            **extra,
        }

        self.logger.log(log_level, message, extra=context)


# Common logging utilities for replacing print() statements
def replace_print_with_logging() -> None:
    """Helper to identify print statements that need replacement.

    This function can be used during development to identify print() usage.
    In production, all print() statements should be replaced with logging calls.
    """
    import builtins
    import traceback

    original_print = builtins.print

    def logging_print(*args, **kwargs):
        """Replacement print that logs usage for identification."""
        # Get caller information
        frame = traceback.extract_stack()[-2]

        logger = get_logger("autocut.print_replacement")
        logger.warning(
            "print() statement found - should be replaced with logging",
            extra={
                "filename": frame.filename,
                "line": frame.lineno,
                "function": frame.name,
                "print_args": str(args),
            },
        )

        # Still call original print for backward compatibility
        original_print(*args, **kwargs)

    builtins.print = logging_print
