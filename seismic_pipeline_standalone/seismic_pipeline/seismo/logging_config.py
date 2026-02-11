"""
Centralized logging configuration for the mod package.

This module provides a standardized logging setup for all transformers and utilities
in the mod package, ensuring consistent log formatting and file output.
"""

import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
from logging.handlers import RotatingFileHandler


class SafeRotatingFileHandler(RotatingFileHandler):
    """
    A RotatingFileHandler that handles concurrent access safely.
    
    This handler catches and suppresses errors that occur during log rotation
    when multiple processes try to rotate the same file simultaneously.
    """
    
    def doRollover(self):
        """
        Do a rollover, as described in __init__().
        
        This version catches FileNotFoundError that can occur in multi-process
        scenarios where multiple processes try to rotate files simultaneously.
        """
        try:
            super().doRollover()
        except (FileNotFoundError, OSError) as e:
            # In a multi-process environment, another process may have already
            # rotated the file. This is expected and can be safely ignored.
            pass
    
    def emit(self, record):
        """
        Emit a record with safe error handling.
        
        This version catches errors during emission that might occur in
        multi-process scenarios.
        """
        try:
            super().emit(record)
        except (FileNotFoundError, OSError):
            # If file operations fail (e.g., during rotation), try to recreate
            # the stream and emit again
            try:
                self.close()
                self.stream = self._open()
                super(RotatingFileHandler, self).emit(record)
            except Exception:
                # If it still fails, give up silently to avoid breaking the application
                pass


class ModLoggerYt:
    """
    Centralized logger for the mod package.
    
    This class provides a standardized logging interface for all modules in the
    mod package, ensuring consistent formatting and file output.
    """
    
    _loggers = {}
    _log_dir = None
    _initialized = False
    
    @classmethod
    def setup_logging(cls, 
                     log_dir: str = './logs',
                     log_level: str = 'INFO',
                     log_format: Optional[str] = None,
                     max_file_size: int = 10 * 1024 * 1024,  # 10MB
                     backup_count: int = 5):
        """
        Set up centralized logging for the mod package.
        
        Parameters
        ----------
        log_dir : str, default='./logs'
            Directory for log files
        log_level : str, default='INFO'
            Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format : str, optional
            Custom log format string
        max_file_size : int, default=10485760
            Maximum log file size in bytes (10MB)
        backup_count : int, default=5
            Number of backup log files to keep
        """
        if cls._initialized:
            return
            
        # Create log directory
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up log format
        if log_format is None:
            log_format = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(filename)s:%(lineno)d - %(funcName)s() - %(message)s'
            )
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create file handler with rotation (using a custom safe handler)
        log_file = cls._log_dir / f'mod_package_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = SafeRotatingFileHandler(
            log_file, 
            maxBytes=max_file_size, 
            backupCount=backup_count
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Create console handler for critical errors only
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.ERROR)
        console_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        
        # Add handlers to root logger
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Set up specific loggers for different modules
        cls._setup_module_loggers()
        
        cls._initialized = True
        
        # Log initialization
        logger = cls.get_logger('mod.logging_config')
        logger.info(f"Logging initialized - Level: {log_level}, Directory: {log_dir}")
    
    @classmethod
    def _setup_module_loggers(cls):
        """Set up specific loggers for different modules."""
        modules = [
            'mod.cache_management',
            'mod.cross_validationyt',
            'mod.decompositionyt',
            'mod.event_transformers',
            'mod.grid_searchyt',
            'mod.hypnogram_cache_manager',
            'mod.hypnogram_transformers',
            'mod.metadata_adder',
            'mod.pcayt',
            'mod.pipelineyt',
            'mod.preprocessingyt',
            'mod.pyriemannyt',
            'mod.regressionyt',
            'mod.rem_profile_calculator',
            'mod.scaleryt',
            'mod.scoreryt',
            'mod.seismic_event_pipeline_v2',
            'mod.seismic_pipeline',
            'mod.selectionyt',
            'mod.sklearnbaseyt'
        ]
        
        for module in modules:
            logger = logging.getLogger(module)
            logger.setLevel(logging.DEBUG)
            # Don't add handlers here - they inherit from root logger
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """
        Get a logger for a specific module.
        
        Parameters
        ----------
        name : str
            Logger name (typically module name)
            
        Returns
        -------
        logging.Logger
            Configured logger instance
        """
        if not cls._initialized:
            cls.setup_logging()
        
        if name not in cls._loggers:
            cls._loggers[name] = logging.getLogger(name)
        
        return cls._loggers[name]
    
    @classmethod
    def get_log_file_path(cls) -> Optional[Path]:
        """
        Get the current log file path.
        
        Returns
        -------
        Path or None
            Path to current log file, or None if not initialized
        """
        if cls._log_dir is None:
            return None
        return cls._log_dir / f'mod_package_{datetime.now().strftime("%Y%m%d")}.log'
    
    @classmethod
    def cleanup_old_logs(cls, days_to_keep: int = 30):
        """
        Clean up old log files.
        
        Parameters
        ----------
        days_to_keep : int, default=30
            Number of days of logs to keep
        """
        if cls._log_dir is None or not cls._log_dir.exists():
            return
        
        from datetime import timedelta
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        for log_file in cls._log_dir.glob('mod_package_*.log*'):
            try:
                file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_time < cutoff_date:
                    log_file.unlink()
                    logger = cls.get_logger('mod.logging_config')
                    logger.info(f"Cleaned up old log file: {log_file}")
            except Exception as e:
                logger = cls.get_logger('mod.logging_config')
                logger.warning(f"Failed to clean up log file {log_file}: {e}")


def get_mod_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a mod logger.
    
    Parameters
    ----------
    name : str
        Logger name (typically module name)
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    return ModLoggerYt.get_logger(name)


# Initialize logging when module is imported
ModLoggerYt.setup_logging()
