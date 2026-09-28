"""Centralized Logging Setup for the Indian Stock Prediction Platform."""
import logging
import sys
from pathlib import Path
from config.settings import LOGS_DIR, AppConfig

_configured = False

def get_logger(name: str = "indian_stock_pred") -> logging.Logger:
    """Return a configured logger with console and file handlers."""
    global _configured
    logger = logging.getLogger(name)
    
    if not _configured:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOGS_DIR / "app.log"
        
        cfg = AppConfig()
        level_name = getattr(logging, cfg.log_level.upper(), logging.INFO)
        logger.setLevel(level_name)
        
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # File Handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level_name)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        # Console Handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level_name)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        _configured = True
        
    return logger
