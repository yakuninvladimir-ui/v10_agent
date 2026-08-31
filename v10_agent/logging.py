"""
Logging Module
"""
import logging
from typing import Any, Dict

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def log_audit_event(event_type: str, data: Dict[str, Any]):
    """Log structured audit events."""
    logger = get_logger("audit")
    logger.info(f"AUDIT_EVENT: {event_type} - {data}")
