# System imports
import os
# Third-party imports
import logging
from datetime import datetime


def setup_logger(log_dir, log_file_name):
    """
    Setup a logger to log events to a file
    Args:
        log_dir: Directory where the log file will be stored
        log_file_name: Name of the log file
    Returns:
        logger: Logger object
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    log_file = os.path.join(log_dir, log_file_name)
    handler = logging.FileHandler(log_file, mode='a')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def log_event(event):
    """
    Log an event to the log file
    Args:
        event: Event to log
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f'{{"event": "{event}", "timestamp": "{timestamp}"}}'
    print(message)
    logging.info(message)
