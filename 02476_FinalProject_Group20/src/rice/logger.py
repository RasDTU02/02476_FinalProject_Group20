# logger.py
import logging
import os
from datetime import datetime

# Define log directory
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Generate timestamped log filename
log_filename = os.path.join(LOG_DIR, f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),  # Write logs to a file
        logging.StreamHandler()  # Print logs to console
    ]
)

# Function to get logger
def get_logger(name):
    return logging.getLogger(name)
