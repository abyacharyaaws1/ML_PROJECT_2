import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
log_directory = "logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure logging
log_filename = os.path.join(log_directory, f"{datetime.now().strftime('%Y-%m-%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='a'),  # 'a' mode for appending
        logging.StreamHandler()
    ]
)

# Logger instance
logger = logging.getLogger(__name__)

