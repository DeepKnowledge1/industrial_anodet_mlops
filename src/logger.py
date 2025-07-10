import logging
import os

from logging.config import dictConfig
import yaml

def setup_logging(default_path='monitoring/log_config.yaml', default_level=logging.INFO):
    """Setup logging configuration"""
    path = default_path
    # Ensure logs directory exists
    log_dir = os.path.join(os.getcwd(), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = yaml.safe_load(f)
        dictConfig(config)
    else:
        logging.basicConfig(level=default_level)

logger = logging.getLogger("industrial-mlops")
