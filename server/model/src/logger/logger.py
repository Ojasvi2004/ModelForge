import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from pathlib import Path
import os


try:
    from from_root import from_root
except ImportError:
    def from_root()->str:
        return str(Path(
            __file__
        ).resolve().parents[4])
    
LOGS_DIR='logs'
LOG_FILE=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
MAX_LOG_SIZE=5*1024*1024
BACKUP_COUNT=3

log_dir_file=os.path.join(from_root(),LOGS_DIR)
os.makedirs(log_dir_file,exist_ok=True)
log_file_path=os.path.join(log_dir_file,LOG_FILE)

print("Root:", from_root())
print("Log directory:", log_dir_file)
print("Log file:", log_file_path)

def configure_logger():
    """
    Configurations of logger
    """
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    
    formatter=logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    
    
    file_handler=RotatingFileHandler(log_file_path,maxBytes=MAX_LOG_SIZE,backupCount=BACKUP_COUNT)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)
    print(file_handler.baseFilename)
    
    console_handler=logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)
    
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    
configure_logger()
for h in logging.getLogger().handlers:
    print(type(h), getattr(h, "baseFilename", None))

