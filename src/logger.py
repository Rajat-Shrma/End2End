import logging
import os
from datetime import datetime


log_path=os.path.join(os.getcwd(),"logs")
os.makedirs(log_path,exist_ok=True)

LOG_FILE_NAME=f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH=os.path.join(log_path,LOG_FILE_NAME)
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="%(asctime)s - [%(levelname)s] - %(filename)s:%(lineno)d - %(message)s",
    level=logging.INFO
)
