import os
import sys

import numpy as np

import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

def save_object(file_path,obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        
        with open(file_path,'wb') as f:
            dill.dump(obj,f)

        logging.info(f'The preprocessor is saved to location: {file_path}')
    except Exception as e:
        raise CustomException(e,sys)