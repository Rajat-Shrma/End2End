import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score

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
    

def evaluate_models(x_train,y_train,x_test,y_test,models):
    logging.info('Model Training Started.')
    model_report={}

    for model_name,model in models.items():
        model.fit(x_train,y_train)

        logging.info(f'{model_name} Trained')

        ypred_test=model.predict(x_test)

        model_r2_score_test=r2_score(y_pred=ypred_test,y_true=y_test)

        model_report[model_name]=model_r2_score_test

        logging.info(f'{model_name} Accuracy Reported.')
    return model_report
        


