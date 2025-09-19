import pandas as pd
import numpy as np
import sys
import os

from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str=os.path.join('artifacts','model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_training(self, train_arr,test_arr, preprocessor_path):
        try:
            logging.info('Splitting training and test input data.')

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models = {
                "LinearRegression": LinearRegression(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor()
            }

            models_report:dict=evaluate_models(x_train=x_train,y_train=y_train, x_test=x_test, y_test=y_test,models=models)

            best_accuracy=max(models_report.values())
            best_model_name=list(models_report.keys())[list(models_report.values()).index(best_accuracy)]

            best_model=models[best_model_name]

            if best_accuracy<0.6:
                raise CustomException('None of the  Model Accuracy cross the threshold.')
            
            logging.info('Best model Found!')
            
            best_model.fit(x_train,y_train)
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2score=r2_score(y_test,predicted)

            return r2score

        except Exception as e:
            raise CustomException(e,sys)
            



