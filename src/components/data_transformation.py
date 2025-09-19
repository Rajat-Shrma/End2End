import pandas as pd
import numpy as np
import os
import sys

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from dataclasses import dataclass

@dataclass
class DataCustomTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')




class DataCustomTransformation:
    def __init__(self):
        self.data_transformation_config=DataCustomTransformationConfig()

    def get_transformer_object(self):
        numberic_features=['reading_score', 'writing_score']
        categorical_features=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

        num_pipeline=Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('standard_scaler',StandardScaler())
            ]
        )

        cat_pipeline=Pipeline(
            steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()),
            ]
        )

        preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numberic_features),
            ('cat_pipeline',cat_pipeline,categorical_features)
        ])

        logging.info('Preprocessor is ready.')

        return preprocessor
        
        
    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df=pd.read_csv(train_data_path)
            test_df=pd.read_csv(test_data_path)

            logging.info('Reading of train & test data is completed.')

            TARGET_FEATURE="math_score"
            x_train=train_df.drop(TARGET_FEATURE,axis=1)
            y_train=train_df[TARGET_FEATURE]

            x_test=test_df.drop(TARGET_FEATURE,axis=1)
            y_test=test_df[TARGET_FEATURE]

            preprocessor_obj=self.get_transformer_object()
            x_train_arr=preprocessor_obj.fit_transform(x_train)
            x_test_arr=preprocessor_obj.transform(x_test)

            train_arr=np.c_[
                x_train_arr,np.array(y_train)
            ]

            test_arr=np.c_[
                x_test_arr,np.array(y_test)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )


            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
         