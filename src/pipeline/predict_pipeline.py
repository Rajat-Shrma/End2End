import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
from src.components.data_ingestion import DataCustomTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


MODEL_PATH=ModelTrainerConfig.trained_model_file_path
PREPROCESSOR_PATH=DataCustomTransformationConfig.preprocessor_obj_file_path
class PredictPipeline:
    def __init__(self):
        self.model=load_object(MODEL_PATH)
        self.preprocessor=load_object(PREPROCESSOR_PATH)

    def predict(self,features):
        try:
            input_data=self.preprocessor.transform(features)

            prediction=self.model.predict(input_data)

            return prediction
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomDataset():
    def __init__(self,gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,reading_score,writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    

    def get_dataframe(self):
        dataset={
            'gender':[self.gender],
            'race_ethnicity':[self.race_ethnicity],
            'parental_level_of_education':[self.parental_level_of_education],
            'lunch':[self.lunch],
            'test_preparation_course':[self.test_preparation_course],
            'reading_score':[self.reading_score],
            'writing_score':[self.writing_score]

        }

        dataframe=pd.DataFrame(dataset)

        return dataframe



    