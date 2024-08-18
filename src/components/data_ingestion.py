import os
print(os.getcwd())
import sys 
# print(os.path.abspath('E:\\mlproject\\src'))
# sys.path.append(os.path.abspath('E:\\mlproject\\src'))
# import src
# from ..exception import CustomException
# from ..logger import logging
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from src.components.data_transformation import DataTransformation, DataTransformationconfig
from src.components.model_trainer import ModelTrainer,ModelTrainerconfig



from sklearn.model_selection import train_test_split
from dataclasses import dataclass



class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info('Starting data ingestion process')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('read the data set as dataframe')

            # os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            os.makedirs('artifacts',exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Train test split initiated')

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=32)

            train_set.to_csv(self.ingestion_config.train_data_path)

            test_set.to_csv(self.ingestion_config.test_data_path)

            logging.info('ingestion of data completed')
            return (
            self.ingestion_config.raw_data_path,
            self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__ == '__main__':
    obj = DataIngestion()
    train_data,test_data,file_path=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))
    
