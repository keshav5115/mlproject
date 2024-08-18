import os
import sys 
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler



class DataTransformationconfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationconfig()

    def get_data_transformer_object(self):
        '''
        function is responsible for data transformation
        
        '''
        try:
            numerical_features = ['writing_score','reading_score']
            categorical_features = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course',
                ]
            num_pipeline=Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy='median')),
                ('scaler',StandardScaler(with_mean=False)),
                ]
            )
            logging.info('numerical_features scaling completed')
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False)),
                ]
            )
            logging.info('categorical features encoding completed')
            preprocessors = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_features),
                    ('cat_pipeline',cat_pipeline,categorical_features)
                ]
            )
            logging.info('preprocessing completed')
            return preprocessors
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Reading train and test data is completed')

            logging.info('obtaining preprocessing object')

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = 'math_score'
            numerical_features = ['writing_score','reading_score']

            input_features_train_df=train_df.drop(target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_features_test_df=test_df.drop(target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('applying preprocessing object on training datframes and testing datframes')

            input_feature_train_arr=preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_features_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
                         ]
            
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
                         ]
            
            logging.info('Saved preprocessing objects')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)
            
