import os
import sys
from sklearn.base import TransformerMixin
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class BMI_prediction_DataTransformationConfig:

    scaler_obj_file_path = os.path.join('artifacts','bmi_scaler.pkl')

class BMI_prediction_DataTransformation:
    def __init__(self):
        self.bmi_data_transformation_config = BMI_prediction_DataTransformationConfig()


    def scaler(self,trainig_data,testing_data):
        try:
            logging.info("Scaling of BMI data started")
            bmi_robust_scaler = RobustScaler()
            scaled_training_data_aray  = bmi_robust_scaler.fit_transform(trainig_data)
            logging.info(f"Scaling done for training dataset:\n{scaled_training_data_aray}")
            scaled_testing_data_array = bmi_robust_scaler.transform(testing_data)
            logging.info(f"Scaling done for testing dataset:\n{scaled_testing_data_array}")
            logging.info("Scaling of bmi data completed")

            save_object(
                file_path=self.bmi_data_transformation_config.scaler_obj_file_path,
                obj=bmi_robust_scaler
                       )
            logging.info("Saved the bmi scaling object as .pkl file")
        except Exception as e:
            logging.info("Exception occured while scaling the data")
            raise CustomException(e,sys)
        return scaled_training_data_aray,scaled_testing_data_array,self.bmi_data_transformation_config.scaler_obj_file_path
    
    def initiate_data_transformation_bmi_data_and_get_transformer_objects(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Sucessfully read the training and testing data for bmi prediction from facial features')
            logging.info(f'Train Dataframe Head : \n{train_df.head()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head()}')
            logging.info("Preprocesing starts")

            target_column_name = 'BMI'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            logging.info(f"Sucessfully got input features for training datase\n{input_feature_train_df.head()}")
            target_feature_train_df = train_df[target_column_name]
            logging.info(f"Sucessfully got target features for training dataset\n{target_feature_train_df.head()}")

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            logging.info(f"Sucessfully got input features for testing dataset\n{input_feature_test_df.head()}")
            target_feature_test_df = test_df[target_column_name]
            logging.info(f"Sucessfully got target features for testing dataset\n{target_feature_test_df.head()}")

            input_feature_train_arr,input_feature_test_arr,scaler_obj_path = self.scaler(input_feature_train_df,input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            logging.info(f"We have training array:\{train_arr}")
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"We have testing array:\{test_arr}")

            return(
                train_arr,
                test_arr,
                scaler_obj_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the datatransformation of height and gender to weight prediction")

            raise CustomException(e,sys)