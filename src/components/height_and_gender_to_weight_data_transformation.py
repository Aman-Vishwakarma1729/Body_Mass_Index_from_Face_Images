import os
import sys
from sklearn.base import TransformerMixin 
from sklearn.preprocessing import LabelEncoder
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
class Height_and_Gender_to_Weight_DataTransformationConfig:
    encoder_obj_file_path=os.path.join('artifacts','hgw_gender_encoder.pkl')
    scaler_obj_file_path = os.path.join('artifacts','hgw_scaler.pkl')

class HandG_to_W_DataTransformation:
    def __init__(self):
        self.handg_to_w_data_transformation_config = Height_and_Gender_to_Weight_DataTransformationConfig()

    def outliers_remover(self,data):

        try:
            logging.info(f"Handling outliers started for\n{data.head(3)}")
            logging.info(f"Before removing outliers the shape of dataset is {data.shape}")
            logging.info("Getting numerical columns")
            num_col = ['Height','Weight']
            for col in num_col:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - (IQR*1.5)
                upper_bound = Q3 + (IQR*1.5)
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                data = data[~outliers]
                data = data.reset_index(drop=True)

        except Exception as e:
            logging.info("Exception occured while handling outiers")
            raise CustomException(e,sys)
        return data
    
    def encoder(self,trainig_data,testing_data):
        try:
            logging.info("Getting Categorica columns")
            cat_col = "Gender"
            logging.info(f"Encoding started for {cat_col} column")
            hwg_gender_encoder = LabelEncoder()
            gender_encoded_taining_data = hwg_gender_encoder.fit_transform(trainig_data[cat_col])
            logging.info(f"Encoding done for traning data set")
            trainig_data[cat_col] = gender_encoded_taining_data
            logging.info(f"After encoding the trainig data\n{trainig_data.head()}")
            gender_encoded_testing_data = hwg_gender_encoder.transform(testing_data[cat_col])
            logging.info("Encoding done for testing data set")
            testing_data[cat_col] = gender_encoded_testing_data
            logging.info(f"After encoding the testing data\n{testing_data.head()}")
            logging.info(f"Encoding completed for {cat_col} column")
            
            save_object(
                file_path=self.handg_to_w_data_transformation_config.encoder_obj_file_path,
                obj=hwg_gender_encoder
                       )
            logging.info("Saved the encoding object as .pkl file")
        except Exception as e:
            logging.info("Exception occured while encoding the data")
            raise CustomException(e,sys)
        return trainig_data,testing_data,self.handg_to_w_data_transformation_config.encoder_obj_file_path
    
    def scaler(self,trainig_data,testing_data):
        try:
            logging.info("Scaling of data started")
            hwg_robust_scaler = RobustScaler()
            scaled_training_data_aray  = hwg_robust_scaler.fit_transform(trainig_data)
            logging.info(f"Scaling done for training dataset:\n{scaled_training_data_aray}")
            scaled_testing_data_array = hwg_robust_scaler.transform(testing_data)
            logging.info(f"Scaling done for testing dataset:\n{scaled_testing_data_array}")
            logging.info("Scaling of data completed")

            save_object(
                file_path=self.handg_to_w_data_transformation_config.scaler_obj_file_path,
                obj=hwg_robust_scaler
                       )
            logging.info("Saved the scaling object as .pkl file")
        except Exception as e:
            logging.info("Exception occured while scaling the data")
            raise CustomException(e,sys)
        return scaled_training_data_aray,scaled_testing_data_array,self.handg_to_w_data_transformation_config.scaler_obj_file_path
    
    def initiate_data_transformation_hgw_data_and_get_transformer_objects(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            train_df = train_df.drop(columns="Unnamed: 0")
            test_df = pd.read_csv(test_path)
            test_df = test_df.drop(columns="Unnamed: 0")
            logging.info('Sucessfully read the training and testing data for weight prediction from height and gender')
            logging.info(f'Train Dataframe Head : \n{train_df.head()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head()}')
            logging.info("Preprocesing starts")
            train_df = self.outliers_remover(train_df)
            logging.info(f"After removing outliers the shape of dataset is {train_df.shape}")
            test_df = self.outliers_remover(test_df)
            logging.info(f"After removing outliers the shape of dataset is {test_df.shape}")

            target_column_name = 'Weight'
            drop_columns = [target_column_name]

            input_feature_train_df = train_df.drop(columns=drop_columns,axis=1)
            logging.info(f"Sucessfully got input features for training datase\n{input_feature_train_df.head()}")
            target_feature_train_df = train_df[target_column_name]
            logging.info(f"Sucessfully got target features for training dataset\n{target_feature_train_df.head()}")

            input_feature_test_df = test_df.drop(columns=drop_columns,axis=1)
            logging.info(f"Sucessfully got input features for testing dataset\n{input_feature_test_df.head()}")
            target_feature_test_df = test_df[target_column_name]
            logging.info(f"Sucessfully got target features for testing dataset\n{target_feature_test_df.head()}")

            encoded_trainig_data,encoded_testing_data,encoder_obj_path = self.encoder(input_feature_train_df,input_feature_test_df)
            input_feature_train_arr,input_feature_test_arr,scaler_obj_path = self.scaler(encoded_trainig_data,encoded_testing_data)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            logging.info(f"We have training array:\{train_arr}")
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"We have testing array:\{test_arr}")

            return(
                train_arr,
                test_arr,
                encoder_obj_path,
                scaler_obj_path
            )
        
        except Exception as e:
            logging.info("Exception occured in the datatransformation of height and gender to weight prediction")

            raise CustomException(e,sys)