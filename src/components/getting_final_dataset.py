import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd
from sklearn.preprocessing import RobustScaler
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

@dataclass
class FinalDatasetConfig:
    final_dataset_path = os.path.join('artifacts','final_bmi_dataset.csv')

class FinalDataset:
    def __init__(self):
        self.final_dataset_config = FinalDatasetConfig()

    def object_loader(self):
        try:
            logging.info("Loading objects required for getting final dataset")
            hgw_gender_encoder_path = os.path.join('artifacts','hgw_gender_encoder.pkl')
            hg_to_w_predictor_path = os.path.join('artifacts','hg_to_w_predictor_model.pkl')

            hgw_gender_encoder = load_object(hgw_gender_encoder_path)
            hg_to_w_predictor = load_object(hg_to_w_predictor_path)

        except Exception as e:
            logging.info("Exception occured while getting the objects required to get final dataset")
            raise CustomException(e,sys)
        
        return hgw_gender_encoder,hg_to_w_predictor
    
    def predicting_weight_from_height_and_gender(self):
        try:
            gw_gender_encoder,hg_to_w_predictor = self.object_loader()
            logging.info("Starting prediction of weights from height and gender")
            img_name_height_gender = os.path.join('artifacts','img_name_height_gender.csv')
            data1 = pd.read_csv(img_name_height_gender)
            logging.info(f"Successfully read the dataset\n{data1.head()}")
            logging.info('Changing the column name of dataset')
            data1 = data1.rename(columns={'gender':'Gender','height':'Height'})
            logging.info(f"Dataset after renaming the name of columns\n{data1.head()}")
            logging.info("Conveting the height form centimeters to meters")
            data1["Height"] = [(h/100) for h in data1["Height"]]
            logging.info("Converted the height data from centimeter to meter")
            logging.info("Started encoding gender column")
            data1["Gender"] = gw_gender_encoder.transform(data1["Gender"])
            logging.info(f"Encoding of gender column is completed\n{data1.head()}")
            data = data1.copy()
            logging.info("Scaling of data has started")
            hgw_scaler = RobustScaler()
            data["Height"] = hgw_scaler.fit_transform(data[["Height"]])
            data["Gender"] = hgw_scaler.fit_transform(data[["Gender"]])
            logging.info(f"Scaling of data has been done {data.head()}")
            logging.info("Prediction of weight has been started")
            Weight = hg_to_w_predictor.predict(data[['Gender','Height']])
            data["Weight"] = Weight
            logging.info(f"We have predicted weight\n{data.head()}")
            logging.info("Getting the data in original form")
            data1["Weight"] = [w for w in data['Weight']]
            logging.info(f"After we have\n{data1.head()}")
        except Exception as e:
            logging.info("Exception occured while predicting weight from height and gender")
            raise CustomException(e,sys)
        return data1
    
    def BMI_calculation(self):
        try:
            logging.info("Calculating BMI for each data points")
            data = self.predicting_weight_from_height_and_gender()
            data["BMI"] = data["Weight"]/data['Height']**2
            logging.info(f"BMI has been caculated\n{data.head()}")
        except Exception as e:
            logging.info("Exception occured while calculating BMI")
            raise CustomException(e,sys)
        return data
    
    def Final_dataset(self):
        try:
            logging.info("Getting final dataset")
            BMI_data = self.BMI_calculation()
            BMI_data = BMI_data.rename(columns={'image_name':'Image_name'})
            facial_features_data_path = os.path.join('artifacts','image_facial_features.csv')
            facial_deature_data = pd.read_csv(facial_features_data_path)
            facial_deature_data = facial_deature_data.drop(index=0)
            facial_deature_data = facial_deature_data.reset_index(drop=True)
            final_data = facial_deature_data.merge(BMI_data, on='Image_name', how='inner')
            final_data = final_data.drop(columns='Image_name')
            final_data = final_data.drop(columns='Height')
            final_data = final_data.drop(columns='Weight')
            logging.info(f"Finally we have dataset with facial features and coressponding BMI's\n{final_data.head()}")
            final_data.to_csv(self.final_dataset_config.final_dataset_path,index=False,header=True)
        except Exception as e:
            logging.info("Exception occured while getting final dataset")
            raise CustomException(e,sys)
        return self.final_dataset_config.final_dataset_path