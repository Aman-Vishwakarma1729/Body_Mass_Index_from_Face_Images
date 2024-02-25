import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass

@dataclass
class BMI_DataIngestion_Config:
    bmi_train_data_path=os.path.join('artifacts','bmi_train.csv')
    bmi_test_data_path=os.path.join('artifacts','bmi_test.csv')
    bmi_raw_data_path=os.path.join('artifacts','bmi_raw.csv')



class BMI_DataIngestion:
    def __init__(self):
        self.bmi_dataingestion_config = BMI_DataIngestion_Config()
    
    def balancing_data(self,data):
        logging.info("Before data ingestion starts we need to balance the dataset")
        logging.info("Labeling started for balacing\nbmi<18.5 : Malnourished\nbmi>24 : Overweigt\nelse: Normal")
        category = []
        for bmi in data['BMI']:
            if bmi < 18.5:
               category.append('Malnourished')
            elif bmi > 24:
               category.append('Overweight')
            else:
               category.append('Normal')
        logging.info("Labeling comlpleted")
        data["BMI_category"] = category
        logging.info(f"Here is the overview before balacing:\n{data['BMI_category'].value_counts()}")
        logging.info("Encoding the 'BMI_category' column for SMOTE")
        bmi_category_encoder = LabelEncoder()
        data['BMI_category'] = bmi_category_encoder.fit_transform(data['BMI_category'])
        logging.info("Balacing started")
        X = data.drop(columns='BMI_category')
        y = data['BMI_category']
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X,y)
        X_resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled_df = pd.Series(y_resampled, name=y.name)
        df = pd.concat([X_resampled_df,y_resampled_df],axis=1)
        logging.info(f"Here is the overview after balacing:\n{df['BMI_category'].value_counts()}")
        df = df.drop(columns="BMI_category")
        return df
    
    def initiate_bmi_data_ingestion(self,):
        logging.info('Data ingestion started for predicton of bmi from facial features')

        try:
            df=pd.read_csv(os.path.join('notebooks\BMI_Data_Analysis_and_Modelling','final_bmi_dataset.csv'))
            logging.info(f'Dataset read as pandas Dataframe\n{df.head()}')

            os.makedirs(os.path.dirname(self.bmi_dataingestion_config.bmi_raw_data_path),exist_ok=True)
            
            df = self.balancing_data(df)

            df.to_csv(self.bmi_dataingestion_config.bmi_raw_data_path,index=False)
            logging.info("bmi raw data has been saved to artifacts folder")

            logging.info("Splitting the bmi data as train and test data")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.bmi_dataingestion_config.bmi_train_data_path,index=False,header=True)
            test_set.to_csv(self.bmi_dataingestion_config.bmi_test_data_path,index=False,header=True)

            logging.info('Data ingestion completed for predicton of bmi from facial features')

            return(
                self.bmi_dataingestion_config.bmi_train_data_path,
                self.bmi_dataingestion_config.bmi_test_data_path

            )



        except Exception as e:
            logging.info('Error occured in Data Ingestion config')