import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class Height_and_Gender_to_Weight_DataIngestion_Config:
    hgw_train_data_path=os.path.join('artifacts','hgw_train.csv')
    hgw_test_data_path=os.path.join('artifacts','hgw_test.csv')
    hgw_raw_data_path=os.path.join('artifacts','hgw_raw.csv')



class HandG_to_W_DataIngestion:
    def __init__(self):
        self.handg_to_w_dataingestion_config = Height_and_Gender_to_Weight_DataIngestion_Config()

    def initiate_handg_to_w_data_ingestion(self,):
        logging.info('Data ingestion started for predicton of weight from height and gender')

        try:
            df=pd.read_csv(os.path.join('notebooks\Height_to_Weight_Data_and_Models','HWG_Data_Updated.csv'))
            logging.info(f'Dataset read as pandas Dataframe\n{df.head()}')

            os.makedirs(os.path.dirname(self.handg_to_w_dataingestion_config.hgw_raw_data_path),exist_ok=True)

            df.to_csv(self.handg_to_w_dataingestion_config.hgw_raw_data_path,index=False)
            logging.info("raw data has been saved to artifacts folder")

            logging.info("Splitting the data as train and test data")
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=42)

            train_set.to_csv(self.handg_to_w_dataingestion_config.hgw_train_data_path,index=False,header=True)
            test_set.to_csv(self.handg_to_w_dataingestion_config.hgw_test_data_path,index=False,header=True)

            logging.info('Data ingestion copleted for predicton of weight from height and gender')

            return(
                self.handg_to_w_dataingestion_config.hgw_train_data_path,
                self.handg_to_w_dataingestion_config.hgw_test_data_path

            )



        except Exception as e:
            logging.info('Error occured in Data Ingestion config')







