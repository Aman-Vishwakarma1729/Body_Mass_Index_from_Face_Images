import os
import sys
import pickle
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        logging.info(f"An Exception occured while saving {obj}")
        raise CustomException(e, sys)

def train_and_evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        logging.info("Model training and evaluation has started")
        report = {}
        for i in range(len(models)):
            metrics = {}
            model = list(models.values())[i]
            logging.info(f"Training {model} model")
            model.fit(X_train,y_train)
            y_test_pred =model.predict(X_test)
            r2 = r2_score(y_test,y_test_pred)
            logging.info(f"Evaluating {model} model")
            metrics["r2"] = r2
            mse = mean_squared_error(y_test,y_test_pred)
            metrics["mse"] = mse
            mae = mean_absolute_error(y_test,y_test_pred)
            metrics["mae"] = mae
            rmse = mse**0.5
            metrics["rmse"] = rmse
            report[list(models.keys())[i]] = metrics
            logging.info(f"The metrics for {model} are {metrics}")
            metrics = {}
        logging.info(f"The final report is here\n{report}")
        return report  
    except Exception as e:
            logging.info('Exception occured during model training and evaluation')
            raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)