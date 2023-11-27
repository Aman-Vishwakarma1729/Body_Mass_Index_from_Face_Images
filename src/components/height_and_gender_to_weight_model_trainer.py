import os
import sys
from src.utils import save_object
from src.utils import train_and_evaluate_model
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


@dataclass 
class HWG_ModelTrainerConfig:
    hg_to_w_trained_model_file_path = os.path.join('artifacts','hg_to_w_predictor_model.pkl')

class HWG_ModelTrainer:
    def __init__(self):
        self.hwg_model_trainer_config = HWG_ModelTrainerConfig()
    
    def initate_hwg_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            ridge_with_cv = Ridge()
            param_grid = {'alpha': [0.05,0.07,0.09,0.1]}
            

            models={
            'LinearRegression':LinearRegression(),
            'Ridge':Ridge(),
            'ridge_with_cv' : GridSearchCV(ridge_with_cv, param_grid, cv=5, scoring='neg_mean_squared_error')
             }
            
            model_report = train_and_evaluate_model(X_train,y_train,X_test,y_test,models)
            print(model_report)
            print('\n====================================================================================\n')
            logging.info(f'Model Report : {model_report}')
            
            r2_scores = []
            for key,value in model_report.items():
                for met_name,score in value.items():  
                    if met_name == "r2":
                        r2_scores.append(score)
            max_r2 = max(r2_scores)

            for key,value in model_report.items():
                for met_name,score in value.items():  
                    if score == max_r2:
                        best_model_name = key

            best_model = models[best_model_name]
            print(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {max_r2}')
            print('\n====================================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {max_r2}')

            save_object(
                 file_path= self.hwg_model_trainer_config.hg_to_w_trained_model_file_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
    
        return self.hwg_model_trainer_config.hg_to_w_trained_model_file_path