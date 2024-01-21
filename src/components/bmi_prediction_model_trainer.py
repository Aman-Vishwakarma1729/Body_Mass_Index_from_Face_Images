import os
import sys
from src.utils import save_object
from src.utils import train_and_evaluate_model
from sklearn.model_selection import KFold
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


@dataclass 
class BMI_ModelTrainerConfig:
    bmi_trained_model_file_path = os.path.join('artifacts','bmi_predictor_model.pkl')

class BMI_ModelTrainer:
    def __init__(self):
        self.bmi_model_trainer_config = BMI_ModelTrainerConfig()
    
    def initate_bmi_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test BMI data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            random_forest_regressor_with_cv = RandomForestRegressor()
            param_grid = {
                           'n_estimators': [50, 100, 200],
                           'max_depth': [None, 10, 20, 30],
                         }
            

            models={
                     "random_forest_regressor_with_cv": GridSearchCV(random_forest_regressor_with_cv,param_grid, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1)    
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
                 file_path=  self.bmi_model_trainer_config.bmi_trained_model_file_path,
                 obj=best_model
            )
        except Exception as e:
            logging.info('Exception occured at Model Training')
            raise CustomException(e,sys)
    
        return self.bmi_model_trainer_config.bmi_trained_model_file_path