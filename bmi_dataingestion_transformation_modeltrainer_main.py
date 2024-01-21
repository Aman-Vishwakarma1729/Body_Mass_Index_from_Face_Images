from src.components.bmi_data_ingestion import BMI_DataIngestion
from src.components.bmi_data_transformation import BMI_prediction_DataTransformation
from src.components.bmi_prediction_model_trainer import BMI_ModelTrainer

bmi_data_ingector = BMI_DataIngestion()
bmi_data_transformer = BMI_prediction_DataTransformation()
bmi_model_trainer = BMI_ModelTrainer()
bmi_train_path,bmi_test_path = bmi_data_ingector.initiate_bmi_data_ingestion()
train_arr,test_arr,scaler_obj_path = bmi_data_transformer.initiate_data_transformation_bmi_data_and_get_transformer_objects(bmi_train_path,bmi_test_path)
print(scaler_obj_path)
bmi_model_path = bmi_model_trainer.initate_bmi_model_training(train_arr,test_arr)
print(bmi_model_path)