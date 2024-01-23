from src.components.height_and_gender_to_weight_data_ingestion import HandG_to_W_DataIngestion
from src.components.height_and_gender_to_weight_data_transformation import HandG_to_W_DataTransformation
from src.components.height_and_gender_to_weight_model_trainer import HWG_ModelTrainer

data_ingector = HandG_to_W_DataIngestion()
hgw_train_path,hgw_test_path = data_ingector.initiate_handg_to_w_data_ingestion()

data_transformer = HandG_to_W_DataTransformation()
hgw_train_array,hgw_test_array,encoder_obj_path,scaler_obj_path = data_transformer.initiate_data_transformation_hgw_data_and_get_transformer_objects(hgw_train_path,hgw_test_path)
print(encoder_obj_path)
print(scaler_obj_path)
hwg_model_trainer = HWG_ModelTrainer()
hwg_model_trainer.initate_hwg_model_training(hgw_train_array,hgw_test_array)

print("Done")


    




