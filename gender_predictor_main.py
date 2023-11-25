from src.components.image_and_height_data_scrapping_1 import Image_and_Height_Data_Scrapping
from src.components.facial_feature_extraction import Facial_Feature_Extraction_From_Images
from src.components.Image_to_gender_prediction import Facial_Images_to_Gender_Prediction

scrapped_images_path = "C:\\Users\\HP\\Desktop\\PW_Projects\\Body_Mass_Index_from_Face_Images\\images_data"
gender_prediction = Facial_Images_to_Gender_Prediction()
image_name_gender = gender_prediction.gender_predictor(scrapped_images_path)
img_name_and_height_data_path = "C:\\Users\\HP\\Desktop\\PW_Projects\\Body_Mass_Index_from_Face_Images\\artifacts\\img_name_and_height.csv"
path = gender_prediction.getting_dataset_img_name_height_gender(img_name_and_height_data_path,image_name_gender)
print(path)