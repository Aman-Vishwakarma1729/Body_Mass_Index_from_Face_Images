from src.components.image_and_height_data_scrapping_1 import Image_and_Height_Data_Scrapping
from src.components.facial_feature_extraction import Facial_Feature_Extraction_From_Images
from src.components.Image_to_gender_prediction import Facial_Images_to_Gender_Prediction

if __name__=='__main__':

    initialize_scrapping = Image_and_Height_Data_Scrapping()
    pg_links_list = initialize_scrapping.get_urls("http://xn-----6kcczalffeh6afgdgdi2apgjghic4org.xn--p1ai/")
    initialize_scrapping.create_req_folders()
    img_name_and_height_dict = initialize_scrapping.get_image_and_height_from_urls(pg_links_list)
    scrapped_images_path,img_name_and_height_dataframe_path = initialize_scrapping.get_imgname_and_height_dataframe(img_name_and_height_dict)
    cropped_images_path = initialize_scrapping.crop(scrapped_images_path)
    initialize_scrapping.remove_folder_images_data()


    gender_prediction = Facial_Images_to_Gender_Prediction()
    image_name_gender = gender_prediction.gender_predictor(cropped_images_path)
    gender_prediction.getting_dataset_img_name_height_gender(img_name_and_height_dataframe_path,image_name_gender)


    feature_exctractor = Facial_Feature_Extraction_From_Images()
    img_name_list = feature_exctractor.load_images_in_directory(cropped_images_path)
    image_name_and_land_marks = feature_exctractor.get_facial_landmarks(img_name_list,cropped_images_path)
    facial_features = feature_exctractor.facial_feature_extraction(image_name_and_land_marks)
    feature_exctractor.facial_features_dataframe(facial_features)

    print("Done")