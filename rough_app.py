import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import mediapipe as mp
import json
import cv2
from deepface import DeepFace

from src.pipelines.bmi_prediction_pipeline import BMI_Prediction_Pipeline

if __name__ == "__main__":
    BMI_Prediction_Pipeline().create_required_folders()
    directory_path = BMI_Prediction_Pipeline().get_directory_path()
    BMI_Prediction_Pipeline().capture_with_timer_and_save(directory_path)
    croped_image_for_prediction_directory_path = BMI_Prediction_Pipeline().crop(directory_path)

    def gender_predictor(croped_image_for_prediction_directory_path):
        logging.info("Gender prediction from input facial image is initialized")
        image_name_gender = {}
        try:
            for filename in os.listdir(croped_image_for_prediction_directory_path):
                logging.info(f"gender prediction started for {filename}")
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    image_path = os.path.join(croped_image_for_prediction_directory_path, filename)
                
                    image = cv2.imread(image_path,0)
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    for (x, y, w, h) in faces:
                        face = image[y:y + h, x:x + w]
                        result = DeepFace.analyze(face, enforce_detection=False, actions=['gender'])
                        gender = result[0]['gender']
                        max_gender = max(gender, key=lambda k: gender[k])
                        logging.info(f'The gender with the maximum count for {filename} is: {max_gender}')
                        image_name_gender[filename] = max_gender
            logging.info(f"Finally we have gender for input images with their name:\n{image_name_gender}")
        except Exception as e:
            logging.info("There is error in predicting gender")
            raise CustomException(e,sys)
            
        return image_name_gender

    image_name_gender = gender_predictor(croped_image_for_prediction_directory_path)
    gender = BMI_Prediction_Pipeline().get_gender(image_name_gender)
    
    df = BMI_Prediction_Pipeline().get_facial_features(croped_image_for_prediction_directory_path,gender)
    scaled_data = BMI_Prediction_Pipeline().scaling_encoding_input_data(df)
    predicted_bmi = BMI_Prediction_Pipeline().predict_bmi(scaled_data)
    print(predicted_bmi)
    BMI_Prediction_Pipeline().remove_folder_images_data()
    BMI_Prediction_Pipeline().clear_input_image(croped_image_for_prediction_directory_path)


