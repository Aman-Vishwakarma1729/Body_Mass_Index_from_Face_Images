import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import cv2
import mediapipe as mp
import json
import cv2
from deepface import DeepFace

@dataclass
class Facial_Images_to_Gender_Prediction_Config:
    predicted_gender_data = os.path.join('artifacts','img_name_height_gender.csv')

class Facial_Images_to_Gender_Prediction:
    def __init__(self):
        self.gender_predictor_config = Facial_Images_to_Gender_Prediction_Config()

    def gender_predictor(self,directory_path):
        logging.info("Gender prediction from facial images is initialized")
        image_name_gender = {}
        try:
            for filename in os.listdir(directory_path):
                logging.info(f"gender prediction started for {filename}")
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                   image_path = os.path.join(directory_path, filename)
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
            logging.info(f"Finally we have gender for all images with their names:\n{image_name_gender},{type(image_name_gender)}")
        except Exception as e:
            logging.info("There is error in predicting gender")
            raise CustomException(e,sys)
        
        return image_name_gender


    def getting_dataset_img_name_height_gender(self,img_name_and_height_data_path,image_name_gender):
        logging.info("Started creating dataset with image name, height and gender information")
        image_name = []
        gender = []
        try:
            for key,value in image_name_gender.items():
                image_name.append(key)
                if value == 'Man':
                    gender.append("Male")
                else:
                    gender.append("Female")
            logging.info(f"There are total {len(image_name)} images")
            logging.info(f"There are total {len(gender)} genders")
            df1 = pd.DataFrame({
                "image_name" : image_name,
                "gender" : gender
            })
            df2 = pd.read_csv(img_name_and_height_data_path)

            df = df1.merge(df2, on='image_name', how='inner')
            logging.info(f"The Dataset we get is\n{df}")
            df.to_csv(self.gender_predictor_config.predicted_gender_data,index=False,header=True)
         
        except Exception as e:
            logging.info("There is error in creating a dataframe with image name, height and gender")
            raise CustomException(e,sys)

        return self.gender_predictor_config.predicted_gender_data
    



     