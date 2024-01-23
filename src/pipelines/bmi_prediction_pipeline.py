import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
from dataclasses import dataclass
import mediapipe as mp
from src.components.facial_feature_extraction import Facial_Feature_Extraction_From_Images
import pandas as pd
import cv2
import mediapipe as mp
from deepface import DeepFace

@dataclass
class BMI_Prediction_Pipeline:
    def __init__(self):
        pass
    
    def get_directory_path(self,):
        try:
            directory_path = os.path.join(os.getcwd(),"input_images")
            logging.info(f"Got acces to the directory that has input image for bmi prediction:\n{directory_path}")   
        except Exception as e:
            logging.info("Exception occured in getting input image for prediction")
            raise CustomException(e,sys)
        return directory_path
    

    def get_gender(self,image_name_gender):
        try:
            logging.info("Extracting gender for input image")
            gender = []
            if len(image_name_gender) == 0:
                gen = input("Enter the gender: Male or Female")
                gender.append(gen)
            else:
                for _,value in image_name_gender.items():
                    if value == "Man":
                       gender.append("Male")
                    else:
                       gender.append("Female")
            logging.info(f"The predicted gender for the input image is:\n{gender}")
        except Exception as e:
            logging.info("Exception occured in extracting gender for input image")
            raise CustomException(e,sys)
        return gender[-1]
    
    def facial_features_dataframe_for_input_image(self,facial_features):
        logging.info("Data frame creation for extracted features has been started")
        try:
            Image_name = []
            FL = []                                          ## Face Lenght
            FRHL = []                                        ## Fore Head Length
            ETL = []                                         ## Ear Tip Length
            MEL = []                                         ## Mid Ear Length
            BEL = []                                         ## Bottom Ear Length
            LIPL_L = []                                      ## LIP Line Length
            JL = []                                          ## Jaw Length
            FLL = []                                         ## Face Length Left
            FLR = []                                         ## Face Length Right
            DFL_L_R = []                                     ## Diagonal Face Length Left to Right
            DFL_R_L = []                                     ## Diagonal Face Length Right to Left
            NWL= []                                          ## Nose Width Length
            L_L = []                                         ## Lip Length                        
            L_W= []                                          ## Lip Width
            LEL = []                                         ## Left Eye Length                  
            LEW = []                                         ## LEft Eye Width
            REL = []                                         ## Right Eye Length
            REW = []                                         ## Right Eye Width
            FA = []                                          ## Facial Area
            for key,value in facial_features.items():
                Image_name.append(key)
                FL.append(value[0])                                       
                FRHL.append(value[1])                               
                ETL.append(value[2])                                    
                MEL.append(value[3])                                     
                BEL.append(value[4])                                    
                LIPL_L.append(value[5])                             
                JL.append(value[6])                               
                FLL.append(value[7])                                   
                FLR.append(value[8])                                   
                DFL_L_R.append(value[9])                                
                DFL_R_L.append(value[10])                                   
                NWL.append(value[11])                                    
                L_L.append(value[12])                                                     
                L_W.append(value[13])                                  
                LEL.append(value[14])                                      
                LEW.append(value[15])                                      
                REL.append(value[16])                                   
                REW.append(value[17])
                FA.append(value[18])
            df = pd.DataFrame({
                "Image_name" : Image_name,
                "FL" : FL,
                "FRHL" : FRHL,
                "ETL" : ETL,
                "MEL" : MEL,
                "BEL" : BEL,
                "LIPL_L" : LIPL_L,
                "JL" : JL,
                "FLL" : FLL,
                "FLR" : FLR,
                "DFL_L_R" : DFL_L_R,
                "DFL_R_L" : DFL_R_L,
                "NWL" : NWL,
                "L_L" : L_L,                       
                "L_W" : L_W,
                "LEL" : LEL,                 
                "LEW" : LEW,
                "REL" : REL,
                "REW" : REW,
                "FA" : FA
            })

            df = df.drop(columns=['Image_name'])
            df = df.drop(0)
            logging.info(f"The dataframe of faical feature is:\n{df}")
        except Exception as e:
            logging.info("An error occured while creating dataframe of facial features for input images")
            raise CustomException(e,sys)
       
        return df

    mpDraw = mp.solutions.drawing_utils 
    mpFaceMesh = mp.solutions.face_mesh
    FaceMesh = mpFaceMesh.FaceMesh()
    drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def get_facial_features(self,directory_path,gender):
        try:
            logging.info("Facial feature extraction for input image has been initiated")
            feature_exctractor = Facial_Feature_Extraction_From_Images()
            img_name_list = feature_exctractor.load_images_in_directory(directory_path)
            image_name_and_land_marks = feature_exctractor.get_facial_landmarks(img_name_list,directory_path)
            facial_features = feature_exctractor.facial_feature_extraction(image_name_and_land_marks)
            df = self.facial_features_dataframe_for_input_image(facial_features)
            df["Gender"] = [gender]
            logging.info(f"The extracted facial features from input images:\n{df}")
        except Exception as e:
            logging.info("Error occured while extracting facial features from input images")
            raise CustomException(e,sys)
        return df
    
    def scaling_encoding_input_data(self,data):
        try:
            logging.info("Loading scaler and encoder")
            hgw_gender_encoder_path = os.path.join('artifacts','hgw_gender_encoder.pkl')
            bmi_scaler_path = os.path.join('artifacts','bmi_scaler.pkl')

            hgw_gender_encoder = load_object(hgw_gender_encoder_path)
            bmi_data_scaler = load_object(bmi_scaler_path)
            logging.info(f"Encoder and Scaler models has been loaded\n{hgw_gender_encoder},{bmi_data_scaler}")

            data["Gender"] = hgw_gender_encoder.transform(data["Gender"])
            logging.info(f"The gender is encoded:\n{data}")

            scaled_data = bmi_data_scaler.transform(data)
            logging.info(f"The data is scaled:\n{scaled_data}")

        except Exception as e:
            logging.info("Error occured while scaling and encoding data for input images")
            raise CustomException(e,sys)
        return scaled_data

    def predict_bmi(self,scaled_data):
        try:
            logging.info(f"Prediction has been started for the obtained inputs:\n{scaled_data}")
            model_path=os.path.join('artifacts','bmi_predictor_model.pkl')
            
            model=load_object(model_path)
            logging.info(f"Got the predictor:\n{model}")

            predicted_bmi = model.predict(scaled_data)
            logging.info(f"The predcited bmi for given input is\n{predicted_bmi}")
            return predicted_bmi
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
    

    def clear_input_image(self,directory_path):
        try:
            for filename in os.listdir(directory_path):
                image_path = os.path.join(directory_path,filename)
                os.remove(image_path)
                logging.info("The input image taken for prediction has been removed from folder")
        except Exception as e:
            logging.info("There is an error in removing input image from folder")
            raise CustomException(e,sys)