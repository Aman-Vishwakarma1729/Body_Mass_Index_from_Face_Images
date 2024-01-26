import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import cv2
import mediapipe as mp
import json


@dataclass
class Facial_Feature_Extraction_From_Images_Config:
    extracted_facial_features = os.path.join('artifacts','image_facial_features.csv')
    
class Facial_Feature_Extraction_From_Images:
    def __init__(self):
        self.feature_extrator_config = Facial_Feature_Extraction_From_Images_Config()

    def load_images_in_directory(self,directory_path):
        logging.info(f"loading images name from {directory_path}")
        try:
            image_list = []
            for filename in os.listdir(directory_path):
                image_list.append(filename)
            logging.info(f"The loaded image list is:\n{image_list}")
        except Exception as e:
            logging.info(f"There is error in getting images name from {directory_path}")
            raise CustomException(e,sys)
        return image_list

    logging.info("Ready with all libraries to extract facial features")
    mpDraw = mp.solutions.drawing_utils 
    mpFaceMesh = mp.solutions.face_mesh
    FaceMesh = mpFaceMesh.FaceMesh()
    drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)

    def get_facial_landmarks(self,image_list,scrapped_images_path):
        logging.info("Started with getting facial landmarks for scrapped images")
        try:
            mpDraw = mp.solutions.drawing_utils
            drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)
            image_name_and_land_marks = {}
            for img in image_list:
                logging.info(f"We are at {img} image")
                image = f"{scrapped_images_path}\\{img}"
                logging.info(f"Sucessfully accessed image {image}")
                image = cv2.imread(image)
                imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)                
                FaceMesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)    
                results = FaceMesh.process(imageRGB)                              
                if results.multi_face_landmarks:
                    for faceLandMarks in results.multi_face_landmarks:
                        mp.solutions.drawing_utils.draw_landmarks(image, faceLandMarks, mp.solutions.face_mesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
                        img_land_marks = []
                        for id, land_marks in enumerate(faceLandMarks.landmark):
                            land_mark_details = []
                            id = id
                            ih, iw, ic = image.shape
                            x, y, z = int(land_marks.x * iw), int(land_marks.y * ih), int(land_marks.z * ic)
                            land_mark_details.append(id)
                            land_mark_details.append(x)
                            land_mark_details.append(y)
                            land_mark_details.append(z)
                            img_land_marks.append(land_mark_details)
                            land_mark_details = []
                image_name_and_land_marks[img] = img_land_marks 
                logging.info(f"The landarmarks for image {img} is\n{img_land_marks}")
            logging.info(f"The dictionary with image name and landmarks details :\n{image_name_and_land_marks}")
        except Exception as e:
            logging.info("There is error in getting landmakrs")
            raise CustomException(e,sys)
            
            
        return image_name_and_land_marks
    
    def facial_feature_extraction(self,image_name_and_land_marks):
        logging.info("Facial Feature extraction started")
        try: 
            facial_features = {"Features":['Face lenght','Fore head lenght','lenght from left to right ear tips','lenght from left to right mid ears','lenght from left to right bottom ears','length through lip','Chick bones lenght','Face length left','Face length right','Digonal face length left to right','Digonal face length right to left','nose width','lip length','lip width','left eye lenght','left eye width','right eye lenght','right eye width','facial_area']}
            data = image_name_and_land_marks
            for key,value in data.items():
                logging.info(f"Extracting facial features of {key}")
                features = []
                face_lenght_landmarks = []
                forehead_lenght_landmarks = []
                eartips_lenght_landmarks = []
                midear_lenght_landmarks = []
                bottomear_lenght_landmarks = []
                lips_line_lenght_landmarks = []
                jaw_lenght_landmarks = []
                face_lenght_left_landmarks = []
                face_lenght_right_landmarks = []
                Digonal_face_length_left_to_right_landmarks = []
                Digonal_face_length_right_to_left_landmarks = []
                nose_width_landmarks = []
                lip_lenght_landmarks = []
                lip_width_landmarks = []
                left_eye_lenght_landmarks = []
                left_eye_width_landmarks = []
                right_eye_lenght_landmarks = []
                right_eye_width_landmarks = []
                


                for landmarks in value:
                    if (landmarks[0] == 10) or (landmarks[0] == 152):
                        face_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 54) or (landmarks[0] == 284):
                        forehead_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 127) or (landmarks[0] == 356):
                        eartips_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 234) or (landmarks[0] == 447):
                        midear_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 132) or (landmarks[0] == 361):
                        bottomear_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 58) or (landmarks[0] == 288):
                        lips_line_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 136) or (landmarks[0] == 365):
                        jaw_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 103) or (landmarks[0] == 150):
                        face_lenght_left_landmarks.append(landmarks)
                    elif (landmarks[0] == 332) or (landmarks[0] == 379):
                        face_lenght_right_landmarks.append(landmarks)
                    elif (landmarks[0] == 68) or (landmarks[0] == 397):
                        Digonal_face_length_left_to_right_landmarks.append(landmarks)
                    elif (landmarks[0] == 298) or (landmarks[0] == 172):
                        Digonal_face_length_right_to_left_landmarks.append(landmarks)
                    elif (landmarks[0] == 115) or (landmarks[0] == 344):
                        nose_width_landmarks.append(landmarks)
                    elif (landmarks[0] == 57) or (landmarks[0] == 287):
                        lip_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 0) or (landmarks[0] == 17):
                        lip_width_landmarks.append(landmarks)
                    elif (landmarks[0] == 130) or (landmarks[0] == 133):
                        left_eye_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 23) or (landmarks[0] == 27):
                        left_eye_width_landmarks.append(landmarks)
                    elif (landmarks[0] == 463) or (landmarks[0] == 359):
                        right_eye_lenght_landmarks.append(landmarks)
                    elif (landmarks[0] == 257) or (landmarks[0] == 253):
                        right_eye_width_landmarks.append(landmarks)
                
                face_lenght = (((face_lenght_landmarks[0][1] - face_lenght_landmarks[1][1])**2) + ((face_lenght_landmarks[0][2] - face_lenght_landmarks[1][2])**2))**0.5
                features.append(face_lenght)
                logging.info(f"face_length for {key} is {face_lenght}")
                forehead_lenght = (((forehead_lenght_landmarks[0][1] - forehead_lenght_landmarks[1][1])**2) + ((forehead_lenght_landmarks[0][2] - forehead_lenght_landmarks[1][2])**2))**0.5
                features.append(forehead_lenght)
                logging.info(f"forehead_length for {key} is {forehead_lenght}")
                eartip_length = (((eartips_lenght_landmarks[0][1] - eartips_lenght_landmarks[1][1])**2) + ((eartips_lenght_landmarks[0][2] - eartips_lenght_landmarks[1][2])**2))**0.5
                features.append(eartip_length)
                logging.info(f"eartip_length for {key} is {eartip_length}")
                midear_length = (((midear_lenght_landmarks[0][1] - midear_lenght_landmarks[1][1])**2) + ((midear_lenght_landmarks[0][2] - midear_lenght_landmarks[1][2])**2))**0.5
                features.append(midear_length)
                logging.info(f"midear_length for {key} is {midear_length}")
                bottomear_length = (((bottomear_lenght_landmarks[0][1] - bottomear_lenght_landmarks[1][1])**2) + ((bottomear_lenght_landmarks[0][2] - bottomear_lenght_landmarks[1][2])**2))**0.5
                features.append(bottomear_length)
                logging.info(f"bottomear_length for {key} is {bottomear_length}")
                lipline_length = (((lips_line_lenght_landmarks[0][1] - lips_line_lenght_landmarks[1][1])**2) + ((lips_line_lenght_landmarks[0][2] - lips_line_lenght_landmarks[1][2])**2))**0.5
                features.append(lipline_length)
                logging.info(f"lipline_length for {key} is {lipline_length}")
                jaw_length = (((jaw_lenght_landmarks[0][1] - jaw_lenght_landmarks[1][1])**2) + ((jaw_lenght_landmarks[0][2] - jaw_lenght_landmarks[1][2])**2))**0.5
                features.append(jaw_length)
                logging.info(f"jaw_length for {key} is {jaw_length}")
                face_lenght_left = (((face_lenght_left_landmarks[0][1] - face_lenght_left_landmarks[1][1])**2) + ((face_lenght_left_landmarks[0][2] - face_lenght_left_landmarks[1][2])**2))**0.5
                features.append(face_lenght_left)
                logging.info(f"face_length_left for {key} is {face_lenght_left}")
                face_lenght_right = (((face_lenght_right_landmarks[0][1] - face_lenght_right_landmarks[1][1])**2) + ((face_lenght_right_landmarks[0][2] - face_lenght_right_landmarks[1][2])**2))**0.5
                features.append(face_lenght_right)
                logging.info(f"face_length_right for {key} is {face_lenght_right}")
                Digonal_face_length_left_to_right = (((Digonal_face_length_left_to_right_landmarks[0][1] - Digonal_face_length_left_to_right_landmarks[1][1])**2) + ((Digonal_face_length_left_to_right_landmarks[0][2] - Digonal_face_length_left_to_right_landmarks[1][2])**2))**0.5
                features.append(Digonal_face_length_left_to_right)
                logging.info(f"Digonal_face_length_left_to_right for {key} is {Digonal_face_length_left_to_right}")
                Digonal_face_length_right_to_left = (((Digonal_face_length_right_to_left_landmarks[0][1] - Digonal_face_length_right_to_left_landmarks[1][1])**2) + ((Digonal_face_length_right_to_left_landmarks[0][2] - Digonal_face_length_right_to_left_landmarks[1][2])**2))**0.5
                features.append(Digonal_face_length_right_to_left)
                logging.info(f"Digonal_face_length_right_to_left for {key} is {Digonal_face_length_right_to_left}")
                nose_width = (((nose_width_landmarks[0][1] - nose_width_landmarks[1][1])**2) + ((nose_width_landmarks[0][2] - nose_width_landmarks[1][2])**2))**0.5
                features.append(nose_width)
                logging.info(f"nose_width for {key} is {nose_width}")
                lip_lenght = (((lip_lenght_landmarks[0][1] - lip_lenght_landmarks[1][1])**2) + ((lip_lenght_landmarks[0][2] - lip_lenght_landmarks[1][2])**2))**0.5
                features.append(lip_lenght)
                logging.info(f"lip_lenght for {key} is {lip_lenght}")
                lip_width = (((lip_width_landmarks[0][1] - lip_width_landmarks[1][1])**2) + ((lip_width_landmarks[0][2] - lip_width_landmarks[1][2])**2))**0.5
                features.append(lip_width)
                logging.info(f"lip_width for {key} is {lip_width}")
                left_eye_lenght = (((left_eye_lenght_landmarks[0][1] - left_eye_lenght_landmarks[1][1])**2) + ((left_eye_lenght_landmarks[0][2] - left_eye_lenght_landmarks[1][2])**2))**0.5
                features.append(left_eye_lenght)
                logging.info(f"left_eye_lenght for {key} is {left_eye_lenght}")
                left_eye_width = (((left_eye_width_landmarks[0][1] - left_eye_width_landmarks[1][1])**2) + ((left_eye_width_landmarks[0][2] - left_eye_width_landmarks[1][2])**2))**0.5
                features.append(left_eye_width)
                logging.info(f"left_eye_width for {key} is {left_eye_width}")
                right_eye_lenght = (((right_eye_lenght_landmarks[0][1] - right_eye_lenght_landmarks[1][1])**2) + ((right_eye_lenght_landmarks[0][2] - right_eye_lenght_landmarks[1][2])**2))**0.5
                features.append(right_eye_lenght)
                logging.info(f"right_eye_lenght for {key} is {right_eye_lenght}")
                right_eye_width = (((right_eye_width_landmarks[0][1] - right_eye_width_landmarks[1][1])**2) + ((right_eye_width_landmarks[0][2] - right_eye_width_landmarks[1][2])**2))**0.5
                features.append(right_eye_width)
                logging.info(f"right_eye_width for {key} is {right_eye_width}")
                
                area = 0
                for i in range(len(value)):
                    j = (i + 1) % len(value)
                    area += (value[i][1] * value[j][2]) - (value[j][2] * value[i][2])
                area /= 2
                features.append(abs(area))
                logging.info(f"Area of the face for {key} is {area}")

                facial_features[key] = features
                features = []
        except Exception as e:
            logging.info("Error appears while extracting various facial features from images")
            raise CustomException(e,sys)
        logging.info(f"Here is the dictionary containing image names and the facial features coressponding to them\n{facial_features}")
        return facial_features
    
    def facial_features_dataframe(self,facial_features):
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
            logging.info(f"The dataframe of faical feature is:\n{df}")
            df.to_csv(self.feature_extrator_config.extracted_facial_features,index=False,header=True)
        except Exception as e:
            logging.info("An error occured while creating dataframe of facial features")
            raise CustomException(e,sys)
       
        return self.feature_extrator_config.extracted_facial_features

        
