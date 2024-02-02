from flask import Flask,request,render_template,redirect, url_for, send_file, g, flash, session
from itsdangerous import URLSafeTimedSerializer
import cv2
import os
import numpy as np
import datetime
import sys
import time
import shutil
import mediapipe as mp
from deepface import DeepFace
from src.pipelines.bmi_prediction_pipeline import BMI_Prediction_Pipeline
from src.logger import logging
from src.exception import CustomException

application=Flask(__name__)

app=application
app.secret_key = '1729@AmanV' 
@app.route('/')
def index_page():
    try:
       logging.info("Sucessfully made connection with index.html")
       remove_croped_image_for_prediction_folder_path = os.path.join(os.getcwd(),"croped_image_for_prediction")
       remove_input_image_for_prediction_folder_path = os.path.join(os.getcwd(),"input_image_for_prediction")
       if os.path.exists(remove_croped_image_for_prediction_folder_path):
            shutil.rmtree(remove_croped_image_for_prediction_folder_path)
            logging.info(f"The directory '{remove_croped_image_for_prediction_folder_path}' has been removed.")
       else:
            logging.info("The folder {remove_croped_image_for_prediction_folder_path} was already removed")

       if os.path.exists(remove_input_image_for_prediction_folder_path):
            shutil.rmtree(remove_input_image_for_prediction_folder_path)
            logging.info(f"The directory '{remove_input_image_for_prediction_folder_path}' has been removed.")
       else:
            logging.info("The folder {remove_input_image_for_prediction_folder_path} was already removed")
       return render_template('index.html')
    except Exception as e:
        logging.info("An error occured while accessing index.html")
        raise CustomException(e,sys)
    
@app.route('/about')
def about_page():
    try:
       logging.info("Sucessfully made connection with about.html")
       return render_template('about.html')
    except Exception as e:
        logging.info("An error occured while accessing about.html")
        raise CustomException(e,sys)


@app.route('/get_image_capture', methods=['POST'])
def get_image_capture():
    try:
        if request.form['capture_action'] == 'capture':
            logging.info("Request to campture an image has been made")
            camera = cv2.VideoCapture(0)
            camera.set(3, 1021) 
            camera.set(4, 1021) 
            logging.info("Camera has started")
            time.sleep(3)
            _, frame = camera.read()
            logging.info("Frame has been captured")
            camera.release()
            BMI_Prediction_Pipeline().create_required_folders()
            logging.info("Required folder has been created")
            img_folder = os.path.join(os.getcwd(),"input_image_for_prediction")
            logging.info(f"Got access to folder to save the input image: {img_folder}")
            timestamp = time.strftime("%Y%m%d%H%M%S")
            img_name = f"captured_image_{timestamp}.png"
            logging.info(f"The input image will be saved as: {img_name}")
            cv2.imwrite(os.path.join(img_folder,img_name), frame)
            logging.info(f"Image has been saved as {img_name} at {img_folder}")
            logging.info(f"The image is being saved: {os.path.join(img_folder,img_name)}")
            return render_template('prediction.html', image_path=img_name)
        else:
            return render_template('index.html')
    except Exception as e:
        logging.info("An error occured while accessing capturing image for prediction of BMI")
        raise CustomException(e,sys)


@app.route('/get_image_upload', methods=['POST'])
def get_image_upload():
    try:
        logging.info("Uploading image for prediction")
        if request.form['upload_action'] == 'upload':
            logging.info(request.form)
            BMI_Prediction_Pipeline().create_required_folders()
            logging.info("The required folder has been created")
            uploaded_file = request.files['image_file']
            logging.info(uploaded_file)
            if uploaded_file.filename != '':
                logging.info("Required folder has been created")
                img_folder = os.path.join(os.getcwd(),"input_image_for_prediction")
                logging.info(f"Got access to folder to save the input image: {img_folder}")
                file_path = os.path.join(img_folder, uploaded_file.filename)
                uploaded_file.save(file_path)
                logging.info(f"The input image has been saved to {file_path}")
                return render_template('prediction.html',image_path=uploaded_file.filename)
        else:
            return render_template('index.html')
    except Exception as e:
        logging.info("An error occured while uploading image for prediction of BMI")
        raise CustomException(e,sys)


   
@app.route('/serve_image/<filename>')
def serve_image(filename):
    img_folder = os.path.join(os.getcwd(), "input_image_for_prediction")
    logging.info("The input image has been send to server to display... !!!")
    return send_file(os.path.join(img_folder, filename))


@app.route('/prediction', methods=['POST'])
def result():
    try:
        logging.info("Prediction of BMI from input image has been started")
        if request.form['prediction'] == 'predict':
           logging.info(request.form)
           directory_path = BMI_Prediction_Pipeline().get_directory_path()
           face_detected,croped_image_for_prediction_directory_path = BMI_Prediction_Pipeline().crop(directory_path)
           logging.info(f"Got the path of image for gender prediction")

           if not face_detected:
                flash("Unable to detect your face in the given image. Please try again.")
                BMI_Prediction_Pipeline().remove_folder_images_data()
                BMI_Prediction_Pipeline().clear_input_image(croped_image_for_prediction_directory_path)
                return redirect(url_for('index_page'))
           
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
           bmi_value = predicted_bmi
           bmi_value  = bmi_value[0]
           bmi_value  = f"{bmi_value:.2f}"

           for filename in os.listdir(croped_image_for_prediction_directory_path):
               image_path = os.path.join(os.path.join(os.getcwd(),"croped_image_for_prediction"),filename)
               logging.info(f"The image to display on prediction page is locatted at: {image_path}")
           return render_template('prediction.html', bmi_value=bmi_value, image_path=image_path)
        
    except Exception as e:
        logging.info("An error occurred while predicting BMI")
        flash("An error occurred during BMI prediction. Please try again.")
        return redirect(url_for('index_page'))
    
@app.route('/clear_folders', methods=['POST'])
def clear_folders():
    try:
        if request.form['clear_action'] == 'clear':
            logging.info("Got request to clear the folder that contain input and cropped images")
            remove_croped_image_for_prediction_folder_path = os.path.join(os.getcwd(),"croped_image_for_prediction")
            remove_input_image_for_prediction_folder_path = os.path.join(os.getcwd(),"input_image_for_prediction")
            if os.path.exists(remove_croped_image_for_prediction_folder_path):
               shutil.rmtree(remove_croped_image_for_prediction_folder_path)
               logging.info(f"The directory '{remove_croped_image_for_prediction_folder_path}' has been removed.")
            else:
                logging.info("The folder {remove_croped_image_for_prediction_folder_path} was already removed")

            if os.path.exists(remove_input_image_for_prediction_folder_path):
               shutil.rmtree(remove_input_image_for_prediction_folder_path)
               logging.info(f"The directory '{remove_input_image_for_prediction_folder_path}' has been removed.")
            else:
                logging.info("The folder {remove_input_image_for_prediction_folder_path} was already removed")

            return redirect(url_for('index_page'))
    except Exception as e:
        logging.info("An error occurred while clearing folders.")
        flash("An error occurred during clearing")
        return redirect(url_for('index_page'))


## Application
if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)

        