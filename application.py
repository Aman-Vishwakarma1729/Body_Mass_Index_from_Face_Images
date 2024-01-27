from flask import Flask,request,render_template,redirect, url_for, send_file
import cv2
import os
import numpy as np
import datetime
import sys
import time
from src.pipelines.bmi_prediction_pipeline import BMI_Prediction_Pipeline
from src.logger import logging
from src.exception import CustomException

application=Flask(__name__)

app=application

@app.route('/')
def index_page():
    try:
       logging.info("Sucessfully made connection with index.htm")
       return render_template('index.html')
    except Exception as e:
        logging.info("An error occured while accessing index.html")
        raise CustomException(e,sys)
    
@app.route('/about')
def about_page():
    try:
       logging.info("Sucessfully made connection with about.htm")
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
            logging.info("Camera has started")
            _, frame = camera.read()
            logging.info("Frame has been captured")
            camera.release()
            BMI_Prediction_Pipeline().create_required_folders()
            img_folder = os.path.join(os.getcwd(),"input_image_for_prediction")
            timestamp = time.strftime("%Y%m%d%H%M%S")
            img_name = f"captured_image_{timestamp}.png"
            cv2.imwrite(os.path.join(img_folder,img_name), frame)
            logging.info(f"Image has been saved as {img_name} at {img_folder}")
            logging.info(f"The image is being saved at: {os.path.join(img_folder,img_name)}")
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
            uploaded_file = request.files['image_file']
            logging.info(uploaded_file)
            if uploaded_file.filename != '':
                BMI_Prediction_Pipeline().create_required_folders()
                logging.info("Required folder has been created")
                img_folder = os.path.join(os.getcwd(),"input_image_for_prediction")
                file_path = os.path.join(img_folder, uploaded_file.filename)
                uploaded_file.save(file_path)
                logging.info("File has been saved")
                return render_template('prediction.html',image_path=uploaded_file.filename)
        else:
            return render_template('index.html')
    except Exception as e:
        logging.info("An error occured while uploading image for prediction of BMI")
        raise CustomException(e,sys)

   
@app.route('/serve_image/<filename>')
def serve_image(filename):
    img_folder = os.path.join(os.getcwd(), "input_image_for_prediction")
    return send_file(os.path.join(img_folder, filename))


@app.route('/prediction', methods=['POST'])
def result():
    try:
        logging.info("Prediction has been started")
        if request.form['prediction'] == 'predict':
           logging.info(request.form)
           bmi_value = "10"
           image_path = request.form.get('image_path', '')

           return render_template('prediction.html', bmi_value=bmi_value, image_path=image_path)
    except Exception as e:
        logging.info("An error occurred while predicting BMI")
        raise CustomException(e, sys)

 
if __name__=="__main__":
    app.run(host='0.0.0.0')

        