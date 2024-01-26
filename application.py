from flask import Flask,request,render_template,redirect, url_for
import cv2
import os
import numpy as np
import datetime
from src.pipelines.bmi_prediction_pipeline import BMI_Prediction_Pipeline

application=Flask(__name__)

app=application

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/get_image', methods=['POST'])
def get_image():
    if request.form['action'] == 'capture':
        return render_template('prediction.html')
    elif request.form['action'] == '-upload':
        return render_template('prediction.html')
    else:
        return render_template('index.html')

@app.route('/prediction')
def result_page():
    
    image_path = request.args.get('image_path', '')

    return render_template('result.html', image_path=image_path)

if __name__=="__main__":
    app.run(host='0.0.0.0')

        