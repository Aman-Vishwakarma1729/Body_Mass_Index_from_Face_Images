# <div align="center">Body Mass Index Prediction from Face Images</div>
<div align="center">
  <img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/c60a64ae-a725-4143-a6e5-cf094a32b030" alt="Designer" width="500"/>
</div>

## Table of content
--------------
1. [Introduction](#introduction)
2. [Tools Used](#tools-used)
3. [About Data](#about-data)
4. [Project Workflow](#project-workflow)

## Introduction
--------------
Obesity is a major global health concern, affecting over **41 million children under the age of five** and **650 million adults worldwide**. It is a major risk factor for many chronic diseases, including **heart disease**, **stroke**, **type 2 diabetes**, and **certain types of cancer**. Accurate and timely estimation of Body Mass Index (BMI) is essential for assessing an individual's health status and risk of obesity-related diseases. Sometimes there are some cases regarding malnourishment which might be risk factor for many chronic liver disease, including Heptacelluar Carcinoma and liver cirrhosis. However, traditional BMI measurement methods, such as weighing and measuring height, can be inconvenient and impractical, especially for large populations. **Computer vision-based BMI prediction from facial images** offers a promising **non-invasive** and **accessible** alternative. By leveraging the power of **machine learning**, **computer vision models** can extract subtle features from facial images that correlate with BMI.
#### This project uses DEEP COMPUTER VISION for predicting BODY MASS INDEX from facial images.

## Tools used
--------------
<div style="display: flex; justify-content: space-around;">

<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/7c50e610-78ac-4a06-84d8-aec658403dca" alt="Python" width="200"/> 
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/f396c5ef-b037-43b6-b999-c03620f8a309" alt="Pandas" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/8528f822-17f5-411a-9466-4aefc3addc0d" alt="Numpy" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/13aba99d-8572-467f-9404-4d2ce7bfddf1" alt="Matplotlib" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/1b771704-2fbc-44b6-94f5-254965785c38" alt="Seaborn" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/74b4c8da-cb24-40ff-97a7-b3692b59fe9d" alt="Sci-Kit Learn" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/c6ede8f6-4d17-403e-ab30-e05fb9431255" alt="Tensorflow" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/2e514047-5d42-4db3-a122-75c621db0f08" alt="Mediapipe" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/e2c909d3-277e-42a6-9e2d-56da92d69f8b" alt="Keras" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/1ba19e5a-d7f7-4fee-8037-2a31a5212d7f" alt="Git Hub Actions" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/9a49a761-e08d-4f02-ab9a-d948bb7f64a7" alt="MongoDB" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/82f6be95-533e-48b2-8016-5eecc031048a" alt="Dockers" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/757da853-d49e-4b71-ae02-a0a846306009" alt="Amazon ECR" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/41ebc113-df65-4421-b45a-d593224cef3e" alt="Amazon App Runner" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/adbdc8ff-aa49-4762-8c96-a626156fa94b" alt="Flask" width="200"/>
&nbsp;&nbsp;&nbsp;
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/69d963e5-4615-44d9-8ebb-fdf9e16fcb8f" alt="Anaconda" width="200"/>

</div>

## About Data
--------------
We have used dataset that contain facial images of celebrities and their height which we have scrapped from [Celeb's facial image data](http://xn-----6kcczalffeh6afgdgdi2apgjghic4org.xn--p1ai/) and also we have used another data set to build a model that predicts **WEIGHTS** from **HEIGHT** and **GENDER** as the above given data source does not have weights data, and that data is present at [HWG_Data.csv](notebooks/Height_to_Weight_Data_and_Models/HWG_Data.csv).

## Project Workflow
--------------
1) The data that contains **facial images** of celeb's and thier **height** is scrapped using the scrapping technique and libraries like **BeautifulSoup** the code for scrapping **Images** and **Height** is at [image_and_height_data_scrapping_1.py](src/components/image_and_height_data_scrapping_1.py).

2) Pandas **DataFrame** is created that contains **Image names** and respective **Height**. The same data is located at [img_name_and_height.csv](artifacts/img_name_and_height.csv).

3) The image that are scrapped is **cropped** using pre-trained model **haarcascade_frontalface** that locate face and crop it.we cropped the images so that we have image of similar dimensions and we only want facial image so we cropped facial part and those images are saved at [final_images_data](final_images_data) with respective image name.
   
4) We used a predtrained model that predicts **Gender** using **Face Image** so that we get **Gender** of images that we scrapped  and we already have their **Height** the code for same is located at [Image_to_gender_prediction.py](src/components/Image_to_gender_prediction.py). The dataset than obtained consist **Image name**, **Height** and **Gender** and is saved at [img_name_height_gender.csv](artifacts/img_name_height_gender.csv).

5) To get the **BODY MASS INDEX (BMI)** of or celeb's facial data we needed their **Weights** and we now already have their **Gender** and **Height** so we build a model that predict's the **Weight** using **Gender** and **Height** and for that we used this data [HWG_Data.csv](notebooks/Height_to_Weight_Data_and_Models/HWG_Data.csv). We have done some preprocessing on **HWG_Data.csv** data which can be seen at [Height_Weight_Data_Analysis.ipynb](notebooks/Height_to_Weight_Data_and_Models/Height_Weight_Data_Analysis.ipynb). And obtain the data that we needed to develop a model that can predict **Weight** using **Gender** and **Height** and that data is located at [HWG_Data_Updated.csv](notebooks/Height_to_Weight_Data_and_Models/HWG_Data_Updated.csv).
 
6) We have done modular coding to get a model that can predict **Weight** from **Gender** and **Height**. We first did **data ingestion** to get **raw data** [hgw_raw.csv](artifacts/hgw_raw.csv), **train data** [hgw_train.csv](artifacts/hgw_train.csv), and **test data** [hgw_test.csv](artifacts/hgw_test.csv),and code for data ingestion is at [height_and_gender_to_weight_data_ingestion.py](src/components/height_and_gender_to_weight_data_ingestion.py). Once the data ingestion is completed we have done modular coding for automated data preprocessing where we have done **scalling** [hgw_scaler.pkl](artifacts/hgw_scaler.pkl) and **data encoding** [hgw_gender_encoder.pkl](artifacts/hgw_gender_encoder.pkl) and code for data transformation/preprocessing is at [height_and_gender_to_weight_data_transformation.py](src/components/height_and_gender_to_weight_data_transformation.py). Once data transformation is done we have automated the process of model training and saving a model that predict **Weights** from **Gender** and **Height** with best accuracy the codel for same is at [height_and_gender_to_weight_model_trainer.py](src/components/height_and_gender_to_weight_model_trainer.py), and we got the height predictor model [hg_to_w_predictor_model.pkl](artifacts/hg_to_w_predictor_model.pkl). Than we have used this model to predict the **Weight** of the celeb's data where we already had there **Gender** and **Height** and the code for same is at [getting_final_dataset.py](src/components/getting_final_dataset.py). So we got new dataset with **Image name**, **Gender**, **Height** and **Weight** and than we using simple **BMI** formula which is **(Weight in KG)/(Height in meter)^2** and got final data set which we used further.

7) To extract facial features from celeb's face images we used **MEDIAPIPE** which uses **Deep Computer Vision** to locate various **land marks** on face and each land mark has **Unique ID** and **Co-ordinate**. Using this **land marks** and **Co-ordinates** we extracted **19 facial features** from each image the code for same is [facial_feature_extraction.py](src/components/facial_feature_extraction.py). And those facial features are as follows:
   
      | Acronym | Description                         |      
      |---------|-------------------------------------|
      | FL      | Face Length                         |
      | FRHL    | Forehead Length                     |
      | ETL     | Ear Tip Length                      |
      | MEL     | Mid Ear Length                      |
      | BEL     | Bottom Ear Length                   |
      | LIPL_L  | Lip Line Length                     |
      | JL      | Jaw Length                          |
      | FLL     | Face Length Left                    |
      | FLR     | Face Length Right                   |
      | DFL_L_R | Diagonal Face Length Left to Right  |
      | DFL_R_L | Diagonal Face Length Right to Left  |
      | NWL     | Nose Width Length                   |
      | L_L     | Lip Length                          |
      | L_W     | Lip Width                           |
      | LEL     | Left Eye Length                     |
      | LEW     | Left Eye Width                      |
      | REL     | Right Eye Length                    |
      | REW     | Right Eye Width                     |
      | FA      | Facial Area                         |

<div>
<img src="https://github.com/Aman-Vishwakarma1729/Body_Mass_Index_from_Face_Images/assets/110922641/1d282122-a6bb-4ddc-927c-9a1065a54b5a" alt="Refference Image" width="500"/>
</div>

Once all this features are extracted for all the celeb's images it saved in pandas dataframe and then we **joined** it with our previous data set that we obtained **step 6**. And finally we have data dataset with all this features and coressponding **BMI** with **Gender** and dataset are joined on the basis of **Image name**. The final data set looks like [final_bmi_dataset.csv](artifacts/final_bmi_dataset.csv).

8) Once we have the **final_bmi_dataset.csv** dataset we perform basic analysis and rough model building in the notebook at [BMI_Data_Analysis.ipynb](notebooks/BMI_Data_Analysis_and_Modelling/BMI_Data_Analysis.ipynb).
