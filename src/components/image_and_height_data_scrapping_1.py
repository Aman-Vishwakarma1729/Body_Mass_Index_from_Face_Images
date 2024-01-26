
import os
import sys
import shutil
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.request import urlretrieve
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import cv2


@dataclass
class Image_and_Height_Data_Scrapping_Config:
    scrapped_images_path = os.path.join(os.getcwd(),'images_data')
    croped_images_path = os.path.join(os.getcwd(),'final_images_data')
    scrapped_height_data_path = os.path.join('artifacts','img_name_and_height.csv')

class Image_and_Height_Data_Scrapping:
    def __init__(self):
        self.scrapping_config = Image_and_Height_Data_Scrapping_Config()
        
    def get_urls(self,url):
        logging.info("Image and height data scrapping is initiated")
        logging.info("stage-1: Scrapping links for each image on each page")
        image_page_link = []
        for i in range(1,51):                 
            logging.info(f"We are on page {i}")
            url = urljoin(url,f"?page{i}")
            try:
                response = requests.get(url)
                logging.info(f"Connection with {url} established sucessfully")
            except Exception as e:
                logging.info(f'There is problem while accesing the given url: {url} and error is: {e} on page number {i}')
                raise CustomException(e,sys)
            try:
                soup = BeautifulSoup(response.content,"html.parser")
                parsing_1 = soup.find_all('div',{'class':'content_body'})[0].find_all('div',{'class':'proton_tile'})
            except Exception as e:
                logging.info(f'There is problem in parsing the link {url}')
                raise CustomException(e,sys)
            try:
                for element in parsing_1:
                    pg_links = element.find_all('a')
                    for pg_link in pg_links:
                        link = pg_link.get('href')
                        link = urljoin(url,link)
                        image_page_link.append(link)
                logging.info(f"Successfully retrived links for all images on page {i}")               
            except Exception as e:
                logging.info(f'There is problem in getting image page link for {url} of page {i}')
                raise CustomException(e,sys)       
        logging.info(f"Here is the link of all the images\n{image_page_link}\n**There are {len(image_page_link)}** images")
        return image_page_link
    
    def create_req_folders(self,):
        try:
            logging.info("Created a folder named 'Images_data' and 'final_image_data'")
            images_data_folder = os.path.join(os.getcwd(), "images_data")
            if not os.path.exists(images_data_folder):
               os.mkdir(images_data_folder)
            final_images_data_folder = os.path.join(os.getcwd(), "final_images_data")
            if not os.path.exists(final_images_data_folder):
               os.mkdir(final_images_data_folder)
            artifacts_folder = os.path.join(os.getcwd(), "artifacts")
            if not os.path.exists(artifacts_folder):
               os.mkdir(artifacts_folder)
        except Exception as e:
                logging.info(f'Exception occured while creating an required folders')
                raise CustomException(e,sys)
    

    def locate_and_crop_face(self,image_path,output_path):
        try:
            logging.info(f"Working on image {image_path} : locating and croping face")
            image = cv2.imread(image_path)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                new_w = int(w * 1.2)
                new_h = int(h * 1.2)
                x -= int((new_w - w) / 2)
                y -= int((new_h - h) / 2)
                cropped_face = image[y:y+new_h, x:x+new_w]
                cv2.imwrite(output_path, cropped_face)
                logging.info(f"Face located and cropped. Saved to: {output_path}")
            else:
                logging.info("No face detected in the image.")
        except Exception as e:
                logging.info(f'Error occur while locating and croping the face')
                raise CustomException(e,sys) 
    
    def get_image_and_height_from_urls(self,image_page_link_list):
        logging.info(f"stage-2: Scrapping image and respective height for each image")
        count = 0
        image_names_with_height = {}
        for img_links in image_page_link_list:
            count=count+1
            logging.info(f'We are on {count} image')
            try:
                response = requests.get(img_links)
                logging.info(f"Connection with {img_links} has been created successfully")
            except Exception as e:
                logging.info(f'There is problem while accesing the given url: {img_links} and error is: {e} on image number {count}')
                raise CustomException(e,sys)
            try:
                soup = BeautifulSoup(response.content,'html.parser')
                parsing_2 = soup.find('div',{'class':'content_body'}).find('div').find('div',{'class':'eMessage'}).find_all('div')[0]
                parsing_3 = soup.find('div',{'class':'content_body'}).find('span')
            except Exception as e:
                logging.info(f'There is problem in parsing the link {img_links}')
                raise CustomException(e,sys)
            try:
                img_tags = parsing_2.find_all('img')
                text = parsing_3.text
                height = text.split(" ")[2]
                img_url = img_tags[0].get('src')
                img_url = img_url.strip()
                if img_url:
                            img_url = urljoin("http://xn-----6kcczalffeh6afgdgdi2apgjghic4org.xn--p1ai",img_url)
                            img_name = os.path.basename(img_url)
                            img_path = os.path.join("images_data", img_name)
                            try:
                                urlretrieve(img_url, img_path)
                                logging.info(f"Downloaded: {img_url}")
                                image_names_with_height[img_name] = height
                                logging.info(f"{img_name}:{height}")
                            except Exception as e:
                                logging.info(f"Failed to download: {img_url}")
                                logging.info(f"Error: {e}")
                                raise CustomException(e,sys)
                           

            except Exception as e:
                logging.info("There is problem in getting imag_tags and text")
                raise CustomException(e,sys)
        return image_names_with_height
    
    

    def get_imgname_and_height_dataframe(self,image_names_with_height):
        logging.info("stage-3: Finally getting dataframe")
        try:
            img_name = []
            height = []
            for key,value in image_names_with_height.items():
                img_name.append(key)
                height.append(value)
            
            df = pd.DataFrame({"image_name":img_name,"height":height})
            logging.info("Pandas dataframe has been created sucessfully")
            logging.info("Dataframe consist of image names and the respective heights")
            df.to_csv(self.scrapping_config.scrapped_height_data_path,index=False,header=True)
            
        except Exception as e:
            logging.info("Error occured while creating dataframe")
            raise CustomException(e,sys)
     
        return self.scrapping_config.scrapped_images_path,self.scrapping_config.scrapped_height_data_path
    
    
    def locate_and_crop_face(self,image_path,output_path):
        try:
            logging.info(f"Working on image {image_path} : locating and croping face")
            image = cv2.imread(image_path)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                new_w = int(w * 1.2)
                new_h = int(h * 1.2)
                x -= int((new_w - w) / 2)
                y -= int((new_h - h) / 2)
                cropped_face = image[y:y+new_h, x:x+new_w]
                cv2.imwrite(output_path, cropped_face)
                logging.info(f"Face located and cropped. Saved to: {output_path}")
            else:
                logging.info("No face detected in the image.")
        except Exception as e:
                logging.info(f'Error occur while locating and croping the face')
                raise CustomException(e,sys)
    
    def crop(self,scrapped_image_folder_path):
        count = 0
        for filename in os.listdir(scrapped_image_folder_path):
            count = count + 1
            try:
                logging.info(f"gender cropping started for {filename}")
                logging.info(f"We are on {count} image")
                if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    image_path = os.path.join(scrapped_image_folder_path, filename)
                    output_path = os.path.join(os.path.join(os.getcwd(),"final_images_data"),filename)
                    self.locate_and_crop_face(image_path,output_path)
            except Exception as e:
                   logging.info(f'Error occur while locating and croping the face:{filename}')


        return self.scrapping_config.croped_images_path
    
    def remove_folder_images_data(self,):
        try:
            logging.info("Removing the 'Images_data' folder as we have croped images in 'final_images_data' folder which will bw used for futher task")
            directory_to_remove = "images_data"
            if os.path.exists(directory_to_remove):
               shutil.rmtree(directory_to_remove)
               logging.info(f"The directory '{directory_to_remove}' has been removed.")
            else:
               logging.info(f"The directory '{directory_to_remove}' does not exist.")
        except Exception as e:
            logging.info("Error occured while removing 'images_data' folder")
            raise CustomException(e,sys)
