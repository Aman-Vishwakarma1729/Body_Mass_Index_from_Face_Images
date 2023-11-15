
import os
import sys
import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from urllib.request import urlretrieve
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class Image_and_Height_Data_Scrapping_Config:
    scrapped_images_path = os.path.join(os.getcwd(),'images_data')
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
    
    def get_image_and_height_from_urls(self,image_page_link_list):
        logging.info(f"stage-2: Scrapping image and respective height for each image")
        count = 0
        os.makedirs("images_data", exist_ok=True)
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
    

