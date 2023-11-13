from src.components.image_and_height_data_scrapping import Image_and_Height_Data_Scrapping

initialize_scrapping = Image_and_Height_Data_Scrapping()
pg_links_list = initialize_scrapping.get_urls("http://xn-----6kcczalffeh6afgdgdi2apgjghic4org.xn--p1ai/")
img_name_and_height_dict = initialize_scrapping.get_image_and_height_from_urls(pg_links_list)
img_name_and_height_dataframe = initialize_scrapping.get_imgname_and_height_dataframe(img_name_and_height_dict)
print(f"The path for dataframe consisting of image name and repsective height is {img_name_and_height_dataframe}")