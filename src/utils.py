import os
import sys
import pickle
import numpy as np 
import pandas as pd
from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.info(f"An Excveption occured while saving {obj}")
        raise CustomException(e, sys)