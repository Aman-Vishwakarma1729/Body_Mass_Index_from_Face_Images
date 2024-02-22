import pandas as pd
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGODB_URI = os.environ['MONGODB_URI']
client = MongoClient(MONGODB_URI)

db = client["Final_BMI_Dataset"]
BMI_collection = db["bmi_collection"]

bmi_data_path = os.path.join(os.getcwd(),os.path.join('artifacts','final_bmi_dataset.csv'))
bmi_data = pd.read_csv(bmi_data_path)

for FL,FRHL,ETL,MEL,BEL,LIPL_L,JL,FLL,FLR,DFL_L_R,DFL_R_L,NWL,L_L,L_W,LEL,LEW,REL,REW,FA,Gender,BMI in zip(bmi_data['FL'],bmi_data['FRHL'],bmi_data['ETL'],bmi_data['MEL'],bmi_data['BEL'],bmi_data['LIPL_L'],bmi_data['JL'],bmi_data['FLL'],bmi_data['FLR'],bmi_data['DFL_L_R'],bmi_data['DFL_R_L'],bmi_data['NWL'],bmi_data['L_L'],bmi_data['L_W'],bmi_data['LEL'],bmi_data['LEW'],bmi_data['REL'],bmi_data['REW'],bmi_data['FA'],bmi_data['Gender'],bmi_data['BMI']):
    data = {

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
                "FA" : FA,
                "Gender" : Gender,
                "BMI" : BMI

    }
    
    BMI_collection.insert_one(data)
