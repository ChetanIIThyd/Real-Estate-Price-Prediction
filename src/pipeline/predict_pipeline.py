import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        total_sqft : float,
        bath: int,
        BHK: int,
        location: str,
        ):

        self.total_sqft = total_sqft

        self.bath = bath

        self.BHK = BHK

        self.location = location

       

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "total_sqft": [self.total_sqft],
                "bath": [self.bath],
                "BHK": [self.BHK],
                "location": [self.location],               
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
        