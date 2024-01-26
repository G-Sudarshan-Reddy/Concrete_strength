import os
import sys
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictionPipeline:
    # ... (other methods)

    def predict(self, feature_df):
        try:
            model_path = os.path.join('artifacts', 'model.pkl')
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)

            print(f"Type of model: {type(model)}")
            print(f"Type of preprocessor: {type(preprocessor)}")

            data_scaled = preprocessor.transform(feature_df)
            prediction = model.predict(data_scaled)
            return prediction

        except Exception as e:
            raise CustomException(e, sys)


# class CustomData:
#     def __init__(self,
#                  cement: int,
#                  Blast_Furnace_Slag: int,
#                  Fly_Ash: int,
#                  Water: int,
#                  Superplasticizer: int,
#                  Coarse_Aggregate: int,
#                  Fine_Aggregate: int,
#                  Age: str):
#         self.cement = cement
#         self.Blast_Furnace_Slag = Blast_Furnace_Slag
#         self.Fly_Ash = Fly_Ash
#         self.Water = Water
#         self.Superplasticizer = Superplasticizer
#         self.Coarse_Aggregate = Coarse_Aggregate
#         self.Fine_Aggregate = Fine_Aggregate
#         self.Age = Age

#     def get_data_to_df(self):
#         try:
#             data_input_dict = {
#                 "cement": [self.cement],
#                 "Blast_Furnace_Slag": [self.Blast_Furnace_Slag],
#                 "Fly_Ash": [self.Fly_Ash],
#                 "Water": [self.Water],
#                 "Superplasticizer": [self.Superplasticizer],
#                 "Coarse_Aggregate": [self.Coarse_Aggregate],
#                 "Fine_Aggregate": [self.Fine_Aggregate],
#                 "Age": [self.Age]
#             }

#             return pd.DataFrame(data_input_dict)
#         except Exception as e:
#             raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 Cement: int,
                 Blast_Furnace_Slag: int,
                 Fly_Ash: int,
                 Water: int,
                 Superplasticizer: int,
                 Coarse_Aggregate: int,
                 Fine_Aggregate: int,
                 Age: str):
        self.Cement = Cement
        self.Blast_Furnace_Slag = Blast_Furnace_Slag
        self.Fly_Ash = Fly_Ash
        self.Water = Water
        self.Superplasticizer = Superplasticizer
        self.Coarse_Aggregate = Coarse_Aggregate
        self.Fine_Aggregate = Fine_Aggregate
        self.Age = Age

    def get_data_to_df(self):
        try:
            data_input_dict = {
                "Cement": [self.Cement],
                "Blast_Furnace_Slag": [self.Blast_Furnace_Slag],
                "Fly_Ash": [self.Fly_Ash],
                "Water": [self.Water],
                "Superplasticizer": [self.Superplasticizer],
                "Coarse_Aggregate": [self.Coarse_Aggregate],
                "Fine_Aggregate": [self.Fine_Aggregate],
                "Age": [self.Age]
            }

            return pd.DataFrame(data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)

