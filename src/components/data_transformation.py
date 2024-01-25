import os, sys
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

@dataclass
class Data_transformation_config:
    preprocessor_file_path = os.path.join("artifacts", "preprocessor.pkl")

class Datatransformation:
    def __init__(self):
        self.transformation_config = Data_transformation_config()

    def get_transformation(self):
        logging.info("Transformation phase started")
        try:
            
            col_list = ['Cement', 'Blast_Furnace_Slag', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarse_Aggregate', 'Fine_Aggregate', 'Age']

            col_pipeline = Pipeline(
                steps=[
                    ("MinMax",MinMaxScaler())
                ]
            )

            logging.info("preprocessor started")
            preprocessor = ColumnTransformer(
                [
                    ("col_pipeline",col_pipeline, col_list)
                ]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            # logging.INFO("Data Transformation function entered")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            print(train_df.head(5))

            preprocessor_obj = self.get_transformation()

            logging.info("Splitting of dataset done")

            target_col = "Concrete_strength"

            input_train_df = train_df.drop(columns=[target_col], axis=1)
            target_train_df = train_df[target_col]

            input_test_df = test_df.drop(columns=[target_col], axis=1)
            target_test_df = test_df[target_col]
            logging.info("Target and input splitted part 1")

            input_train_arr = preprocessor_obj.fit_transform(input_train_df)
            input_test_arr = preprocessor_obj.transform(input_test_df)

            logging.info("Target and input splitted part 2")
            train_arr = np.c_[
                input_train_arr, np.array(target_train_df)
            ]

            test_arr = np.c_[
                input_test_arr, np.array(target_test_df)
            ]

            save_object(

                file_path=self.transformation_config.preprocessor_file_path,
                obj = preprocessor_obj
            )

            logging.info("Data Transformation successfully Completed")
            return(
                train_arr,
                test_arr,
                self.transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)