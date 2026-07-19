import sys
import pandas as pd
import numpy as np
from src.entity.artifacts import DataTranformationArtifactEntity,DataIngestionArtifactEntity
from src.entity.config_entity import DataTransformationConfig
from src.constants import *
from src.logger import logger
from src.exception import MyException
from feature_engineering import FeatureEngineering
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from src.utils.main_utils import read_yaml_file,write_yaml,load_numpy_array_data,load_object



class DataTransformation:

    def __init__(self,data_tranformation_config_:DataTransformationConfig,
                 data_ingestion_artifact_=DataIngestionArtifactEntity,
                 data_tranformation_artifact_=DataTranformationArtifactEntity):
        try:
            self.data_transformation_config=data_tranformation_config_
            self.data_ingestion_artifact=data_ingestion_artifact_
            self.data_transformation_artifact=data_tranformation_config_

            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
            logger.logging.info("Schema loaded successfully")

        except Exception as e:
            raise MyException(e,sys)
        
    
    @staticmethod
    def read_data(file_path:str)->pd.DataFrame:
        try:
            df=pd.read_csv(file_path)
            logger.logging.info(f"Data loaded successfully from {file_path}")
            return df
        except Exception as e:
            raise MyException(e,sys)

    def train_test_split(self,dataframe:pd.DataFrame)->None:
        try:
            train=dataframe[dataframe["date"]<self.data_ingestion_configuration.train_test_split_date].copy()
            test=dataframe[dataframe["date"]>=self.data_ingestion_configuration.train_test_split_date].copy()
            dir_path=os.path.dirname(self.data_ingestion_configuration.data_ingestion_directory)
            os.makedirs(dir_path,exist_ok=True)
            train.to_csv(self.data_ingestion_configuration.train_file_path,index=True,header=True)
            test.to_csv(self.data_ingestion_configuration.test_file_path,index=True,header=True)
            logger.logging.info("Saved the training and testing file")
    
        except Exception as e:
            raise MyException(e,sys)  


    def feature_engineering(self,df:pd.DataFrame)->pd.DataFrame:
        try:
            logger.logging.info("Starting feature engineering on the dataset.")
            sort_columns=self._schema_config["sort_columns"]
            df=df.sort_values(sort_columns)
            logger.logging.info(f"Data sorted by {sort_columns}.")
            logger.logging.info("Engineering technical features and multi-horizon targets...")
            g = df.groupby(self._schema_config["group_by_columns"])
            


            pass
        except Exception as e:
            raise MyException(e,sys)
        
