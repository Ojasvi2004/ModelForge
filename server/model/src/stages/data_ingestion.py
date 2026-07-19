import os
import pandas as pd
import numpy
from src.data.market_mind_data import Market_Mind_Data
from src.constants import *
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifacts import DataIngestionArtifactEntity
from src.exception import MyException
import sys
from src.logger import logger


class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig=DataIngestionConfig()):
        try:
            self.data_ingestion_configuration=data_ingestion_config
        
        except Exception as e:
            raise MyException(e,sys)
        
    
    def import_raw_data(self)->pd.DataFrame:
        try:
            logger.logging.info("Importing data from mongoDB")
            my_data=Market_Mind_Data()
            dataframe=my_data.export_collection_to_dataframe(self.data_ingestion_configuration.collection_name)
            raw_data_path=self.data_ingestion_configuration.raw_data_file_path
            dir_path=os.path.dirname(raw_data_path)
            os.makedirs(dir_path,exist_ok=True)
            logger.logging.info(f"Saving the raw data in {dir_path}")
            dataframe.to_csv(raw_data_path,index=False,header=True)
            return dataframe
        except Exception as e:
            raise MyException(e,sys)
        
    # def train_test_split(self,dataframe:pd.DataFrame)->None:
    #     try:
    #         train=dataframe[dataframe["date"]<self.data_ingestion_configuration.train_test_split_date].copy()
    #         test=dataframe[dataframe["date"]>=self.data_ingestion_configuration.train_test_split_date].copy()
    #         dir_path=os.path.dirname(self.data_ingestion_configuration.data_ingestion_directory)
    #         os.makedirs(dir_path,exist_ok=True)
    #         train.to_csv(self.data_ingestion_configuration.train_file_path,index=True,header=True)
    #         test.to_csv(self.data_ingestion_configuration.test_file_path,index=True,header=True)
    #         logger.logging.info("Saved the training and testing file")
    
    #     except Exception as e:
    #         raise MyException(e,sys)
        

    def initiate_data_ingestion(self)->DataIngestionArtifactEntity:
        try:
            dataframe=self.import_raw_data()
            logger.logging.info("Got the data from MongoDB")
            data_ingestion_artifact=DataIngestionArtifactEntity(
                raw_file_path=self.data_ingestion_configuration.raw_data_file_path
            )
            logger.logging.info(f"Data ingestion artifact created {data_ingestion_artifact}")

            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
        


        
    