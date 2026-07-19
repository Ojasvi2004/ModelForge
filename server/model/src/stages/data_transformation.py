import sys
import pandas as pd
import numpy as np
from src.entity.artifacts import DataTranformationArtifactEntity,DataIngestionArtifactEntity
from src.entity.config_entity import DataTransformationConfig
from src.constants import *
from src.logger import logger
from src.exception import MyException
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
        
    def get_data_transformer_object(self)->Pipeline:

        logger.logging("Creating Data tranfromation pipeline.")

        try:
            ss=StandardScaler()
            mm=MinMaxScaler()
            logger.logging.info("Scalers initialized: StandardScaler, MinMaxScaler")
            
            num_features=self._schema_config('numerical_columns')

            logger.logging.info(f"Numerical columns :{num_features}")

            preprocessing=ColumnTransformer(
                transformers=[
                    ("Standard Scalar",ss,num_features),
                ],
                remainder="passthrough"
            )

            pipeline=Pipeline(steps=[("preprrocessing",preprocessing)])

            logger.logging.info("Data transformer pipeline created successfully")

            return pipeline

        except Exception as e:
            logger.logging.exception("Error creating data transformer pipeline")
            raise MyException(e,sys)
        
