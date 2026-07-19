import sys
import pandas as pd
import numpy as np
from src.entity.artifacts import DataValidationArtifact,DataIngestionArtifactEntity
from src.entity.config_entity import DataValidationConfig
from src.constants import *
from src.logger import logger
from src.exception import MyException
from src.utils.main_utils import read_yaml_file


class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifactEntity,
                 data_validation_config:DataValidationConfig=None):
        try:
            if data_validation_config is None:
                data_validation_config=DataValidationConfig()
            
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self._schema_config=read_yaml_file(file_path=SCHEMA_FILE_PATH)
            
        except Exception as e:
            raise MyException(e,sys)
        
    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        
        try:
            status=len(dataframe.columns)==self._schema_config['numerical_columns'] + self._schema_config['categorical_columns']
            logger.logger.logging.info(f"Is required column present: {status}")
            return status
            
        except Exception as e:
            raise MyException(e,sys)
        
    def is_column_exist(self, df: pd.DataFrame) -> bool:
       
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if len(missing_numerical_columns)>0:
                logger.logging.info(f"Missing numerical column: {missing_numerical_columns}")


            for column in self._schema_config["categorical_columns"]:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)

            if len(missing_categorical_columns)>0:
                logger.logging.info(f"Missing categorical column: {missing_categorical_columns}")
            correct_data=True    


            if len(missing_categorical_columns)>0 or len(missing_numerical_columns)>0:
                correct_data=False
            return correct_data
        except Exception as e:
            raise MyException(e, sys) from e

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logger.logging.info("Starting data validation")
            df=self.read_data(self.data_ingestion_artifact.raw_file_path)
            validation_error=""
            expected_columns=self._schema_config['columns']
            missing_columns=[col for col in expected_columns  if col not in df.columns ]
            if missing_columns:
                validation_error=validation_error+f"Columns missing {missing_columns} from data."
            else:
                logger.logging.info("All required columns are present")
            correct_data=False
            if len(validation_error)==0:
                correct_data=True
            
            data_validation_artifact=DataValidationArtifact(
                validation_status=correct_data,
                message=validation_error
            )

            return data_validation_artifact


        except Exception as e:
            raise MyException(e,sys)


        