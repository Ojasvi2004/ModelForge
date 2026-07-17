import pandas as pd
import sys
import numpy as np
from typing import Optional


from src.exception import MyException
from src.logger import logger
from src.constants import DATABASE_NAME,COLLECTION_NAME
from src.configurations.mongo_db_connection import MongoDBClient


class Market_Mind_Data:
    """
    Class to import the data from mongodb and export it to pandas dataframe
    """
    def __init__(self):
        try:
            self.mongoClient=MongoDBClient(database_name=DATABASE_NAME)

        
        except Exception as e:
            raise MyException(e,sys)
        
    def export_collection_to_dataframe(self,collection_name:str,database_name:Optional[str]=None)->pd.DataFrame:

        try:
            if database_name is None:
                collection=self.mongoClient.database[collection_name]
            else:
                collection=self.mongoClient[database_name][collection_name]
            
            print("Fetching Data from MongoDB")

            df=pd.DataFrame(list(collection.find().limit(10)))

            print(f"{len(df)} Data records fetched")

            if "id" in df.columns.to_list():
                df=df.drop(columns=['id'],axis=1)
            df.replace({"na":np.nan},inplace=True)
            return df
        
        except Exception as e:
            raise MyException(e,sys)
