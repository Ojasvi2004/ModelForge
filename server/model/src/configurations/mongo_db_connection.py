from pymongo import MongoClient
from dotenv import load_dotenv,find_dotenv
import os
import certifi
import logging


from src.exception import MyException
from src.logger import logger
import sys

from src.constants import DATABASE_NAME,COLLECTION_NAME

load_dotenv(override=True)


ca=certifi.where()


class MongoDBClient:
    client=None

    def __init__(self,database_name:str=DATABASE_NAME)->None:
        try:
            print("Database:", DATABASE_NAME)
            print("Collection:", COLLECTION_NAME)
            print("Mongo URL:", os.getenv("MONGODB_URL"))
            if MongoDBClient.client is None:
                mongo_url=os.getenv("MONGODB_URL")
                if mongo_url is None:
                    raise Exception(f"Environment variable MONGODB_URL not found.")
                MongoDBClient.client=MongoClient(mongo_url)

                MongoDBClient.client.admin.command("ping")

            self.client=MongoDBClient.client
            self.database=self.client[DATABASE_NAME]
            self.database_name=database_name
            logger.logging.info("MongoDB Connection Successfull.")
        
        except Exception as e:
            raise MyException(e,sys)
                

