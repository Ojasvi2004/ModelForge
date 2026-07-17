import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv,find_dotenv

from src.constants import DATABASE_NAME, COLLECTION_NAME

load_dotenv()
env_path = find_dotenv()
print("ENV FILE:", env_path)

load_dotenv(env_path, override=True)

print("MONGODB_URL:", os.getenv("MONGODB_URL"))

client = MongoClient(os.getenv("MONGODB_URL"))

db = client[DATABASE_NAME]
collection = db[COLLECTION_NAME]

collection.delete_many({})

chunk_size = 10000

for chunk in pd.read_csv(
    "C:/Essesntial/ML/ModelForge/server/model/src/datasets/sp500/sp500_stocks.csv",
    chunksize=chunk_size
):
    collection.insert_many(chunk.to_dict(orient="records"))
    print(f"Uploaded {len(chunk)} rows")

print("Upload Complete")