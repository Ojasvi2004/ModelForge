import os
from datetime import date


DATABASE_NAME = "Market_mind"
COLLECTION_NAME = "SP500_Data"


PIPELINE_NAME: str = ""
ARTIFACT_DIR: str = "artifact"

MODEL_FILE_NAME = "model.pth"

CURRENT_YEAR = date.today().year
PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"

FILE_NAME: str = "data.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")


AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "ProjStockPred_Data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_RAW_DATA_DIR: str = "raw_data"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_END_DATE="2022-01-01"
DATA_INGESTION_VALIDATION_END_DATE="2023-01-01"

"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

"""
MODEL TRAINER related constant start with MODEL_TRAINER var name
"""
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "best_model.pth"
MODEL_TRAINER_INPUT_SIZE:int=8
MODEL_TRAINER_HIDDEN_SIZE:int=128
MODEL_TRAINER_NUM_LAYERS:int=2
MODEL_TRAINER_OUTPUT_SIZE:int=4
MODEL_TRAINER_BATCH_SIZE:int=512
MODEL_TRAINER_SEQUENCE_LENGTH:int=60
MODEL_TRAINER_DROPOUT:float=0.2
MODEL_TRAINER_OPTIMIZER:str="Adam"
MODEL_TRAINER_EPOCHS:int=20
MODEL_TRAINER_LEARNING_RATE:float=1e-3
MODEL_TRAINER_LOSS_FUNCTION:str='SmoothL1Loss'
MODEL_TRAINER_PATIENCE:int=5
MODEL_TRAINER_DEVICE:str="cuda"
MODEL_TRAINER_GRADIENT_CLIP:float=1.0



"""
MODEL Evaluation related constants
"""
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE: float = 0.02
MODEL_BUCKET_NAME = "my-model-mlopsproj2004"
MODEL_PUSHER_S3_KEY = "model-registry"


APP_HOST = "0.0.0.0"
APP_PORT = 5000