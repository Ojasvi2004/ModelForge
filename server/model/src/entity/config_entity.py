from dataclasses import dataclass
from datetime import datetime
import os
from src.constants import *

TIMESTAMP=datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
@dataclass
class TrainingPipelineConfig:
    pipeline_name:str=PIPELINE_NAME
    artifact_dir:str=os.path.join(ARTIFACT_DIR,TIMESTAMP)
    timestamp:str=TIMESTAMP


training_pipeline_config:TrainingPipelineConfig=TrainingPipelineConfig()

@dataclass
class DataIngestionConfig:
    data_ingestion_directory=os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    raw_data_file_path=os.path.join(data_ingestion_directory,DATA_INGESTION_RAW_DATA_DIR)
    collection_name=COLLECTION_NAME


@dataclass
class DataTransformationConfig:
    data_transformation_directory=os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_data_path=os.path.join(data_transformation_directory,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TRAIN_FILE_NAME.replace(".csv",".npy"))
    transformed_test_data_path=os.path.join(data_transformation_directory,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_FILE_NAME.replace(".csv",".npy"))
    tranformed_object_path=os.path.join(data_transformation_directory,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCSSING_OBJECT_FILE_NAME)


@dataclass
class DataValidationConfig:
    data_validation_dir=os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    validation_report_file_path=os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)
    

@dataclass
class FeatureEngineeringConfig:
    feature_engineering_directory=os.path.join(training_pipeline_config.artifact_dir,FEATURE_ENGINEERING_DIR_NAME)
    featured_engineering_raw_file_path=os.path.join(feature_engineering_directory,FEATURE_ENGINEERING_RAW_FILE_NAME)

from dataclasses import dataclass
import os

@dataclass
class ModelTrainerConfig:

    model_trainer_dir: str = os.path.join(
        training_pipeline_config.artifact_dir,
        MODEL_TRAINER_DIR_NAME
    )

    trained_model_file_path: str = os.path.join(
        model_trainer_dir,
        MODEL_TRAINER_TRAINED_MODEL_DIR,
        MODEL_TRAINER_TRAINED_MODEL_NAME
    )

    input_size: int = MODEL_TRAINER_INPUT_SIZE
    hidden_size: int = MODEL_TRAINER_HIDDEN_SIZE
    num_layers: int = MODEL_TRAINER_NUM_LAYERS
    output_size: int = MODEL_TRAINER_OUTPUT_SIZE

    batch_size: int = MODEL_TRAINER_BATCH_SIZE
    sequence_length: int = MODEL_TRAINER_SEQUENCE_LENGTH

    dropout: float = MODEL_TRAINER_DROPOUT

    optimizer: str = MODEL_TRAINER_OPTIMIZER
    learning_rate: float = MODEL_TRAINER_LEARNING_RATE
    epochs: int = MODEL_TRAINER_EPOCHS

    loss_function: str = MODEL_TRAINER_LOSS_FUNCTION

    patience: int = MODEL_TRAINER_PATIENCE
    gradient_clip: float = MODEL_TRAINER_GRADIENT_CLIP

    device: str = MODEL_TRAINER_DEVICE


@dataclass
class ModelEvaluationConfig:
    changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME  


@dataclass
class ModelPusherConfig:
    bucket_name: str = MODEL_BUCKET_NAME
    s3_model_key_path: str = MODEL_FILE_NAME


@dataclass
class StockPredictorConfig:
    model_file_path: str = MODEL_FILE_NAME
    model_bucket_name: str = MODEL_BUCKET_NAME

