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
class WalkForwardFoldConfig:
    walk_forward_directory: str = os.path.join(
        training_pipeline_config.artifact_dir, 
        WALK_FORWARD_FOLD_DIR_NAME
    )
    fold_metadata_path: str = os.path.join(
        walk_forward_directory, 
        WALK_FORWARD_FOLD_METADATA_FILE_NAME
    )

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
    
    @dataclass
    class HybridHorizonLossConfig:
        alpha=HYBRID_HORIZON_LOSS_APLHA

    class Schedular:
        mode='min'
        factor=0.5
        patience=2

    @dataclass
    class UnifiedDeepEncoderConfig:
        embed_dim= UNIFIED_DEEP_ENCODER_TRANSFORMER_EMBED_DIM
        hidden_dim= UNIFIED_DEEP_ENCODER_TRANSFORMER_HIDDEN_DIM
        horizons:int=UNIFIED_DEEP_ENCODER_TRANSFORMER_HORIZONS
        dropout: float = UNIFIED_DEEP_ENCODER_TRANSFORMER_DROPOUT

    @dataclass
    class TreeModelsConfig:
        n_estimators=MODEL_TRAINER_TREE_MODELS_N_ESTIMATORS
        max_depth=MODEL_TRAINER_TREE_MODELS_MAX_DEPTH
        learning_rate=MODEL_TRAINER_LEARNING_RATE
        subsample=MODEL_TRAINER_TREE_MODELS_SUBSAMPLE
        colsample_bytree=MODEL_TRAINER_TREE_MODELS_COLSAMPLE_BYTREE
        reg_alpha=MODEL_TRAINER_TREE_MODELS_REG_ALPHA
        res_lambda=MODEL_TRAINER_TREE_MODELS_RES_LAMBDA
        

    input_size: int = MODEL_TRAINER_INPUT_SIZE
    hidden_dim: int =UNIFIED_DEEP_ENCODER_TRANSFORMER_HIDDEN_DIM
    num_layers: int =UNIFIED_DEEP_ENCODER_TRANSFORMER_NUM_LAYERS
    output_size: int = MODEL_TRAINER_OUTPUT_SIZE
    num_heads:int=UNIFIED_DEEP_ENCODER_TRANSFORMER_NUM_HEADS
    model_dim:int=UNIFIED_DEEP_ENCODER_TRANSFORMER_MODEL_DIM
    batch_size: int = MODEL_TRAINER_BATCH_SIZE
    sequence_length: int = MODEL_TRAINER_SEQUENCE_LENGTH
    horizons:int=UNIFIED_DEEP_ENCODER_TRANSFORMER_HORIZONS
    dropout: float = UNIFIED_DEEP_ENCODER_TRANSFORMER_DROPOUT

    optimizer: str = MODEL_TRAINER_OPTIMIZER
    learning_rate: float = MODEL_TRAINER_LEARNING_RATE
    epochs: int = MODEL_TRAINER_EPOCHS

    loss_function: str = MODEL_TRAINER_LOSS_FUNCTION

    patience: int = MODEL_TRAINER_PATIENCE
    gradient_clip: float = MODEL_TRAINER_GRADIENT_CLIP
    saved_model_path:str=MODEL_TRAINER_TRAINED_MODEL_NAME
    device: str = MODEL_TRAINER_DEVICE
    model_trainer_directory:str=MODEL_TRAINER_DIR_NAME


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

