import os
from dataclasses import dataclass
from datetime import datetime

@dataclass
class DataIngestionArtifactEntity:
    train_file_path:str
    test_file_path:str


@dataclass
class DataTranformationArtifactEntity:
    transformed_object_file_path:str
    tranformed_train_file_path:str
    transformed_test_file_path:str


@dataclass
class ClassificationMetricsArtifactEntity:
    mse_loss:float
    mae_loss:float
    rmse_loss:float
    smoothL1_loss:float
    best_val_loss:float

@dataclass
class ModelTrainerArtifactEntity:
    is_model_accepted:bool
    changed_loss:float
    s3_model_path:str
    trained_model_path:str


@dataclass
class ModelPusherArtifactEntity:
    bucket_name:str
    s3_model_path:str
    
