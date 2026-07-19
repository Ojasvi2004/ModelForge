from src.stages.data_ingestion import DataIngestion
from src.stages.feature_engineering import FeatureEngineering
from src.stages.data_validation import DataValidation
from src.logger import logger


if __name__=="__main__":
    test_data_ingestion_object=DataIngestion()
    data_ingestion_artifact=test_data_ingestion_object.initiate_data_ingestion()
    test_data_validation_object=DataValidation(data_ingestion_artifact=data_ingestion_artifact)
    data_validation_artifact=test_data_validation_object.initiate_data_validation()
    if data_validation_artifact.validation_status==False:
        logger.logging.info("Data validation module return false.")
        raise InterruptedError 
    else:
        test_feature_ingestion_object=FeatureEngineering(data_ingestion_artifact=data_ingestion_artifact)
        test_feature_ingestion_object.initiate_feature_engineering_artifact()
