import sys
import pandas as pd
import numpy as np
from src.entity.artifacts import FeatureEngineeringArtifactEntity,DataIngestionArtifactEntity
from src.entity.config_entity import FeatureEngineeringConfig
from src.constants import *
from src.logger import logger
from src.exception import MyException
from src.utils.main_utils import read_yaml_file


class FeatureEngineering:
    
    def __init__(self,data_ingestion_artifact:DataIngestionArtifactEntity,feature_engineering_config:FeatureEngineeringConfig=None):
        logger.logging.info("Initializing Feature Engineering component.")
        try:
            logger.logging.info("Loading feature engineering configuration.")
            if feature_engineering_config is None:
                feature_engineering_config=FeatureEngineeringConfig()
            self.data_ingestion_artifact=data_ingestion_artifact
            self.feature_engineer_config=feature_engineering_config
            logger.logging.info(f"Readin the scheme configuration file: {SCHEMA_FILE_PATH} ")
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
            logger.logging.info(f"Reading feature configuration from: {FEATURE_CONFIG_PATH}")

            self._feature_config=read_yaml_file(FEATURE_CONFIG_PATH)
            logger.logging.info(f"Loading raw dataset from: {self.data_ingestion_artifact.raw_file_path}")
            self.df=pd.read_csv(self.data_ingestion_artifact.raw_file_path)
            sort_columns=self._schema_config["sort_columns"]
            self.df=self.df.sort_values(sort_columns)
            logger.logging.info(f"Data sorted by {sort_columns}.")
            self.df["date"] = pd.to_datetime(self.df["date"])
            logger.logging.info(f"Raw dataset loaded successfully. Shape: {self.df.shape}")
        except Exception as e:
            raise MyException(e,sys)
        
    @staticmethod
    def calculate_rsi(series, period=14):
        try:
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-8)
            return 100.0 - (100.0 / (1.0 + rs + 1e-8))
        except Exception as e:
            raise MyException(e,sys)

    def engineer_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], int]:
        logger.logging.info("Starting feature engineering.")
        try:   
            logger.logging.info("Engineering technical features and multi-horizon targets...")
            logger.logging.info(f"Input dataframe shape: {df.shape}")
            logger.logging.info("Grouping dataframe by stock symbol.")
            g = df.groupby("symbol")
            df["daily_return"] = g["close"].pct_change()
            df["ma5"] = g["close"].transform(lambda x: x.rolling(5).mean())
            df["ma10"] = g["close"].transform(lambda x: x.rolling(10).mean())
            df["ma20"] = g["close"].transform(lambda x: x.rolling(20).mean())
            df["ma50"] = g["close"].transform(lambda x: x.rolling(50).mean())
            logger.logging.info("Moving average features generated successfully.")
            df["close_to_ma5"] = df["close"] / (df["ma5"] + 1e-8)
            df["close_to_ma20"] = df["close"] / (df["ma20"] + 1e-8)
            df["close_to_ma50"] = df["close"] / (df["ma50"] + 1e-8)
            df["ma5_to_ma20"] = df["ma5"] / (df["ma20"] + 1e-8)
            df["high_to_low"] = df["high"] / (df["low"] + 1e-8)
            df["close_to_open"] = df["close"] / (df["open"] + 1e-8)
            df["volatility"] = g["daily_return"].transform(lambda x: x.rolling(20).std())
            logger.logging.info("Volatility and ATR features generated successfully.")
            prev_close = g["close"].shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs()
            ], axis=1).max(axis=1)
            df["true_range"] = tr
            df["atr14"] = df.groupby("symbol")["true_range"].transform(lambda x: x.rolling(14).mean())
            df["atr_pct"] = df["atr14"] / (df["close"] + 1e-8)
            bb_std20 = g["close"].transform(lambda x: x.rolling(20).std())
            df["bollinger_pctb"] = (df["close"] - df["ma20"]) / (2 * bb_std20 + 1e-8)
            ema12 = g["close"].transform(lambda x: x.ewm(span=12, adjust=False).mean())
            ema26 = g["close"].transform(lambda x: x.ewm(span=26, adjust=False).mean())
            df["macd"] = (ema12 - ema26) / (df["close"] + 1e-8)
            logger.logging.info("MACD indicators generated successfully.")
            df["macd_signal"] = df.groupby("symbol")["macd"].transform(lambda x: x.ewm(span=9, adjust=False).mean())
            df["macd_hist"] = df["macd"] - df["macd_signal"]
            df["roc5"] = g["close"].pct_change(5)
            df["roc10"] = g["close"].pct_change(10)
            df["roc20"] = g["close"].pct_change(20)
            df["vol_ma20"] = g["volume"].transform(lambda x: x.rolling(20).mean())
            df["volume_ratio"] = df["volume"] / (df["vol_ma20"] + 1e-8)
            df["volume_z"] = df.groupby("symbol")["volume"].transform(
                lambda x: (x - x.rolling(20).mean()) / (x.rolling(20).std() + 1e-8)
            )
            logger.logging.info("Momentum and volume-based features generated successfully.")
            df["prev_close"] = g["close"].shift(1)
            df["gap"] = df["open"] / (df["prev_close"] + 1e-8) - 1.0
            df["rsi"] = df.groupby("symbol")["close"].transform(lambda x: self.calculate_rsi(x))
            logger.logging.info("RSI feature generated successfully.")
            market_return = df.groupby("date")["daily_return"].transform("mean")
            df["market_return"] = market_return
            df["relative_return"] = df["daily_return"] - df["market_return"]
            logger.logging.info("Market-relative features generated successfully.")

            for col in ["daily_return", "rsi", "volume_ratio", "roc10"]:
                df[f"{col}_xrank"] = df.groupby("date")[col].rank(pct=True)


            df["dow_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofweek / 5.0)
            df["dow_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofweek / 5.0)

            horizon_days = self._feature_config["prediction_horizons"]

            for name, h in horizon_days.items():
                if h==1:
                    df[name]=g["daily_return"].shift(-1)
                else:
                    df[name] = g["close"].shift(-h) / (df["close"] + 1e-8) - 1.0
            logger.logging.info(f"Prediction targets created: {list(horizon_days.keys())}")
            df = df.dropna()
            stock_lengths = df.groupby("symbol").size()
            logger.logging.info("Filtering stocks with fewer than 150 observations.")
            valid_symbols = stock_lengths[stock_lengths >= 150].index
            df = df[df["symbol"].isin(valid_symbols)]
            logger.logging.info(f"Final engineered dataframe shape: {df.shape}")
            

            return df, list(horizon_days.keys()), max(horizon_days.values())

        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_feature_engineering_artifact(self)->FeatureEngineeringArtifactEntity:
            logger.logging.info("Initiating feature engineering pipeline.")
            try:
                df,target_columns, max_horizon=self.engineer_features(df=self.df)
                logger.logging.info(f"Target columns: {target_columns}")
                logger.logging.info(f"Maximum prediction horizon: {max_horizon}")
                logger.logging.info(
                            f"Saving engineered dataset to: {self.feature_engineer_config.featured_engineering_raw_file_path}"
                                    )
                os.makedirs(self.feature_engineer_config.feature_engineering_directory,exist_ok=True)
                df.to_csv(self.feature_engineer_config.featured_engineering_raw_file_path,index=False,header=True)
                feature_engineering_artifact=FeatureEngineeringArtifactEntity(
                    feature_engineering_raw_file_path=self.feature_engineer_config.featured_engineering_raw_file_path,
                    target_columns=target_columns,
                    max_horizon=max_horizon
                    )
            
                
                logger.logging.info("FeatureEngineeringArtifact created successfully.")
                return feature_engineering_artifact
            except Exception as e:
                raise MyException(e,sys)
            




