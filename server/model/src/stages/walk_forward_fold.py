# src/components/walk_forward_fold.py
import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any
from dataclasses import dataclass

from src.entity.artifacts import WalkForwardFoldArtifactEntity, FeatureEngineeringArtifactEntity
from src.entity.config_entity import WalkForwardFoldConfig
from src.constants import *
from src.logger import logger
from src.exception import MyException
from src.utils.main_utils import read_yaml_file

@dataclass
class WalkForwardFoldMetadata:
    name: str
    val_year: int
    test_year: int
    train_end: str
    val_end: str
    test_end: str


class WalkForwardFold:

    def __init__(self, 
                 feature_engineering_artifact: FeatureEngineeringArtifactEntity,
                 fold_config: WalkForwardFoldConfig=None):
        try:
            if fold_config is None:
                fold_config = WalkForwardFoldConfig()

            logger.logging.info("Initializing WalkForwardFold component for Walk-Forward Validation.")
            self.fold_config = fold_config
            self.feature_engineering_artifact = feature_engineering_artifact
            
            self._schema_config = read_yaml_file(file_path=FEATURE_CONFIG_PATH)
            
            self.targets = self.feature_engineering_artifact.target_columns
            self.max_horizon = self.feature_engineering_artifact.max_horizon
            
            self.seq_features = self._schema_config["sequence_features"]
            self.curr_features = self._schema_config["current_features"]
            self.seq_len = SEQ_LEN
            
            logger.logging.info("WalkForwardFold component is initialized.")
            
        except Exception as e:
            raise MyException(e, sys)

    def build_walkforward_folds(self, df: pd.DataFrame, min_train_years: int = 2) -> List[Dict[str, Any]]:
        """
        Dynamically calculates boundary timelines for expanding-window walk-forward validation.
        If dataset history is short, it automatically falls back to month-based splitting to prevent lookahead bias.
        """
        logger.logging.info("Constructing walk-forward validation folds.")
        try:
            years = sorted(df["date"].dt.year.unique())
            
           
            if len(years) >= min_train_years + 2:
                logger.logging.info("Sufficient calendar history found. Using standard year-based splits.")
                min_year = years[0]
                max_year = years[-1]
                first_val_year = min_year + min_train_years

                folds = []
                val_year = first_val_year
                while val_year + 1 <= max_year:
                    fold_name = f"train_lt_{val_year}_val_{val_year}_test_{val_year+1}"
                    folds.append({
                        "name": fold_name,
                        "val_year": int(val_year),
                        "test_year": int(val_year + 1),
                        "train_end": f"{val_year}-01-01",
                        "val_end": f"{val_year+1}-01-01",
                        "test_end": f"{val_year+2}-01-01",
                    })
                    val_year += 1
                return folds

            
            logger.logging.warning(
                f"Dataset spans only {len(years)} years. Year-based walk-forward requires at least {min_train_years + 2} years. "
                "Automatically falling back to month-based splits to protect validation integrity."
            )
            
            start_date = df["date"].min()
            end_date = df["date"].max()
            
           
            month_starts = pd.date_range(start=start_date, end=end_date, freq="MS")
            total_months = len(month_starts)
            
            if total_months < 6:
                raise ValueError(
                    f"Dataset spans only {total_months} months. Walk-forward validation requires "
                    f"at least 6 months of historical daily/weekly data."
                )
            
          
            test_size = max(1, min(3, total_months // 8))
            val_size = max(1, min(3, total_months // 8))
            min_train_size = max(3, total_months - (val_size + test_size) - 6) 
            
        
            if min_train_size + val_size + test_size > total_months:
                min_train_size = int(total_months * 0.6)
                val_size = int(total_months * 0.2)
                test_size = total_months - min_train_size - val_size

            logger.logging.info(
                f"Configured Monthly Splits: Min Train = {min_train_size}M, Val = {val_size}M, Test = {test_size}M. "
                f"Total dataset: {total_months} months."
            )

            folds = []
            step_size = max(1, test_size)
            start_val_end_idx = min_train_size + val_size
            
            for val_end_idx in range(start_val_end_idx, total_months - test_size + 1, step_size):
                train_end_dt = month_starts[val_end_idx - val_size]
                val_end_dt = month_starts[val_end_idx]
                
            
                test_end_idx = val_end_idx + test_size
                if test_end_idx >= total_months:
                    test_end_dt = end_date
                else:
                    test_end_dt = month_starts[test_end_idx]
                
                fold_name = f"train_lt_{train_end_dt.strftime('%Y%m')}_val_{val_end_dt.strftime('%Y%m')}_test_{test_end_dt.strftime('%Y%m')}"
                
                folds.append({
                    "name": fold_name,
                    "val_year": int(val_end_dt.year),
                    "test_year": int(test_end_dt.year),
                    "train_end": train_end_dt.strftime("%Y-%m-%d"),
                    "val_end": val_end_dt.strftime("%Y-%m-%d"),
                    "test_end": test_end_dt.strftime("%Y-%m-%d")
                })

            if not folds:
                raise ValueError("No monthly splits could be structured. Check calendar bounds in raw database.")

            return folds
            
        except Exception as e:
            raise MyException(e, sys)

    def initiate_walk_forward_folding(self) -> WalkForwardFoldArtifactEntity:
        logger.logging.info("Starting walk-forward folding pipeline.")
        try:
           
            raw_engineered_path = self.feature_engineering_artifact.feature_engineering_raw_file_path
            df = pd.read_csv(raw_engineered_path)
            df["date"] = pd.to_datetime(df["date"])
            logger.logging.info("Got raw_engineered data in walk_forward_folding.")
            
            sort_columns = self._schema_config.get("sort_columns", ["symbol", "date"])
            df = df.sort_values(sort_columns).reset_index(drop=True)
            
            os.makedirs(self.fold_config.walk_forward_directory, exist_ok=True)
            sorted_data_path = os.path.join(
                self.fold_config.walk_forward_directory, 
                "engineered_sorted_data.csv"
            )
            df.to_csv(sorted_data_path, index=False)
            
            min_train_years = self._schema_config.get("min_train_years", 2)
            folds_meta = self.build_walkforward_folds(df=df, min_train_years=min_train_years)
            
            fold_config_path = self.fold_config.fold_metadata_path
            os.makedirs(os.path.dirname(fold_config_path), exist_ok=True)
            
            with open(fold_config_path, "w") as f:
                json.dump({
                    "seq_features": self.seq_features,
                    "curr_features": self.curr_features,
                    "targets": self.targets,
                    "seq_len": self.seq_len,
                    "max_horizon": self.max_horizon,
                    "folds": folds_meta
                }, f, indent=4)
                
            logger.logging.info(f"Walk-forward configurations successfully exported to {fold_config_path}")
            
            return WalkForwardFoldArtifactEntity(
                walk_forward_dir=sorted_data_path,
                fold_metadata_file_name=fold_config_path
            )
            
        except Exception as e:
            raise MyException(e, sys)