import argparse
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
from typing import Text
import yaml
from src.utils.logs import get_logger


def train_and_save_model(
    train_files: list,
    model_save_path: Path,
    target_column: str,
    model_params: dict,
    logger,
    drop_columns: list,
    cat_features: list
):
    """
    Train and save CatBoost models for a list of training files.
    
    Args:
        train_files (list): List of paths to training files.
        model_save_path (Path): Path to save the trained models.
        target_column (str): Target column name.
        model_params (dict): Parameters for CatBoostClassifier.
        logger: Logger object for logging.
        drop_columns (list): List of columns to drop from training data.
        cat_features (list): List of categorical feature indices.
    """
    for idx, train_file in enumerate(train_files):
        try:
            # Load training data
            logger.info(f"Loading training data from {train_file}")
            data = pd.read_csv(train_file)
            
            if data.empty:
                logger.warning(f"Training file {train_file} is empty. Skipping...")
                continue
            
            # Ensure target column exists
            if target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in {train_file}. Skipping...")
                continue
            
            # Prepare features and target
            X = data.drop(columns=[target_column] + (drop_columns or []), errors='ignore')
            y = data[target_column]
            
            # Train CatBoost model
            logger.info(f"Training CatBoost model on {train_file}")
            model = CatBoostClassifier(**model_params)
            model.fit(X, y, cat_features=cat_features, verbose=0)
            
            # Save the model
            model_file_name = f"catboost_model_{idx}.cbm"
            model_file_path = model_save_path / model_file_name
            model.save_model(model_file_path)
            logger.info(f"Model saved to {model_file_path}")
        
        except Exception as e:
            logger.error(f"Error processing file {train_file}: {e}")

def train_model(config_path: Text) -> None:
    # Load configuration file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Initialize logger
    logger = get_logger("TRAIN", log_level=config['base']['log_level'])
    
    # Get configuration values
    target_column = config['train']['target_column']
    model_params = config['train']['catboost_params']
    drop_columns = config['train']['drop_columns']
    
    # Attestation Data
    train_data_path_attest = Path(config['train_test_split']['train_set']['attestation_data_train'])
    model_save_path_attest = Path(config['model_save_path']['attest_model'])
    model_save_path_attest.mkdir(parents=True, exist_ok=True)
    
    train_files_attest = list(train_data_path_attest.glob("*.csv"))
    logger.info(f"Training CatBoost models for attestation data in {train_data_path_attest}")
    cat_features_attest = config['train']['cat_features_attest']
    train_and_save_model(train_files_attest, model_save_path_attest, target_column, model_params, logger, drop_columns,cat_features=cat_features_attest)
    
    # Movement Data
    train_data_path_movement = Path(config['train_test_split']['train_set']['movement_data_train'])
    model_save_path_movement = Path(config['model_save_path']['movement_model'])
    model_save_path_movement.mkdir(parents=True, exist_ok=True)
    
    train_files_movement = list(train_data_path_movement.glob("*.csv"))
    logger.info(f"Training CatBoost models for movement data in {train_data_path_movement}")
    cat_features_movement = config['train']['cat_features_movement']
    train_and_save_model(train_files_movement, model_save_path_movement, target_column, model_params, logger, drop_columns, cat_features_movement)
    
    # Static Data
    train_data_path_static = Path(config['train_test_split']['train_set']['static_data_train'])
    model_save_path_static = Path(config['model_save_path']['static_model'])
    model_save_path_static.mkdir(parents=True, exist_ok=True)
    
    train_files_static = list(train_data_path_static.glob("*.csv"))
    logger.info(f"Training CatBoost models for static data in {train_data_path_static}")
    cat_features_static = config['train']['cat_features_static']
    train_and_save_model(train_files_static, model_save_path_static, target_column, model_params, logger, drop_columns, cat_features_static)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Train CatBoost models")
    args_parser.add_argument("--config_path", type=str, required=False, default="/Users/macbookpro/Desktop/my_student_retention_exp/params.yaml", help="Path of the config script")
    args = args_parser.parse_args()
    train_model(config_path=args.config_path)
