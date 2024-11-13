import argparse
import os
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from typing import Text
import yaml
from src.utils.logs import get_logger

def train_model(config_path: Text) -> None:
    # Load configuration file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Initialize logger
    logger = get_logger("TRAIN", log_level=config['base']['log_level'])
    
    # Get configuration values
    random_state = config['base']['random_state']
    train_data_path_attest = Path(config['train_test_split']['train_set']['attestation_data_train'])
    model_save_path_attest = Path(config['model_save_path']['attest_model'])
    model_save_path_attest.mkdir(parents=True, exist_ok=True)
    # for movement we do the same
    train_data_path_movement = Path(config['train_test_split']['train_set']['movement_data_train'])
    model_save_path_movement = Path(config['model_save_path']['movement_model'])
    model_save_path_movement.mkdir(parents=True, exist_ok=True)
    # for the static data
    train_data_path_static = Path(config['train_test_split']['train_set']['static_data_train'])
    model_save_path_static = Path(config['model_save_path']['static_model'])
    model_save_path_static.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Training CatBoost models for files in {train_data_path_attest}")
    
    # List all train attest files
    train_files_attest = list(train_data_path_attest.glob("*.csv"))
    if not train_files_attest:
        logger.error(f"No training files found in {train_data_path_attest}")
        return
    
    logger.info(f"Found {len(train_files_attest)} training files.")
    
    for idx, train_file in enumerate(train_files_attest):
        try:
            # Load the training data
            logger.info(f"Loading training data from {train_file}")
            data = pd.read_csv(train_file)
            if data.empty:
                logger.warning(f"Training file {train_file} is empty. Skipping...")
                continue
            
            # Separate features and target
            target_column = config['train']['target_column']
            if target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in {train_file}. Skipping...")
                continue
            
            X = data.drop(columns=[target_column, 'end_date', 'student_id', 'id'], axis=1)
            y = data[target_column]
            
            
            # Train CatBoost model
            logger.info(f"Training CatBoost model on {train_file}")
            model_params = config['train']['catboost_params']
            model = CatBoostClassifier(**model_params)
            model.fit(X, y, verbose=0)
            
            # Save the model
            model_file_name = f"catboost_model_attest_{idx}.cbm"
            model_file_path = model_save_path_attest / model_file_name
            model.save_model(model_file_path)
            logger.info(f"Model saved to {model_file_path}")
        
        except Exception as e:
            logger.error(f"Error processing file {train_file}: {e}")
    
    logger.info("Training completed for all files.")
    
    
    logger.info(f"Training CatBoost models for files in {train_data_path_movement}")
    
    # List all train movement files
    train_files_movement = list(train_data_path_movement.glob("*.csv"))
    if not train_files_movement:
        logger.error(f"No training files found in {train_data_path_movement}")
        return
    
    logger.info(f"Found {len(train_files_movement)} training files.")
    
    for idx, train_file in enumerate(train_files_movement):
        try:
            # Load the training data
            logger.info(f"Loading training data from {train_file}")
            data = pd.read_csv(train_file)
            if data.empty:
                logger.warning(f"Training file {train_file} is empty. Skipping...")
                continue
            
            # Separate features and target
            target_column = config['train']['target_column']
            if target_column not in data.columns:
                logger.error(f"Target column '{target_column}' not found in {train_file}. Skipping...")
                continue
            
            X = data.drop(columns=[target_column, 'end_date', 'student_id', 'id'], axis=1)
            y = data[target_column]
            
            
            # Train CatBoost model
            logger.info(f"Training CatBoost model on {train_file}")
            model_params = config['train']['catboost_params']
            model = CatBoostClassifier(**model_params)
            model.fit(X, y, verbose=0)
            
            # Save the model
            model_file_name = f"catboost_model_movement_{idx}.cbm"
            model_file_path = model_save_path_movement / model_file_name
            model.save_model(model_file_path)
            logger.info(f"Model saved to {model_file_path}")
        
        except Exception as e:
            logger.error(f"Error processing file {train_file}: {e}")
    
    logger.info("Training completed for all files.")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Train CatBoost models")
    args_parser.add_argument("--config_path", type=str, required=False, default="/Users/macbookpro/Desktop/my_student_retention_exp/params.yaml", help="Path of the config script")
    args = args_parser.parse_args()
    train_model(config_path=args.config_path)
