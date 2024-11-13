import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Text
import yaml
from src.utils.logs import get_logger
from pathlib import Path

def data_split(config_path: Text) -> None:
    # Load configuration file
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    # Initialize logger
    logger = get_logger("DATA SPLIT", log_level=config['base']['log_level'])
    
    # Get configuration values
    random_state = config['base']['random_state']
    test_size = config['base']['test_size']
    
    logger.info("Configuration loaded successfully")
    
    # Attestation features processing
    attest_feature_path = Path(config['featurize']['attestation_features'])
    if not attest_feature_path.exists():
        logger.error(f"Attestation feature path {attest_feature_path} does not exist.")
        return
    
    attest_feature_list = list(attest_feature_path.glob('*.csv'))
    if not attest_feature_list:
        logger.error("No attestation feature files found.")
        return
    
    logger.info(f"Found {len(attest_feature_list)} attestation feature files.")
    all_attest = []
    for file in attest_feature_list:
        try:
            df = pd.read_csv(file)
            if df.empty:
                logger.warning(f"File {file} is empty. Skipping...")
                continue
            all_attest.append(df)
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_attest)} attestation datasets for splitting.")
    
    # Define paths for train/test splits
    attest_train_base_path = Path(config['train_test_split']['train_set']['attestation_data_train'])
    attest_test_base_path = Path(config['train_test_split']['test_set']['attestation_data_test'])
    attest_train_base_path.mkdir(parents=True, exist_ok=True)
    attest_test_base_path.mkdir(parents=True, exist_ok=True)

    # Perform train-test split for attestation features
    for idx, dataset in enumerate(all_attest):
        if len(dataset) < 2:
            logger.warning(f"Dataset {idx} has insufficient rows for splitting. Skipping...")
            continue
        
        try:
            train_set, test_set = train_test_split(dataset, test_size=test_size, random_state=random_state)
            train_path = attest_train_base_path / f"attest_train_set_{idx}.csv"
            test_path = attest_test_base_path / f"attest_test_set_{idx}.csv"
            
            train_set.to_csv(train_path, index=False)
            test_set.to_csv(test_path, index=False)
            
            logger.info(f"Split completed for dataset {idx}. Train: {len(train_set)}, Test: {len(test_set)}")
        except Exception as e:
            logger.error(f"Error splitting dataset {idx}: {e}")
    
    # Repeat similar steps for movement features
    movement_feature_path = Path(config['featurize']['movement_features'])
    if not movement_feature_path.exists():
        logger.error(f"Movement feature path {movement_feature_path} does not exist.")
        return
    
    movement_feature_list = list(movement_feature_path.glob("*.csv"))
    if not movement_feature_list:
        logger.error("No movement feature files found.")
        return
    
    logger.info(f"Found {len(movement_feature_list)} movement feature files.")
    all_movement = []
    for file in movement_feature_list:
        try:
            df = pd.read_csv(file)
            if df.empty:
                logger.warning(f"File {file} is empty. Skipping...")
                continue
            all_movement.append(df)
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            continue
    
    movement_train_base_path = Path(config['train_test_split']['train_set']['movement_data_train'])
    movement_test_base_path = Path(config['train_test_split']['test_set']['movement_data_test'])
    movement_train_base_path.mkdir(parents=True, exist_ok=True)
    movement_test_base_path.mkdir(parents=True, exist_ok=True)

    for i, data in enumerate(all_movement):
        if len(data) < 2:
            logger.warning(f"Dataset {i} has insufficient rows for splitting. Skipping...")
            continue
        
        try:
            train_set_movement, test_set_movement = train_test_split(data, test_size=test_size, random_state=random_state)
            movement_train_path = movement_train_base_path / f"movement_train_set_{i}.csv"
            movement_test_path = movement_test_base_path / f"movement_test_set_{i}.csv"
            
            train_set_movement.to_csv(movement_train_path, index=False)
            test_set_movement.to_csv(movement_test_path, index=False)
            
            logger.info(f"Split completed for movement dataset {i}. Train: {len(train_set_movement)}, Test: {len(test_set_movement)}")
        except Exception as e:
            logger.error(f"Error splitting movement dataset {i}: {e}")
    
    # Repeat similar steps for static features
    static_features_path = Path(config['featurize']['static_features'])
    if not static_features_path.exists():
        logger.error(f"Static feature path {static_features_path} does not exist.")
        return
    
    static_feature_list = list(static_features_path.glob("*.csv"))
    if not static_feature_list:
        logger.error("No static feature files found.")
        return
    
    logger.info(f"Found {len(static_feature_list)} static feature files.")
    all_static = []
    for file in static_feature_list:
        try:
            df = pd.read_csv(file)
            if df.empty:
                logger.warning(f"File {file} is empty. Skipping...")
                continue
            all_static.append(df)
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            continue
    
    static_train_base_path = Path(config['train_test_split']['train_set']['static_data_train'])
    static_test_base_path = Path(config['train_test_split']['test_set']['static_data_test'])
    static_train_base_path.mkdir(parents=True, exist_ok=True)
    static_test_base_path.mkdir(parents=True, exist_ok=True)

    for i, static in enumerate(all_static):
        if len(static) < 2:
            logger.warning(f"Dataset {i} has insufficient rows for splitting. Skipping...")
            continue
        
        try:
            train_set, test_set = train_test_split(static, test_size=test_size, random_state=random_state)
            static_path_train = static_train_base_path / f"static_train_set_{i}.csv"
            static_path_test = static_test_base_path / f"static_test_set_{i}.csv"
            
            train_set.to_csv(static_path_train, index=False)
            test_set.to_csv(static_path_test, index=False)
            
            logger.info(f"Split completed for static dataset {i}. Train: {len(train_set)}, Test: {len(test_set)}")
        except Exception as e:
            logger.error(f"Error splitting static dataset {i}: {e}")
    
    logger.info("All features have been successfully split into train and test sets.")

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser(description="Train and test set split")
    args_parser.add_argument("--config_path", type=str, required=False, default="/Users/macbookpro/Desktop/my_student_retention_exp/params.yaml", help="Path of the config script")
    args = args_parser.parse_args()
    data_split(config_path=args.config_path)
