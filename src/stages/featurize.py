import argparse
import pandas as pd
from typing import Text
import yaml
from src.utils.logs import get_logger
from modules.attestation import Attestation
from modules.movement import StudentAnalysis
from modules.static import Static
from pathlib import Path


def featurize(config_path: Text) -> None:
    """Create new features.
    Args:
        config_path {Text}: path to config
    """
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    logger = get_logger('FEATURIZE', log_level=config['base']['log_level'])

    logger.info('Load raw attestation data')
    attest_data_path = Path(config['data_load']['attest_data_csv'])
    # target_data_path = Path(config['data_load']['targets_data_csv'])
    target_data_path = Path("/Users/macbookpro/Desktop/my_student_retention_exp/data/raw/targets_data")

    
    # loop through the data in the targets path use each to combine with attestation
    target_list = list(target_data_path.glob("*.csv"))
    
    # Assume `config` is loaded with the paths from params.yaml
    attest_base_path = Path(config['featurize']['attestation_features'])

    # Ensure the directory exists
    attest_base_path.mkdir(parents=True, exist_ok=True)

    # Process and save each extracted attestation feature set
    for i, target in enumerate(target_list):
        attestation = Attestation(attest_data_path, target)
        attest_features = attestation.extract_features(target)
        
        # Dynamically construct the feature path using the base path and index
        feature_file_path = attest_base_path / f"attest_features_{i}.csv"
        
        # Save the extracted features to the constructed path
        attest_features.to_csv(feature_file_path)
        print(f"Saved attestation features to {feature_file_path}")
    logger.info("Attestation Features Successfully Loaded")

    logger.info("Load movement data")
    movement_data_path = Path(config['data_load']['movement_data_csv'])
    anonymous_data_path = Path(config['data_load']['anonymous_data_csv'])/"СоответствияИД.xlsx"
    # for the movement path
    movement_base_path = Path(config['featurize']['movement_features'])
    
    # extract features for each semester of movement data
    for i, target in enumerate(target_list):
        movement = StudentAnalysis(movement_data_path, anonymous_data_path, target)
        movement_features = movement.extract_features(target)
        
        # construct path to save features
        movement_features_path = movement_base_path / f"movement_features_{i}.csv"
        
        # save the features to the path
        movement_features.to_csv(movement_features_path)
        print(f"Saved movement features to {movement_features_path}")
    logger.info("Movements Features Extracted and Saved")
    
    logger.info("Extracting Features for static data")
    # for static path
    static_data_path = Path(config['data_load']['static_data_csv'])
    # for static features base path
    static_base_path = Path(config['featurize']['static_features'])
    
    # extract features for static data
    for i, target in enumerate(target_list):
        static = Static(static_data_path, target)
        static_features = static.get_features(target)
        
        static_features_path = static_base_path / f"static_features_{i}.csv"
        
        # save the  features to the path specified
        static_features.to_csv(static_features_path)
        print(f"Saved static features to {static_features_path}")
    logger.info("Static Features Extracted and Saved")

            
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser(description="Paths and Params")
    
    # Add an argument for config_path
     # Set the config_path as optional with a default
    args_parser.add_argument('--config_path', type=str, required=False, default="/Users/macbookpro/Desktop/my_student_retention_exp/params.yaml", help="path to config file")

    args_parser.add_argument('--attestation_data_path', type=str,
                             default='data/raw/attest_data', required=False, help="path to input attestation data")
    
    args_parser.add_argument('--movement_data_path', type=str, required=False,
                             default="data/raw/movement_data", help="path to movement data")
    
    args_parser.add_argument('--static_data_path', type=str, required=False,
                             default="data/raw/static_data", help="path to static data")
    
    args_parser.add_argument('--target_data_path', type=str, required=False,
                             default="data/raw/targets_data", help='path to target data')
    
    args_parser.add_argument('--anonymous_data_path', type=str, required=False,
                             default="data/raw/anonymous_data", help='path to anonymous data')
    
    args = args_parser.parse_args()

    # Use args.config_path instead of args.config
    featurize(config_path=args.config_path)
