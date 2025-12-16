"""
NAB Dataset Loader

This module provides utilities to download and load the Numenta Anomaly Benchmark (NAB)
dataset for time-series anomaly detection.

Author: AI/ML Engineer Candidate
"""

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import json


class NABDataLoader:
    """
    Loader for Numenta Anomaly Benchmark (NAB) dataset.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize NAB data loader.
        
        Parameters:
        -----------
        data_dir : str
            Directory to store downloaded NAB data
        """
        self.data_dir = Path(data_dir)
        self.nab_dir = self.data_dir / "NAB"
        self.data_path = self.nab_dir / "data"
        self.labels_path = self.nab_dir / "labels"
        
    def download_nab_dataset(self):
        """
        Download NAB dataset from GitHub if not already present.
        """
        if self.nab_dir.exists():
            print(f"NAB dataset already exists at {self.nab_dir}")
            return
            
        print("Downloading NAB dataset from GitHub...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download NAB dataset
        nab_url = "https://github.com/numenta/NAB/archive/refs/heads/master.zip"
        zip_path = self.data_dir / "NAB-master.zip"
        
        try:
            response = requests.get(nab_url, stream=True)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print("Extracting NAB dataset...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.data_dir)
            
            # Rename extracted folder
            extracted_dir = self.data_dir / "NAB-master"
            
            # Handle Windows path issues with rename
            import shutil
            if self.nab_dir.exists():
                shutil.rmtree(self.nab_dir)
            shutil.move(str(extracted_dir), str(self.nab_dir))
            
            # Clean up zip file
            zip_path.unlink()
            
            print(f"NAB dataset downloaded and extracted to {self.nab_dir}")
            
        except Exception as e:
            print(f"Error downloading NAB dataset: {e}")
            print("Please manually download from: https://github.com/numenta/NAB")
            raise
    
    def load_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Load a single NAB CSV file.
        
        Parameters:
        -----------
        file_path : str
            Relative path to CSV file within NAB data directory
            
        Returns:
        --------
        pd.DataFrame
            Time-series data with timestamp index
        """
        full_path = self.data_path / file_path
        
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        
        df = pd.read_csv(full_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def load_labels(self, file_path: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
        """
        Load anomaly labels for a specific data file.
        
        Parameters:
        -----------
        file_path : str
            Relative path to CSV file within NAB data directory
            
        Returns:
        --------
        List[Tuple[pd.Timestamp, pd.Timestamp]]
            List of anomaly window start and end timestamps
        """
        # NAB labels are in combined_windows.json
        labels_file = self.labels_path / "combined_windows.json"
        
        if not labels_file.exists():
            print(f"Warning: Labels file not found at {labels_file}")
            return []
        
        with open(labels_file, 'r') as f:
            all_labels = json.load(f)
        
        # Construct the key (format: "realAWSCloudwatch/filename.csv")
        file_key = f"realAWSCloudwatch/{Path(file_path).name}"
        
        if file_key not in all_labels:
            print(f"Warning: No labels found for {file_key}")
            return []
        
        anomaly_windows = []
        for window in all_labels[file_key]:
            start = pd.to_datetime(window[0])
            end = pd.to_datetime(window[1])
            anomaly_windows.append((start, end))
        
        return anomaly_windows
    
    def create_anomaly_labels_column(self, df: pd.DataFrame, 
                                     anomaly_windows: List[Tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
        """
        Create binary anomaly labels for the dataframe.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Time-series dataframe with timestamp index
        anomaly_windows : List[Tuple[pd.Timestamp, pd.Timestamp]]
            List of anomaly window start and end timestamps
            
        Returns:
        --------
        pd.Series
            Binary labels (1 for anomaly, 0 for normal)
        """
        labels = pd.Series(0, index=df.index)
        
        for start, end in anomaly_windows:
            labels[(df.index >= start) & (df.index <= end)] = 1
        
        return labels
    
    def load_multivariate_dataset(self, file_paths: List[str]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load multiple NAB files and combine into multivariate dataset.
        
        Parameters:
        -----------
        file_paths : List[str]
            List of relative file paths within NAB data directory
            
        Returns:
        --------
        Tuple[pd.DataFrame, pd.Series]
            Combined multivariate dataframe and binary anomaly labels
        """
        dfs = []
        all_anomaly_windows = []
        
        for i, file_path in enumerate(file_paths):
            print(f"Loading {file_path}...")
            
            # Load data
            df = self.load_single_file(file_path)
            
            # Rename value column to include file identifier
            metric_name = Path(file_path).stem  # filename without extension
            df.rename(columns={'value': metric_name}, inplace=True)
            dfs.append(df)
            
            # Load labels
            anomaly_windows = self.load_labels(file_path)
            all_anomaly_windows.extend(anomaly_windows)
        
        # Combine all dataframes on timestamp index
        combined_df = pd.concat(dfs, axis=1, join='outer')
        
        # Forward fill missing values (common in time-series alignment)
        combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
        
        # Create combined anomaly labels
        labels = self.create_anomaly_labels_column(combined_df, all_anomaly_windows)
        
        print(f"\nMultivariate dataset created:")
        print(f"  Shape: {combined_df.shape}")
        print(f"  Columns: {list(combined_df.columns)}")
        print(f"  Anomalies: {labels.sum()} / {len(labels)} ({100*labels.sum()/len(labels):.2f}%)")
        
        return combined_df, labels
    
    def get_available_files(self, category: str = "realAWSCloudwatch") -> List[str]:
        """
        Get list of available data files in a category.
        
        Parameters:
        -----------
        category : str
            Data category (e.g., 'realAWSCloudwatch', 'artificialNoAnomaly')
            
        Returns:
        --------
        List[str]
            List of available file paths
        """
        category_path = self.data_path / category
        
        if not category_path.exists():
            return []
        
        csv_files = list(category_path.glob("*.csv"))
        return [str(f.relative_to(self.data_path)) for f in csv_files]


def get_default_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load a default multivariate dataset.
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series]
        Multivariate dataframe and anomaly labels
    """
    loader = NABDataLoader()
    
    # Download dataset if needed
    loader.download_nab_dataset()
    
    # Default files: AWS CloudWatch metrics
    default_files = [
        "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
        "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
        "realAWSCloudwatch/ec2_network_in_257a54.csv",
        "realAWSCloudwatch/elb_request_count_8c0756.csv"
    ]
    
    # Check which files are available
    available_files = loader.get_available_files("realAWSCloudwatch")
    
    # Use available files or subset
    files_to_load = []
    for f in default_files:
        if f in available_files:
            files_to_load.append(f)
    
    if not files_to_load:
        # Use first 4 available files
        files_to_load = available_files[:4]
    
    print(f"Loading {len(files_to_load)} AWS CloudWatch metric files...")
    
    return loader.load_multivariate_dataset(files_to_load)


if __name__ == "__main__":
    # Test the data loader
    print("Testing NAB Data Loader...\n")
    
    df, labels = get_default_dataset()
    
    print("\nDataset Summary:")
    print(df.head())
    print(f"\nDate range: {df.index.min()} to {df.index.max()}")
    print(f"Total anomalies: {labels.sum()}")
