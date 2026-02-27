"""
Download Loan Default datasets from Kaggle.
Requires: kaggle API installed and configured

Setup:
1. pip install kaggle
2. Place kaggle.json in ~/.kaggle/kaggle.json
   Download from: https://www.kaggle.com/settings/account
3. Run: python src/datasets/fetch_kaggle_loan.py
"""

import os
import pandas as pd
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def download_lending_club():
    """Download Lending Club data from Kaggle (alternative: direct download)"""
    print("Downloading Lending Club dataset...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('ethon0426/lending-club-loan-defaulters', path=DATA_RAW, unzip=True)
        print("✓ Downloaded to data/raw/")
        return True
    except Exception as e:
        print(f"✗ Kaggle API error: {e}")
        print("\nAlternative: Download manually from:")
        print("https://www.kaggle.com/datasets/ethon0426/lending-club-loan-defaulters")
        return False


def load_and_balance_kaggle_loan(csv_path, target_col='loan_status', test_size=0.2):
    """Load and preprocess Kaggle loan data"""
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Target column: {target_col}")
    print(f"Target distribution:\n{df[target_col].value_counts()}\n")
    
    # Drop rows with missing target
    df = df[df[target_col].notna()]
    
    # Encode target if needed
    if df[target_col].dtype == 'object':
        df[target_col] = (df[target_col].str.lower().str.contains('default|1|yes')).astype(int)
    
    # Drop mostly empty columns
    df = df.loc[:, (df.isnull().sum() / len(df)) < 0.5]
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'missing')
    
    # Sample if dataset is too large (for faster iteration)
    if len(df) > 100000:
        print(f"Sampling 100k rows from {len(df)} for training speed...")
        df = df.sample(n=100000, random_state=42)
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42, 
        stratify=df[target_col] if len(df[target_col].unique()) == 2 else None
    )
    
    print(f"Train set: {len(train_df)} rows")
    print(f"Test set: {len(test_df)} rows")
    
    return train_df, test_df


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("KAGGLE LOAN DATASET DOWNLOADER")
    print("=" * 60)
    
    # Try to download
    success = download_lending_club()
    
    if not success:
        print("\nManual Download Instructions:")
        print("1. Go to: https://www.kaggle.com/datasets/ethon0426/lending-club-loan-defaulters")
        print("2. Download the dataset")
        print("3. Extract CSV to data/raw/")
        print("4. Re-run this script")
        sys.exit(1)
    
    # Find CSV file
    csv_files = list(DATA_RAW.glob("*.csv"))
    if not csv_files:
        print("✗ No CSV files found in data/raw/")
        sys.exit(1)
    
    csv_path = csv_files[0]
    print(f"\nProcessing: {csv_path.name}")
    
    # Load and process
    train_df, test_df = load_and_balance_kaggle_loan(str(csv_path))
    
    # Save as parquet
    train_parquet = DATA_PROCESSED / "train.parquet"
    test_parquet = DATA_PROCESSED / "test.parquet"
    
    train_df.to_parquet(train_parquet, index=False)
    test_df.to_parquet(test_parquet, index=False)
    
    print(f"\n✓ Saved train data to: {train_parquet}")
    print(f"✓ Saved test data to: {test_parquet}")
    print("\nReady to train! Run:")
    print("python src/model_training.py --train data/processed/train.parquet --test data/processed/test.parquet --target loan_status --model artifacts/model_xgb --metrics reports/model_metrics.json")


if __name__ == "__main__":
    main()
