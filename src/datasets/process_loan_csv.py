"""
Alternative: Manual download helper for Kaggle datasets.
If Kaggle API doesn't work, download manually and this script will process it.
"""

import os
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def process_loan_csv(csv_path, target_col='Loan_Status', test_size=0.2):
    """Process any loan CSV dataset"""
    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}\n")
    
    # Auto-detect target column
    possible_targets = ['loan_status', 'Loan_Status', 'default', 'Default', 'target', 'Target', 
                       'y', 'Y', 'label', 'Label', 'status', 'Status', 'is_default', 'bad']
    
    target_col_found = None
    for col in possible_targets:
        if col in df.columns:
            target_col_found = col
            break
    
    if not target_col_found:
        print("Could not find target column. Available columns:")
        print(df.columns.tolist())
        print("\nPlease specify target column name:")
        target_col_found = input("> ").strip()
    
    print(f"Using target: {target_col_found}")
    print(f"Target values:\n{df[target_col_found].value_counts()}\n")
    
    # Drop rows with missing target
    df = df[df[target_col_found].notna()].copy()
    
    # Encode target if text
    if df[target_col_found].dtype == 'object':
        df[target_col_found] = df[target_col_found].str.lower()
        df[target_col_found] = df[target_col_found].isin(['default', 'yes', '1', 'defaulted', 'bad', 'charged off']).astype(int)
    else:
        df[target_col_found] = (df[target_col_found] != 0).astype(int)
    
    print(f"Encoded target:\n{df[target_col_found].value_counts()}\n")
    
    # Remove ID columns
    for id_col in ['id', 'ID', 'Id', 'loan_id', 'Loan_ID', 'account_number']:
        if id_col in df.columns:
            df = df.drop(columns=[id_col])
    
    # Handle missing values
    df = df.loc[:, (df.isnull().sum() / len(df)) < 0.7]  # Drop mostly null columns
    
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        mode_val = df[col].mode()
        fill_val = mode_val[0] if len(mode_val) > 0 else 'missing'
        df[col] = df[col].fillna(fill_val)
    
    # Sample if too large
    if len(df) > 100000:
        print(f"Sampling 100k rows from {len(df)}...")
        df = df.sample(n=100000, random_state=42)
    
    # Split
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=42,
        stratify=df[target_col_found]
    )
    
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    
    return train_df, test_df, target_col_found


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("MANUAL KAGGLE DATASET PROCESSOR")
    print("=" * 70)
    print("\nSteps:")
    print("1. Go to https://www.kaggle.com/search?q=loan+default")
    print("2. Choose a dataset (e.g., 'Lending Club Loan Defaulters')")
    print("3. Download the CSV file")
    print("4. Save it to: C:\\Users\\aniru\\Downloads\\explainable_ai\\data\\raw\\")
    print("5. This script will auto-detect and process it\n")
    
    # Find CSV
    csv_files = list(DATA_RAW.glob("*.csv"))
    
    if not csv_files:
        print("✗ No CSV found in data/raw/")
        print(f"Please download a loan dataset to: {DATA_RAW}")
        sys.exit(1)
    
    csv_path = csv_files[0]
    if len(csv_files) > 1:
        print("Multiple CSVs found:")
        for i, fp in enumerate(csv_files, 1):
            print(f"  {i}. {fp.name}")
        idx = int(input("Choose file (number): ")) - 1
        csv_path = csv_files[idx]
    
    # Process
    train_df, test_df, target_col = process_loan_csv(str(csv_path))
    
    # Save
    train_parquet = DATA_PROCESSED / "train.parquet"
    test_parquet = DATA_PROCESSED / "test.parquet"
    
    train_df.to_parquet(train_parquet, index=False)
    test_df.to_parquet(test_parquet, index=False)
    
    print(f"\n✓ Train: {train_parquet}")
    print(f"✓ Test: {test_parquet}")
    print(f"\nNext command:")
    print(f"python src/model_training.py --train data/processed/train.parquet --test data/processed/test.parquet --target {target_col} --model artifacts/model_xgb --metrics reports/model_metrics.json")


if __name__ == "__main__":
    main()
