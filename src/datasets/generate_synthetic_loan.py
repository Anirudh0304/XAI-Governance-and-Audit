"""
Generate a larger synthetic loan dataset for testing.
This creates a realistic 50k-row loan dataset to improve model accuracy.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"


def generate_synthetic_loan_data(n_samples=50000, random_state=42):
    """Generate realistic synthetic loan dataset"""
    np.random.seed(random_state)
    
    data = {
        'loan_id': np.arange(n_samples),
        'age': np.random.normal(45, 12, n_samples).astype(int).clip(18, 80),
        'annual_income': np.random.exponential(70000, n_samples).astype(int).clip(15000, 500000),
        'loan_amount': np.random.normal(20000, 15000, n_samples).astype(int).clip(1000, 100000),
        'loan_term_months': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'employment_years': np.random.exponential(8, n_samples).astype(int).clip(0, 50),
        'loan_purpose': np.random.choice(['auto', 'home', 'debt_consolidation', 'personal', 'business'], n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int).clip(300, 850),
        'debt_to_income': np.random.beta(2, 5, n_samples) * 0.6,  # 0-60% ratio
        'num_open_accounts': np.random.poisson(5, n_samples),
        'num_delinquencies': np.random.poisson(0.5, n_samples),
        'housing_status': np.random.choice(['own', 'mortgage', 'rent'], n_samples),
        'marital_status': np.random.choice(['single', 'married', 'divorced'], n_samples),
        'num_dependents': np.random.choice([0, 1, 2, 3, 4], n_samples, p=[0.3, 0.3, 0.25, 0.1, 0.05]),
    }
    
    df = pd.DataFrame(data)
    
    # Generate target with realistic correlation to features
    default_prob = (
        0.05 +  # baseline
        (df['credit_score'] < 600) * 0.15 +
        (df['debt_to_income'] > 0.4) * 0.10 +
        (df['num_delinquencies'] > 0) * 0.20 +
        (df['num_open_accounts'] > 10) * 0.08 +
        (df['employment_years'] < 2) * 0.10 +
        (df['age'] < 25) * 0.05 +
        np.random.normal(0, 0.02, n_samples)  # noise
    ).clip(0, 1)
    
    df['loan_default'] = (np.random.random(n_samples) < default_prob).astype(int)
    
    # Remove ID column (will be dropped during preprocessing anyway)
    df = df.drop(columns=['loan_id'])
    
    print(f"Generated synthetic dataset: {df.shape}")
    print(f"Target distribution:")
    print(df['loan_default'].value_counts())
    print(f"\nDefault rate: {df['loan_default'].mean():.1%}")
    
    return df


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING SYNTHETIC LOAN DATASET")
    print("=" * 70)
    
    # Generate data
    df = generate_synthetic_loan_data(n_samples=50000)
    
    # Split
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['loan_default']
    )
    
    # Save
    train_parquet = DATA_PROCESSED / "train.parquet"
    test_parquet = DATA_PROCESSED / "test.parquet"
    
    train_df.to_parquet(train_parquet, index=False)
    test_df.to_parquet(test_parquet, index=False)
    
    print(f"\n✓ Training set: {len(train_df)} rows → {train_parquet}")
    print(f"✓ Test set: {len(test_df)} rows → {test_parquet}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nNext: Run training with target='loan_default':")
    print(f"python src/model_training.py --train data/processed/train.parquet --test data/processed/test.parquet --target loan_default --model artifacts/model_xgb --metrics reports/model_metrics.json")


if __name__ == "__main__":
    main()
