# Optional Advanced Improvements (Use Case by Case)

These are additional optimization techniques you can apply if current results don't reach 85%. Implement these incrementally and test their impact.

---

## 1. Feature Selection (Removes Low-Importance Features)

Add this function to `model_training.py` to reduce feature space and remove noise:

```python
def select_important_features(X_train, y_train, X_test, threshold=0.01):
    """
    Select features based on importance scores to reduce dimensionality and noise.

    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        threshold: Importance threshold (features below this are removed)
    """
    # Use mutual information for feature selection
    selector = SelectFromModel(
        xgb.XGBClassifier(random_state=42, n_jobs=-1),
        prefit=False,
        threshold='median'  # Keep only features above median importance
    )

    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    selected_features = X_train.columns[selector.get_support()]
    print(f"Selected {len(selected_features)} out of {X_train.shape[1]} features")

    return X_train_selected, X_test_selected, selected_features
```

**Integration**: Apply this in `train_and_evaluate()` after preprocessing:
```python
X_train_selected, X_test_selected, selected_features = select_important_features(X_train, y_train, X_test)
# Then use X_train_selected and X_test_selected for training
```

**Expected Gain**: +1-2% accuracy by removing noisy features

---

## 2. Advanced Ensemble - Stacking

Use predictions from base models as features for a meta-learner:

```python
def create_stacking_ensemble(X_train, y_train, X_test, random_state=42):
    """Create a stacking ensemble using RF and XGBoost as base learners."""
    from sklearn.linear_model import LogisticRegression

    # Balance training data with SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Base learners
    rf = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    xgb_clf = xgb.XGBClassifier(n_estimators=300, random_state=random_state, max_depth=6)

    # Meta-learner
    meta_clf = LogisticRegression(random_state=random_state, max_iter=1000)

    from sklearn.ensemble import StackingClassifier
    stacker = StackingClassifier(
        estimators=[('rf', rf), ('xgb', xgb_clf)],
        final_estimator=meta_clf,
        cv=5
    )

    stacker.fit(X_train_bal, y_train_bal)
    return stacker
```

**Expected Gain**: +2-4% accuracy through better meta-learning

---

## 3. LightGBM Integration (Faster, Often Better Than XGBoost)

Add LightGBM as an alternative to XGBoost:

```python
import lightgbm as lgb

def train_lightgbm(X_train, y_train, X_test, y_test, random_state=42):
    """Train LightGBM model with optimized parameters."""

    # Balance data
    smote = SMOTE(random_state=random_state)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    lgb_model = lgb.LGBMClassifier(
        n_estimators=500,
        max_depth=7,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train_bal[y_train_bal==0])/len(y_train_bal[y_train_bal==1]),
        random_state=random_state,
        n_jobs=-1,
        verbose=-1
    )

    lgb_model.fit(X_train_bal, y_train_bal, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(20)])
    return lgb_model
```

**Expected Gain**: +1-3% accuracy, 2-3x faster training

---

## 4. Calibrated Classifiers (Better Probability Estimates)

Calibrate model probabilities for better threshold tuning:

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_model(model, X_calib, y_calib):
    """Calibrate model probabilities for better decision boundaries."""
    calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=5)
    calibrated.fit(X_calib, y_calib)
    return calibrated
```

**Usage**:
```python
# After train_test_split, reserve part of training data for calibration
X_train, X_calib, y_train, y_calib = train_test_split(X_train, y_train, test_size=0.2, random_state=seed)
best_model = calibrate_model(best_model, X_calib, y_calib)
```

**Expected Gain**: +0.5-1% accuracy with better probability estimates

---

## 5. Focused Data Preprocessing (Domain-Specific)

For German Credit dataset specifically, add:

```python
def add_credit_specific_features(X, y=None):
    """Add domain-specific features for credit datasets."""
    X = X.copy()

    # Risk score features (if credit_amount, duration available)
    if 'credit_amount' in X.columns and 'duration' in X.columns:
        X['monthly_payment'] = X['credit_amount'] / X['duration']
        X['payment_to_monthly_income'] = X['monthly_payment'] / X.get('monthly_income', 1)

    # Credit history indicators (if available)
    if 'credit_history' in X.columns:
        X['has_good_credit_history'] = (X['credit_history'] < 3).astype(int)

    # Employment stability (if available)
    if 'employment_duration' in X.columns:
        X['is_long_term_employed'] = (X['employment_duration'] >= 4).astype(int)

    return X
```

**Expected Gain**: +1-2% accuracy with domain knowledge integration

---

## 6. Threshold Optimization Beyond F1

Instead of F1-score, optimize for business metric:

```python
def find_optimal_threshold_by_metric(y_true, y_prob, metric='f1'):
    """Find optimal threshold based on different metrics."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    if metric == 'f1':
        scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    elif metric == 'specificity':  # Minimize false positives
        scores = (1 - precision) / (2 - precision - recall)
    elif metric == 'sensitivity':  # Minimize false negatives
        scores = recall / (1 - recall + 1e-10)
    else:
        scores = (precision + recall) / 2  # Balance

    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx], scores[optimal_idx]
```

**Expected Gain**: +0-3% depending on business requirements

---

## 7. Cross-Validation Ensemble

Combine models trained on different folds:

```python
def train_cv_ensemble(X_train, y_train, X_test, n_splits=5, random_state=42):
    """Train ensemble of models on different CV folds."""
    from sklearn.model_selection import KFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    models = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = train_models(X_tr, y_tr, random_state)[1]  # Use XGBoost
        models.append(model)

    # Average predictions across models
    predictions = np.mean([m.predict_proba(X_test)[:, 1] for m in models], axis=0)
    return predictions
```

**Expected Gain**: +1-2% accuracy with more robust predictions

---

## Implementation Priority

Based on effort vs. expected gain:

1. **Try First** (Quick wins):
   - Feature Selection
   - Threshold optimization by metric
   - Advanced preprocessing

2. **Try Next** (If still below 85%):
   - LightGBM integration
   - Calibrated classifiers
   - Stacking ensemble

3. **Try Last** (Diminishing returns):
   - CV ensemble
   - Custom feature engineering
   - Hyperparameter micro-tuning

---

## Testing Strategy

```python
import time

def test_improvementComparison(model_old, model_new, X_test, y_test):
    """Compare models systematically."""

    for name, model in [('Old', model_old), ('New', model_new)]:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        print(f"\n{name} Model:")
        print(f"  Accuracy: {accuracy_score(y_test, y_pred):.4f}")
        print(f"  ROC-AUC:  {roc_auc_score(y_test, y_prob):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred):.4f}")
```

Track improvements step-by-step and stop when you reach 85% or see diminishing returns.
