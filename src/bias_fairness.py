import pandas as pd
from sklearn.metrics import confusion_matrix

def evaluate_fairness(y_true, y_pred, sensitive):
    """
    Evaluate fairness metrics: demographic parity and equalized odds.
    """
    df = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'sensitive': sensitive})
    
    # Demographic Parity: Difference in positive prediction rates
    grp_pos_rate = df.groupby('sensitive')['y_pred'].mean()
    dp_diff = grp_pos_rate.max() - grp_pos_rate.min()
    
    # Equalized Odds: Difference in TPR and FPR
    groups = df['sensitive'].unique()
    tprs = []
    fprs = []
    for g in groups:
        sub = df[df['sensitive'] == g]
        tn, fp, fn, tp = confusion_matrix(sub['y_true'], sub['y_pred']).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)
    eo_diff = max(max(tprs) - min(tprs), max(fprs) - min(fprs))
    
    return {
        'demographic_parity_difference': float(dp_diff),
        'equalized_odds_difference': float(eo_diff)
    }
