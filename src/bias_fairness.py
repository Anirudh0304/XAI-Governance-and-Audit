import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def _bin_sensitive(sensitive: pd.Series) -> pd.Series:
    """Auto-bin continuous numeric sensitive attributes."""
    if pd.api.types.is_numeric_dtype(sensitive) and sensitive.nunique() > 10:
        return pd.cut(
            sensitive,
            bins=[0, 25, 35, 50, 100],
            labels=['young', 'young_adult', 'middle_aged', 'senior']
        ).astype(str)
    return sensitive


def _compute_dp(y_pred: np.ndarray, sensitive: np.ndarray) -> float:
    """Demographic Parity Difference."""
    df = pd.DataFrame({'y_pred': y_pred, 'sensitive': sensitive})
    rates = df.groupby('sensitive')['y_pred'].mean()
    return float(rates.max() - rates.min()) if len(rates) > 1 else 0.0


def _compute_eo(y_true: np.ndarray, y_pred: np.ndarray,
                sensitive: np.ndarray) -> float:
    """Equalized Odds Difference."""
    groups = np.unique(sensitive)
    tprs, fprs = [], []

    for g in groups:
        mask     = sensitive == g
        yt, yp   = y_true[mask], y_pred[mask]

        if len(np.unique(yt)) < 2 or len(yt) < 2:
            tprs.append(0.0)
            fprs.append(0.0)
            continue

        try:
            cm = confusion_matrix(yt, yp, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            tprs.append(float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0)
            fprs.append(float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0)
        except ValueError:
            tprs.append(0.0)
            fprs.append(0.0)

    if len(tprs) > 1:
        return float(max(max(tprs) - min(tprs), max(fprs) - min(fprs)))
    return 0.0


def bootstrap_fairness_ci(y_true, y_pred, sensitive,
                           n_bootstrap: int = 1000,
                           ci: int = 95,
                           random_state: int = 42) -> dict:
    """
    Compute bootstrap confidence intervals for Demographic Parity
    and Equalized Odds differences.

    Args:
        y_true:       True labels
        y_pred:       Predicted labels
        sensitive:    Sensitive attribute values (already binned)
        n_bootstrap:  Number of bootstrap resamples (default 1000)
        ci:           Confidence level in % (default 95)
        random_state: Random seed for reproducibility

    Returns:
        Dict with mean, lower, upper bounds for both DP and EO
    """
    rng       = np.random.RandomState(random_state)
    y_true    = np.array(y_true)
    y_pred    = np.array(y_pred)
    sensitive = np.array(sensitive)
    n         = len(y_true)

    dp_scores = []
    eo_scores = []

    for _ in range(n_bootstrap):
        idx  = rng.choice(n, size=n, replace=True)
        yt   = y_true[idx]
        yp   = y_pred[idx]
        s    = sensitive[idx]

        dp_scores.append(_compute_dp(yp, s))
        eo_scores.append(_compute_eo(yt, yp, s))

    alpha = (100 - ci) / 2
    return {
        'dp_mean':  float(np.mean(dp_scores)),
        'dp_lower': float(np.percentile(dp_scores, alpha)),
        'dp_upper': float(np.percentile(dp_scores, 100 - alpha)),
        'eo_mean':  float(np.mean(eo_scores)),
        'eo_lower': float(np.percentile(eo_scores, alpha)),
        'eo_upper': float(np.percentile(eo_scores, 100 - alpha)),
        'ci':       ci,
        'n_bootstrap': n_bootstrap,
    }


def evaluate_fairness(y_true, y_pred, sensitive,
                      compute_ci: bool = True,
                      n_bootstrap: int = 1000) -> dict:
    """
    Evaluate fairness metrics with optional bootstrap confidence intervals.

    Args:
        y_true:       True labels
        y_pred:       Predicted labels
        sensitive:    Sensitive attribute values
        compute_ci:   Whether to compute bootstrap CIs (default True)
        n_bootstrap:  Number of bootstrap resamples

    Returns:
        Dict with DP, EO point estimates and optionally CIs
    """
    # Reset indexes to avoid alignment issues
    y_true    = pd.Series(y_true).reset_index(drop=True)
    y_pred    = pd.Series(y_pred).reset_index(drop=True)
    sensitive = pd.Series(sensitive).reset_index(drop=True)

    # Auto-bin continuous numeric sensitive attributes
    sensitive = _bin_sensitive(sensitive)

    # Point estimates
    yt = y_true.values
    yp = y_pred.values
    s  = sensitive.values

    dp_diff = _compute_dp(yp, s)
    eo_diff = _compute_eo(yt, yp, s)

    result = {
        'demographic_parity_difference': dp_diff,
        'equalized_odds_difference':     eo_diff,
    }

    # Bootstrap confidence intervals
    if compute_ci:
        ci_results = bootstrap_fairness_ci(yt, yp, s, n_bootstrap=n_bootstrap)
        result['dp_ci_lower']    = ci_results['dp_lower']
        result['dp_ci_upper']    = ci_results['dp_upper']
        result['eo_ci_lower']    = ci_results['eo_lower']
        result['eo_ci_upper']    = ci_results['eo_upper']
        result['ci_level']       = ci_results['ci']
        result['n_bootstrap']    = ci_results['n_bootstrap']

    return result