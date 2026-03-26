import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================================
# Helper Functions - DRY Principle
# ============================================================================

def split_by_group(pred_arr, label_arr, group_arr):
    """
    Split arrays by group membership (0 and 1).

    Returns:
        tuple: (s_0_preds, s_0_labels, s_1_preds, s_1_labels)
    """
    s_0_mask = (group_arr == 0)
    s_1_mask = (group_arr == 1)

    s_0_preds = pred_arr[s_0_mask]
    s_0_labels = label_arr[s_0_mask] if label_arr is not None else None
    s_1_preds = pred_arr[s_1_mask]
    s_1_labels = label_arr[s_1_mask] if label_arr is not None else None

    return s_0_preds, s_0_labels, s_1_preds, s_1_labels


def positive_prediction_rates(pred_arr, group_arr):
    """
    Calculate positive prediction rates for both groups.

    Returns:
        tuple: (positive_prob_s_0, positive_prob_s_1)
    """
    s_0_preds, _, s_1_preds, _ = split_by_group(pred_arr, None, group_arr)

    positive_prob_s_0 = np.mean(s_0_preds == 1)
    positive_prob_s_1 = np.mean(s_1_preds == 1)

    return positive_prob_s_0, positive_prob_s_1


def true_positive_rates(label_arr, pred_arr, group_arr):
    """
    Calculate true positive rates (TPR) for both groups.

    Returns:
        tuple: (tpr_s_0, tpr_s_1)
    """
    s_0_preds, s_0_labels, s_1_preds, s_1_labels = split_by_group(pred_arr, label_arr, group_arr)

    # Compute confusion matrices
    s_0_conf = confusion_matrix(s_0_labels, s_0_preds)
    s_1_conf = confusion_matrix(s_1_labels, s_1_preds)

    # Extract TPR (TP / (TP + FN))
    _, _, fn_0, tp_0 = s_0_conf.ravel()
    _, _, fn_1, tp_1 = s_1_conf.ravel()

    tpr_s_0 = tp_0 / (tp_0 + fn_0)
    tpr_s_1 = tp_1 / (tp_1 + fn_1)

    return tpr_s_0, tpr_s_1


# ============================================================================
# Fairness Metrics
# ============================================================================

def positive_prediction_ratio(pred_arr, group_arr):
    """Calculate positive prediction rates for both groups."""
    return positive_prediction_rates(pred_arr, group_arr)


def acc_diff_binary(label_arr, pred_arr, group_arr):
    """Calculate accuracy difference between groups."""
    s_0_preds, s_0_labels, s_1_preds, s_1_labels = split_by_group(pred_arr, label_arr, group_arr)

    acc_s_0 = accuracy_score(s_0_labels, s_0_preds)
    acc_s_1 = accuracy_score(s_1_labels, s_1_preds)
    overall_acc = accuracy_score(label_arr, pred_arr)

    return overall_acc, acc_s_1, acc_s_0


def Covariance_ind_binary(pred_arr, group_arr):
    """Calculate covariance between predictions and group membership."""
    s_mean = np.mean(group_arr)
    cov_dp = abs(np.dot((group_arr - s_mean), pred_arr) / len(group_arr))
    return cov_dp


def demographic_parity_binary(pred_arr, group_arr):
    """Calculate demographic parity (difference in positive prediction rates)."""
    positive_prob_s_0, positive_prob_s_1 = positive_prediction_rates(pred_arr, group_arr)
    dp = abs(positive_prob_s_1 - positive_prob_s_0)
    return dp


def Covariance_eo_binary(label_arr, pred_arr, group_arr):
    """Calculate covariance for equal opportunity (conditioned on y=1)."""
    mask_y1 = (label_arr == 1)
    groups_given_y = group_arr[mask_y1]
    y_hat_given_y = pred_arr[mask_y1]

    s_mean_given_y = np.mean(groups_given_y)
    cov_eo = abs(np.dot((groups_given_y - s_mean_given_y), y_hat_given_y) / len(groups_given_y))
    return cov_eo


def equal_opportunity_binary(label_arr, pred_arr, group_arr):
    """Calculate equal opportunity (difference in true positive rates)."""
    tpr_s_0, tpr_s_1 = true_positive_rates(label_arr, pred_arr, group_arr)
    eq_opp = abs(tpr_s_0 - tpr_s_1)
    return eq_opp


def disparate_impact_binary(pred_arr, group_arr, delta):
    """Calculate disparate impact constraints."""
    positive_prob_s_0, positive_prob_s_1 = positive_prediction_rates(pred_arr, group_arr)
    c_bar_1 = delta * positive_prob_s_0 - positive_prob_s_1
    c_bar_2 = delta * positive_prob_s_1 - positive_prob_s_0
    return c_bar_1, c_bar_2, max(c_bar_1, c_bar_2)


def equal_impact_binary(label_arr, pred_arr, group_arr, delta):
    """Calculate equal impact constraints (disparate impact on TPR)."""
    tpr_s_0, tpr_s_1 = true_positive_rates(label_arr, pred_arr, group_arr)
    c_bar_1 = delta * tpr_s_0 - tpr_s_1
    c_bar_2 = delta * tpr_s_1 - tpr_s_0
    return c_bar_1, c_bar_2, max(c_bar_1, c_bar_2)
