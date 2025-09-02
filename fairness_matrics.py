import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score


def positive_prediction_ratio(pred_arr, group_arr):
    # Calculate the proportion of positive predictions for the s_0 group (group 0)
    s_0_indices = np.argwhere(group_arr == 0).flatten()
    s_0_preds = pred_arr[s_0_indices]
    positive_prob_s_0 = np.sum(s_0_preds == 1) / (len(s_0_preds))

    # Calculate the proportion of positive predictions for the s_1 group (group 1)
    s_1_indices = np.argwhere(group_arr == 1).flatten()
    s_1_preds = pred_arr[s_1_indices]
    positive_prob_s_1 = np.sum(s_1_preds == 1) / (len(s_1_preds))

    return positive_prob_s_0, positive_prob_s_1


def acc_diff_binary(label_arr, pred_arr, group_arr):
    # Calculate metrics for the s_0 group (group 0)
    s_0_indices = np.argwhere(group_arr == 0).flatten()
    s_0_labels = label_arr[s_0_indices]
    s_0_preds = pred_arr[s_0_indices]

    acc_s_0 = accuracy_score(s_0_labels, s_0_preds)
    s_0_conf = confusion_matrix(s_0_labels, s_0_preds)

    # Calculate metrics for the s_1 group (group 1)
    s_1_indices = np.argwhere(group_arr == 1).flatten()
    s_1_labels = label_arr[s_1_indices]
    s_1_preds = pred_arr[s_1_indices]

    acc_s_1 = accuracy_score(s_1_labels, s_1_preds)
    
    # Calculate overall accuracy
    overall_acc = accuracy_score(label_arr, pred_arr)
    
    return overall_acc, acc_s_1, acc_s_0 


def Covariance_ind_binary(pred_arr, group_arr):
    s_mean = np.mean(group_arr)
    cov_dp = abs(np.dot((group_arr - s_mean), pred_arr) / (len(group_arr)))
    return cov_dp


def demographic_parity_binary(pred_arr, group_arr):
    # Calculate the proportion of positive predictions for the s_0 group (group 0)
    s_0_indices = np.argwhere(group_arr == 0).flatten()
    s_0_preds = pred_arr[s_0_indices]
    positive_prob_s_0 = np.sum(s_0_preds == 1) / (len(s_0_preds))

    # Calculate the proportion of positive predictions for the s_1 group (group 1)
    s_1_indices = np.argwhere(group_arr == 1).flatten()
    s_1_preds = pred_arr[s_1_indices]
    positive_prob_s_1 = np.sum(s_1_preds == 1) / (len(s_1_preds))

    # Calculate Demographic Parity difference
    dp = abs(positive_prob_s_1 - positive_prob_s_0)

    return dp


def Covariance_eo_binary(label_arr, pred_arr, group_arr):
    groups_given_y = group_arr[label_arr == 1]
    s_mean_given_y = np.mean(group_arr[label_arr == 1])
    y_hat_given_y = pred_arr[label_arr == 1]
    cov_eo = abs(np.dot((groups_given_y - s_mean_given_y), y_hat_given_y) / (len(groups_given_y)))
    return cov_eo


def equal_opportunity_binary(label_arr, pred_arr, group_arr):
    # Calculate metrics for the s_0 group (group 0)
    s_0_indices = np.argwhere(group_arr == 0).flatten()
    s_0_labels = label_arr[s_0_indices]
    s_0_preds = pred_arr[s_0_indices]

    s_0_conf = confusion_matrix(s_0_labels, s_0_preds)
    tn_0, fp_0, fn_0, tp_0 = s_0_conf.ravel()
    tpr_s_0 = tp_0 / (tp_0 + fn_0)

    # Calculate metrics for the s_1 group (group 1)
    s_1_indices = np.argwhere(group_arr == 1).flatten()
    s_1_labels = label_arr[s_1_indices]
    s_1_preds = pred_arr[s_1_indices]

    s_1_conf = confusion_matrix(s_1_labels, s_1_preds)
    tn_1, fp_1, fn_1, tp_1 = s_1_conf.ravel()
    tpr_s_1 = tp_1 / (tp_1 + fn_1)

    eq_opp = abs(tpr_s_0 - tpr_s_1)
    return eq_opp


def disparate_impact_binary(pred_arr, group_arr, delta):
    positive_prob_s_0, positive_prob_s_1 = positive_prediction_ratio(pred_arr, group_arr)
    c_bar_1 = delta * positive_prob_s_0 - positive_prob_s_1
    c_bar_2 = delta * positive_prob_s_1 - positive_prob_s_0
    return c_bar_1, c_bar_2, max(c_bar_1, c_bar_2)


def equal_impact_binary(label_arr, pred_arr, group_arr, delta):
    # Calculate metrics for the s_0 group (group 0)
    s_0_indices = np.argwhere(group_arr == 0).flatten()
    s_0_labels = label_arr[s_0_indices]
    s_0_preds = pred_arr[s_0_indices]

    s_0_conf = confusion_matrix(s_0_labels, s_0_preds)
    tn_0, fp_0, fn_0, tp_0 = s_0_conf.ravel()
    tpr_s_0 = tp_0 / (tp_0 + fn_0)

    # Calculate metrics for the s_1 group (group 1)
    s_1_indices = np.argwhere(group_arr == 1).flatten()
    s_1_labels = label_arr[s_1_indices]
    s_1_preds = pred_arr[s_1_indices]

    s_1_conf = confusion_matrix(s_1_labels, s_1_preds)
    tn_1, fp_1, fn_1, tp_1 = s_1_conf.ravel()
    tpr_s_1 = tp_1 / (tp_1 + fn_1)

    c_bar_1 = delta * tpr_s_0 - tpr_s_1
    c_bar_2 = delta * tpr_s_1 - tpr_s_0

    return c_bar_1, c_bar_2, max(c_bar_1, c_bar_2)