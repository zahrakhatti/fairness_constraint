import torch
from config import get_args


# ============================================================================
# Helper Functions
# ============================================================================

def split_by_group_torch(inputs, groups, labels=None):
    """
    Split inputs (and optionally labels) by group membership.

    Returns:
        If labels is None: (inputs_s_0, inputs_s_1)
        If labels provided: (inputs_s_0, inputs_s_1, labels_s_0, labels_s_1)
    """
    mask_s_0 = (groups == 0)
    mask_s_1 = (groups == 1)

    inputs_s_0 = inputs[mask_s_0]
    inputs_s_1 = inputs[mask_s_1]

    if labels is None:
        return inputs_s_0, inputs_s_1

    labels_s_0 = labels[mask_s_0]
    labels_s_1 = labels[mask_s_1]
    return inputs_s_0, inputs_s_1, labels_s_0, labels_s_1


def split_by_group_and_label(inputs, groups, labels, label_value=1):
    """
    Split inputs by both group and label value (e.g., y=1).

    Returns:
        (inputs_y_s_0, inputs_y_s_1)
    """
    mask_y_s_0 = (groups == 0) & (labels == label_value)
    mask_y_s_1 = (groups == 1) & (labels == label_value)

    return inputs[mask_y_s_0], inputs[mask_y_s_1]


def get_alpha_for_dataset(args):
    """Get alpha value based on dataset if scaled is True."""
    if not args.scaled:
        return getattr(args, 'alpha', 1.0)

    alpha_map = {
        'adult': 100,
        'law': 50,
        'acsincome': 50,
        'dutch': 50
    }
    return alpha_map.get(args.dataset, 50)


def compute_predictions_by_group(model, inputs, groups, threshold=0.5):
    """
    Get model predictions for each group separately.

    Returns:
        (y_hat_s_0, y_hat_s_1)
    """
    inputs_s_0, inputs_s_1 = split_by_group_torch(inputs, groups)

    y_hat_s_0 = model(inputs_s_0).view(-1)
    y_hat_s_1 = model(inputs_s_1).view(-1)

    return y_hat_s_0, y_hat_s_1


def compute_sigmoid_probabilities(y_hat, threshold=0.5, alpha=1.0):
    """
    Compute sigmoid-based probabilities from model outputs.

    Args:
        y_hat: Model outputs (logits or probabilities)
        threshold: Decision threshold
        alpha: Scaling factor for sigmoid

    Returns:
        Probability tensor
    """
    t_hat = alpha * (y_hat - threshold)
    return torch.sigmoid(t_hat)


def compute_smooth_probabilities(y_hat, threshold=0.5, eps_smooth=1e-5):
    """
    Compute smoothed probabilities using sqrt-based approximation.

    Args:
        y_hat: Model outputs
        threshold: Decision threshold
        eps_smooth: Smoothing parameter

    Returns:
        Probability tensor
    """
    t_hat = (y_hat - threshold)
    return 0.5 * (t_hat + torch.sqrt(t_hat**2 + eps_smooth))


def compute_complex_smooth_prob(t_hat, eps_smooth):
    """
    Compute complex smoothed probability (used in DI and EI).
    This implements: 1 - (1/2)*(1 - ((1/2)*(t + 0.5 + sqrt((t+0.5)^2 + eps))) + sqrt(...))

    Args:
        t_hat: Transformed logits
        eps_smooth: Smoothing parameter

    Returns:
        Smoothed probability
    """
    inner = 0.5 * (t_hat + 0.5 + torch.sqrt((t_hat + 0.5)**2 + eps_smooth))
    outer = 1 - inner
    result = 1 - 0.5 * (outer + torch.sqrt(outer**2 + eps_smooth))
    return result


# ============================================================================
# Balancing and Sampling
# ============================================================================

def balanced_sampling(dataset_inputs, dataset_labels, dataset_groups, num_samples_per_group):
    """
    Create a balanced sample from the dataset (equal samples from each group).
    Note: This function appears to have a bug in the original - it doesn't actually sample.
    Kept as-is for backward compatibility.
    """
    s_1_indices = (dataset_groups == 1).nonzero(as_tuple=True)[0]
    s_0_indices = (dataset_groups == 0).nonzero(as_tuple=True)[0]

    # Original code has a bug here - it doesn't slice the indices
    s_1_sampled_indices = s_1_indices[len(s_1_indices)]
    s_0_sampled_indices = s_0_indices[len(s_0_indices)]

    s_1_sampled_inputs = dataset_inputs[s_1_sampled_indices]
    s_0_sampled_inputs = dataset_inputs[s_0_sampled_indices]
    s_1_sampled_labels = dataset_labels[s_1_sampled_indices]
    s_0_sampled_labels = dataset_labels[s_0_sampled_indices]
    s_1_sampled_groups = dataset_groups[s_1_sampled_indices]
    s_0_sampled_groups = dataset_groups[s_0_sampled_indices]

    balanced_inputs = torch.cat((s_1_sampled_inputs, s_0_sampled_inputs))
    balanced_labels = torch.cat((s_1_sampled_labels, s_0_sampled_labels))
    balanced_groups = torch.cat((s_1_sampled_groups, s_0_sampled_groups))

    # Shuffle the balanced dataset
    shuffled_indices = torch.randperm(balanced_inputs.size(0))
    balanced_inputs = balanced_inputs[shuffled_indices]
    balanced_labels = balanced_labels[shuffled_indices]
    balanced_groups = balanced_groups[shuffled_indices]

    return balanced_inputs, balanced_labels, balanced_groups


# ============================================================================
# Demographic Parity Constraints
# ============================================================================

def covariance(model, inputs, groups):
    """Compute covariance between group membership and predictions."""
    s_mean = groups.mean()
    y_hat = model(inputs).view(-1)
    c = torch.dot((groups - s_mean), y_hat) / len(groups)
    return c


def probability_dp(model, inputs, groups, eps_smooth_dp):
    """
    Compute demographic parity using smooth approximation.

    Returns:
        Difference between group probabilities
    """
    y_hat_s_0, y_hat_s_1 = compute_predictions_by_group(model, inputs, groups)

    # Compute smoothed probabilities
    ratio_s_0 = torch.mean(compute_smooth_probabilities(y_hat_s_0, eps_smooth=eps_smooth_dp))
    ratio_s_1 = torch.mean(compute_smooth_probabilities(y_hat_s_1, eps_smooth=eps_smooth_dp))

    return ratio_s_1 - ratio_s_0


def probability_sigmoid_dp(model, inputs, groups, eps_smooth_dp):
    """
    Compute demographic parity using sigmoid approximation.

    Returns:
        Difference between group probabilities
    """
    y_hat_s_0, y_hat_s_1 = compute_predictions_by_group(model, inputs, groups)

    # Compute sigmoid probabilities
    ratio_s_0 = torch.mean(torch.sigmoid(y_hat_s_0 - 0.5))
    ratio_s_1 = torch.mean(torch.sigmoid(y_hat_s_1 - 0.5))

    return ratio_s_1 - ratio_s_0


def probability_sigmoid(model, inputs, groups, delta, args):
    """
    Compute disparate impact using sigmoid approximation.

    Returns:
        (r_s_0, r_s_1, c_1, c_2) where c_1 and c_2 are the constraints
    """
    alpha = get_alpha_for_dataset(args)
    y_hat_s_0, y_hat_s_1 = compute_predictions_by_group(model, inputs, groups)

    # Compute sigmoid probabilities with scaling
    r_s_0 = torch.mean(compute_sigmoid_probabilities(y_hat_s_0, alpha=alpha))
    r_s_1 = torch.mean(compute_sigmoid_probabilities(y_hat_s_1, alpha=alpha))

    # Compute constraints: delta * r_s_0 - r_s_1 <= 0 and delta * r_s_1 - r_s_0 <= 0
    c_1 = delta * r_s_0 - r_s_1
    c_2 = delta * r_s_1 - r_s_0

    return r_s_0, r_s_1, c_1, c_2


def probability_DI(model, inputs, groups, eps_smooth, delta, args):
    """
    Compute Disparate Impact using complex smooth approximation.

    Returns:
        (sum_s0, sum_s1, r_s_0, r_s_1, c_1, c_2)
    """
    alpha = get_alpha_for_dataset(args)
    y_hat_s_0, y_hat_s_1 = compute_predictions_by_group(model, inputs, groups)

    # Transform with alpha
    t_hat_s_0 = alpha * (y_hat_s_0 - 0.5)
    t_hat_s_1 = alpha * (y_hat_s_1 - 0.5)

    # Compute complex smooth probabilities
    prob_s_0 = compute_complex_smooth_prob(t_hat_s_0, eps_smooth)
    prob_s_1 = compute_complex_smooth_prob(t_hat_s_1, eps_smooth)

    sum_s0 = torch.sum(prob_s_0)
    sum_s1 = torch.sum(prob_s_1)
    r_s_0 = torch.mean(prob_s_0)
    r_s_1 = torch.mean(prob_s_1)

    # Compute constraints
    c_1 = delta * r_s_0 - r_s_1
    c_2 = delta * r_s_1 - r_s_0

    return sum_s0, sum_s1, r_s_0, r_s_1, c_1, c_2


# ============================================================================
# Equal Opportunity Constraints
# ============================================================================

def covariance_eo(model, inputs, labels, groups):
    """Compute covariance for equal opportunity (conditioned on y=1)."""
    mask_y1 = (labels == 1)
    groups_given_y = groups[mask_y1]
    s_mean_given_y = groups_given_y.mean()

    y_hat = model(inputs).view(-1)
    y_hat_given_y = y_hat[mask_y1]

    c = torch.dot((groups_given_y - s_mean_given_y), y_hat_given_y) / len(groups_given_y)
    return c


def probability_eo(model, inputs, labels, groups, eps_smooth_eo, args):
    """
    Compute equal opportunity using smooth approximation.

    Returns:
        Difference between group probabilities conditioned on y=1
    """
    alpha = get_alpha_for_dataset(args)

    # Split by group and label
    inputs_y1_s_0, inputs_y1_s_1 = split_by_group_and_label(inputs, groups, labels, label_value=1)

    # Get predictions
    y_hat_y1_s_0 = model(inputs_y1_s_0).view(-1)
    y_hat_y1_s_1 = model(inputs_y1_s_1).view(-1)

    # Transform with alpha
    t_y1_s_0 = alpha * (y_hat_y1_s_0 - 0.5)
    t_y1_s_1 = alpha * (y_hat_y1_s_1 - 0.5)

    # Compute smoothed probabilities
    prob_pos_y1_s_0 = torch.mean(compute_smooth_probabilities(t_y1_s_0, eps_smooth=eps_smooth_eo))
    prob_pos_y1_s_1 = torch.mean(compute_smooth_probabilities(t_y1_s_1, eps_smooth=eps_smooth_eo))

    return prob_pos_y1_s_1 - prob_pos_y1_s_0


def probability_EI(model, inputs, labels, groups, eps_smooth, delta, args):
    """
    Compute Equal Impact using complex smooth approximation.

    Returns:
        (r_y1_s_0, r_y1_s_1, c_1, c_2)
    """
    alpha = get_alpha_for_dataset(args)

    # Split by group and label
    inputs_y1_s_0, inputs_y1_s_1 = split_by_group_and_label(inputs, groups, labels, label_value=1)

    # Get predictions
    y_hat_y1_s_0 = model(inputs_y1_s_0).view(-1)
    y_hat_y1_s_1 = model(inputs_y1_s_1).view(-1)

    # Transform with alpha
    t_y1_s_0 = alpha * (y_hat_y1_s_0 - 0.5)
    t_y1_s_1 = alpha * (y_hat_y1_s_1 - 0.5)

    # Compute complex smooth probabilities
    r_y1_s_0 = torch.mean(compute_complex_smooth_prob(t_y1_s_0, eps_smooth))
    r_y1_s_1 = torch.mean(compute_complex_smooth_prob(t_y1_s_1, eps_smooth))

    # Compute constraints
    c_1 = delta * r_y1_s_0 - r_y1_s_1
    c_2 = delta * r_y1_s_1 - r_y1_s_0

    return r_y1_s_0, r_y1_s_1, c_1, c_2
