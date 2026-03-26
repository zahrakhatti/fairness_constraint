import torch
from .config import get_args


def balanced_sampling(dataset_inputs, dataset_labels, dataset_groups, num_samples_per_group):

    s_1_indices = (dataset_groups == 1).nonzero(as_tuple=True)[0]
    s_0_indices = (dataset_groups == 0).nonzero(as_tuple=True)[0]

    # Ensure there are enough samples in each group
    s_1_sampled_indices = s_1_indices[len(s_1_indices)]
    s_0_sampled_indices = s_0_indices[(len(s_0_indices))]
    # [:num_samples_per_group]
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


# Demographic Parity --------------------------------------------------------
# ---------------------------------------------------------------------------
def covariance(model, inputs, groups):
    s_mean = groups.mean()
    y_hat = model(inputs).view(-1)
    c = torch.dot((groups - s_mean), y_hat) / len(groups)
    return c


def probability_dp(model, inputs, groups, eps_smooth_dp):
    threshold = 0.5

    inputs_s_1 = inputs[groups == 1]
    inputs_s_0 = inputs[groups == 0]

    y_hat_s_1 = model(inputs_s_1).view(-1)
    y_hat_s_0 = model(inputs_s_0).view(-1)

    t_hat_s_1 = (y_hat_s_1 - threshold)
    t_hat_s_0 = (y_hat_s_0 - threshold)

    ratio_given_s_1 = 0.5 * torch.mean(t_hat_s_1 + torch.sqrt(t_hat_s_1**2 + eps_smooth_dp))
    ratio_given_s_0 = 0.5 * torch.mean(t_hat_s_0 + torch.sqrt(t_hat_s_0**2 + eps_smooth_dp))

    c = ratio_given_s_1 - ratio_given_s_0
    return c


# Sigmoid
def probability_sigmoid(model, inputs, groups, delta, args):

    if args.scaled == True:
        if args.dataset == 'adult':
            args.alpha = 100
        if args.dataset == 'law':
            args.alpha = 50
        if args.dataset == 'acsincome':
            args.alpha = 50
        if args.dataset == 'dutch':
            args.alpha = 50

    threshold = 0.5
    inputs_s_1 = inputs[groups == 1]
    inputs_s_0 = inputs[groups == 0]

    y_hat_s_1 = model(inputs_s_1).view(-1)
    y_hat_s_0 = model(inputs_s_0).view(-1)

    t_hat_s_1 = args.alpha * (y_hat_s_1 - threshold)
    t_hat_s_0 = args.alpha * (y_hat_s_0 - threshold)

    r_s_1 = torch.mean(torch.sigmoid(t_hat_s_1))
    r_s_0 = torch.mean(torch.sigmoid(t_hat_s_0))

    c_1 = delta * r_s_0 - r_s_1
    c_2 = delta * r_s_1 - r_s_0
    return r_s_0, r_s_1, c_1, c_2



def probability_sigmoid_dp(model, inputs, groups, eps_smooth_dp):
    threshold = 0.5

    inputs_s_1 = inputs[groups == 1]
    inputs_s_0 = inputs[groups == 0]

    y_hat_s_1 = model(inputs_s_1).view(-1)
    y_hat_s_0 = model(inputs_s_0).view(-1)

    t_hat_s_1 = (y_hat_s_1 - threshold)
    t_hat_s_0 = (y_hat_s_0 - threshold)

    ratio_given_s_1 = torch.mean(torch.sigmoid(t_hat_s_1))
    ratio_given_s_0 = torch.mean(torch.sigmoid(t_hat_s_0))

    c = ratio_given_s_1 - ratio_given_s_0
    return c



def probability_DI(model, inputs, groups, eps_smooth, delta, args):

    if args.scaled == True:
        if args.dataset == 'adult':
            args.alpha = 100
        if args.dataset == 'law':
            args.alpha = 50
        if args.dataset == 'acsincome':
            args.alpha = 50
        if args.dataset == 'dutch':
            args.alpha = 50

    threshold = 0.5
    inputs_s_1 = inputs[(groups == 1)]
    inputs_s_0 = inputs[(groups == 0)]

    y_hat_s_1 = model(inputs_s_1).view(-1)
    y_hat_s_0 = model(inputs_s_0).view(-1)

    t_hat_s_1 = args.alpha * (y_hat_s_1 - threshold)
    t_hat_s_0 = args.alpha * (y_hat_s_0 - threshold)

    sum_s1 = torch.sum(1 - (1/2)*(1 - ((1/2)*(t_hat_s_1 + 0.5 + torch.sqrt((t_hat_s_1 + 0.5)**2 + eps_smooth))) + torch.sqrt((1 - ((1/2)*(t_hat_s_1 + 0.5 + torch.sqrt((t_hat_s_1 + 0.5)**2 + eps_smooth))))**2 + eps_smooth)))
    sum_s0 = torch.sum(1 - (1/2)*(1 - ((1/2)*(t_hat_s_0 + 0.5 + torch.sqrt((t_hat_s_0 + 0.5)**2 + eps_smooth))) + torch.sqrt((1 - ((1/2)*(t_hat_s_0 + 0.5 + torch.sqrt((t_hat_s_0 + 0.5)**2 + eps_smooth))))**2 + eps_smooth)))
    r_s_1 = torch.mean(1 - (1/2)*(1 - ((1/2)*(t_hat_s_1 + 0.5 + torch.sqrt((t_hat_s_1 + 0.5)**2 + eps_smooth))) + torch.sqrt((1 - ((1/2)*(t_hat_s_1 + 0.5 + torch.sqrt((t_hat_s_1 + 0.5)**2 + eps_smooth))))**2 + eps_smooth)))
    r_s_0 = torch.mean(1 - (1/2)*(1 - ((1/2)*(t_hat_s_0 + 0.5 + torch.sqrt((t_hat_s_0 + 0.5)**2 + eps_smooth))) + torch.sqrt((1 - ((1/2)*(t_hat_s_0 + 0.5 + torch.sqrt((t_hat_s_0 + 0.5)**2 + eps_smooth))))**2 + eps_smooth)))

    c_1 = delta * r_s_0 - r_s_1
    c_2 = delta * r_s_1 - r_s_0
    return sum_s0, sum_s1, r_s_0, r_s_1, c_1, c_2



# Equal Opportunity  --------------------------------------------------------
# ---------------------------------------------------------------------------
def covariance_eo(model, inputs, labels, groups):
    groups_given_y = groups[labels == 1]
    s_mean_given_y = groups[labels == 1].mean()
    y_hat = model(inputs).view(-1)
    y_hat_given_y = y_hat[labels == 1]
    c = torch.dot((groups_given_y - s_mean_given_y), (y_hat_given_y)) / len(groups_given_y)
    return c



def probability_eo(model, inputs, labels, groups, eps_smooth_eo, args):
    threshold = 0.5
    if args.scaled == True:
        if args.dataset == 'adult':
            args.alpha = 100
        if args.dataset == 'law':
            args.alpha = 50
        if args.dataset == 'acsincome':
            args.alpha = 50
        if args.dataset == 'dutch':
            args.alpha = 50

    inputs_y_1_s_1 = inputs[(groups == 1) & (labels == 1)]
    inputs_y_1_s_0 = inputs[(groups == 0) & (labels == 1)]

    y_hat_y_1_s_1 = model(inputs_y_1_s_1).view(-1)
    y_hat_y_1_s_0 = model(inputs_y_1_s_0).view(-1)

    t_y_1_s_1 = args.alpha * (y_hat_y_1_s_1 - threshold)
    t_y_1_s_0 = args.alpha * (y_hat_y_1_s_0 - threshold)

    prob_pos_y_1_s_1 = 0.5 * torch.mean(t_y_1_s_1 + torch.sqrt(t_y_1_s_1**2 + eps_smooth_eo))
    prob_pos_y_1_s_0 = 0.5 * torch.mean(t_y_1_s_0 + torch.sqrt(t_y_1_s_0**2 + eps_smooth_eo))

    c = prob_pos_y_1_s_1 - prob_pos_y_1_s_0
    return c


def probability_EI(model, inputs, labels, groups, eps_smooth, delta, args):
    if args.scaled == True:
        if args.dataset == 'adult':
            args.alpha = 100
        if args.dataset == 'law':
            args.alpha = 50
        if args.dataset == 'acsincome':
            args.alpha = 50
        if args.dataset == 'dutch':
            args.alpha = 50

    threshold = 0.5
    inputs_y_1_s_1 = inputs[(groups == 1) & (labels == 1)]
    inputs_y_1_s_0 = inputs[(groups == 0) & (labels == 1)]

    y_hat_y_1_s_1 = model(inputs_y_1_s_1).view(-1)
    y_hat_y_1_s_0 = model(inputs_y_1_s_0).view(-1)

    t_y_1_s_1 = args.alpha *(y_hat_y_1_s_1 - threshold)
    t_y_1_s_0 = args.alpha *(y_hat_y_1_s_0 - threshold)

    r_y_1_s_1 = torch.mean(1 - (1/2)*(1 - ((1/2)*(t_y_1_s_1 + 0.5 + torch.sqrt((t_y_1_s_1 + 0.5)**2 + eps_smooth))) + torch.sqrt((1 - ((1/2)*(t_y_1_s_1 + 0.5 + torch.sqrt((t_y_1_s_1 + 0.5)**2 + eps_smooth))))**2 + eps_smooth)))
    r_y_1_s_0 = torch.mean(1 - (1/2)*(1 - ((1/2)*(t_y_1_s_0 + 0.5 + torch.sqrt((t_y_1_s_0 + 0.5)**2 + eps_smooth))) + torch.sqrt((1 - ((1/2)*(t_y_1_s_0 + 0.5 + torch.sqrt((t_y_1_s_0 + 0.5)**2 + eps_smooth))))**2 + eps_smooth)))

    c_1 = delta * r_y_1_s_0 - r_y_1_s_1
    c_2 = delta * r_y_1_s_1 - r_y_1_s_0

    return r_y_1_s_0, r_y_1_s_1, c_1, c_2
