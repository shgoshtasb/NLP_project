import torch
import numpy as np
from sklearn.linear_model import LogisticRegression

def split(sequence, parts):
    assert parts <= sequence.shape[0]
    part_size = int(np.floor(sequence.shape[0] * 1.0 / parts))
    return torch.split(sequence[:part_size * parts], part_size)

def get_equal_bins(probs, num_bins=10):
    """Get bins that contain approximately an equal number of data points."""
    sorted_probs, indices = torch.sort(probs.reshape(-1,))
    part_size = int(np.ceil(sorted_probs.shape[0] * 1.0 / num_bins))
    idx = torch.arange(0, probs.shape[0], part_size)
    bins = (sorted_probs[idx[1:] - 1] + sorted_probs[idx[1:]]) / 2.0
    if bins[-1] != 1.:
        bins = torch.cat([bins.to(torch.float32), torch.ones(1).cuda().to(torch.float32)], dim=0)
    return bins

def get_equal_prob_bins(probs, num_bins=10):
    return torch.linspace(0., 1., steps=num_bins+1)[1:].cuda()

def get_discrete_bins(data):
    sorted_probs, indices = torch.sort(data.reshape(-1,))
    dups = torch.nonzero(sorted_probs[:-1] - sorted_probs[1:] != 0).reshape(-1,)
    sorted_probs = torch.cat([sorted_probs[dups].reshape(-1,), sorted_probs[-1].reshape(-1,)], dim=0)
    bins = (sorted_probs[:-1] + sorted_probs[1:]) / 2.0
    if bins[-1] != 1.:
        bins = torch.cat([bins.to(torch.float32), torch.ones(1).cuda().to(torch.float32)], dim=0)
    return bins

def get_top_calibration_error_uncertainties(probs, labels, p=2, alpha=0.1):
    return get_calibration_error_uncertainties(probs, labels, p, alpha, mode='top-label')

def get_calibration_error_uncertainties(probs, labels, p=2, alpha=0.1, mode='marginal'):
    """Get confidence intervals for the calibration error.

    Args:
        probs: A torch tensor of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A torch tensor of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.2
        [lower, upper] represents the confidence interval. mid represents the median of
        the bootstrap estimates. When p is not 2 (e.g. for the ECE where p = 1), this
        can be used as a debiased estimate as well.
    """

    def ce_functional(data):
        probs = data[:, :-1]
        labels = data[:, -1]
        return get_calibration_error(probs, labels, p, debias=False, mode=mode)
    data = torch.cat([probs, labels.reshape(-1, 1)], dim=1)
    [lower, mid, upper] = bootstrap_uncertainty(data, ce_functional, num_samples=100, alpha=alpha)
    return [lower, mid, upper]

def get_top_calibration_error(probs, labels, p=2, debias=True):
    return get_calibration_error(probs, labels, p, debias, mode='top-label')


def get_calibration_error(probs, labels, p=2, debias=True, mode='marginal'):
    """Get the calibration error.

    Args:
        probs: A torch tensor of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A torch tensor of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated calibration error, a floating point value.
        The method first uses heuristics to check if the values came from a scaling
        method or binning method, and then calls the corresponding function. For
        more explicit control, use lower_bound_scaling_ce or get_binning_ce.
    """
    if is_discrete(probs):
        #print('discrete', probs[:10], labels[:10], p, debias, mode)
        return get_binning_ce(probs, labels, p, debias, mode=mode)
    else:
        return lower_bound_scaling_ce(probs, labels, p, debias, mode=mode)


def lower_bound_scaling_top_ce(probs, labels, p=2, debias=True, num_bins=15,
                               binning_scheme=get_equal_bins):
    return lower_bound_scaling_ce(probs, labels, p, debias, num_bins, binning_scheme,
                                  mode='top-label')


def lower_bound_scaling_ce(probs, labels, p=2, debias=True, num_bins=15,
                           binning_scheme=get_equal_bins, mode='marginal'):
    """Lower bound the calibration error of a model with continuous outputs.

    Args:
        probs: A torch tensor of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A torch tensor of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        num_bins: Integer number of bins used to estimate the calibration error.
        binning_scheme: A function that takes in a list of probabilities and number of bins,
            and outputs a list of bins. See get_equal_bins, get_equal_prob_bins for examples.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated lower bound for calibration error, a floating point value.
        For scaling methods we cannot estimate the calibration error, but only a
        lower bound.
    """
    return _get_ce(probs, labels, p, debias, num_bins, binning_scheme, mode=mode)


def get_binning_top_ce(probs, labels, p=2, debias=True, mode='marginal'):
    return get_binning_ce(probs, labels, p, debias, mode='top-label')


def get_binning_ce(probs, labels, p=2, debias=True, mode='marginal'):
    """Estimate the calibration error of a binned model.

    Args:
        probs: A torch tensor of shape (n,) or (n, k). If the shape is (n,) then
            we assume binary classification and probs[i] is the model's confidence
            the i-th example is 1. Otherwise, probs[i][j] is the model's confidence
            the i-th example is j, with 0 <= probs[i][j] <= 1.
        labels: A torch tensor of shape (n,). labels[i] denotes the label of the i-th
            example. In the binary classification setting, labels[i] must be 0 or 1,
            in the k class setting labels[i] is an integer with 0 <= labels[i] <= k-1.
        p: We measure the lp calibration error, where p >= 1 is an integer.
        debias: Should we try to debias the estimates? For p = 2, the debiasing
            has provably better sample complexity.
        mode: 'marginal' or 'top-label'. 'marginal' calibration means we compute the
            calibraton error for each class and then average them. Top-label means
            we compute the calibration error of the prediction that the model is most
            confident about.

    Returns:
        Estimated calibration error, a floating point value.
    """
    return _get_ce(probs, labels, p, debias, None, binning_scheme=get_discrete_bins, mode=mode)


def get_ece(probs, labels, debias=False, num_bins=15, mode='top-label'):
    return lower_bound_scaling_ce(probs, labels, p=1, debias=debias, num_bins=num_bins,
                                  binning_scheme=get_equal_prob_bins, mode=mode)


def _get_ce(probs, labels, p, debias, num_bins, binning_scheme, mode='marginal'):
    def ce_1d(probs, labels):
        data = torch.cat([probs.reshape(-1, 1), labels.reshape(-1, 1)], dim=1)
        if binning_scheme == get_discrete_bins:
            assert(num_bins is None)
            bins = binning_scheme(probs)
        else:
            bins = binning_scheme(probs, num_bins=num_bins)
            #print(bins)
        #print(bins, data[:10])
        if p == 2 and debias:
            #print('discrte')
            return unbiased_l2_ce(bin(data, bins))
        elif debias:
            return normal_debiased_ce(bin(data, bins), power=p)
        else:
            return plugin_ce(bin(data, bins), power=p)
    if mode != 'marginal' and mode != 'top-label':
        raise ValueError("mode must be 'marginal' or 'top-label'.")
    if len(labels.shape) != 1:
        raise ValueError('labels should be a 1D numpy array.')
    if probs.shape[0] != labels.shape[0]:
        raise ValueError('labels and probs should have the same number of entries.')
    if probs.shape[1] == 1:
        # If 1D (2-class setting), compute the regular calibration error.
        if torch.min(labels) != 0 or torch.max(labels) != 1:
            raise ValueError('If probs is 1D, each label should be 0 or 1.')
        #print('here')
        return ce_1d(probs, labels)
    elif probs.shape[1] > 1:
        if torch.min(labels) != 0 or torch.max(labels) != probs.shape[1] - 1:
            raise ValueError('labels should be between 0 and num_classes - 1.')
        if mode == 'marginal':
            labels_one_hot = get_labels_one_hot(labels, k=probs.shape[1])
            assert probs.shape == labels_one_hot.shape
            marginal_ces = []
            for k in range(probs.shape[1]):
                cur_probs = probs[:, k]
                cur_labels = labels_one_hot[:, k]
                marginal_ces.append(ce_1d(cur_probs, cur_labels) ** p)
            return torch.mean(marginal_ces) ** (1.0 / p)
        elif mode == 'top-label':
            preds = get_top_predictions(probs)
            correct = (preds == labels).to(torch.float32)
            confidences = get_top_probs(probs)
            return ce_1d(confidences, correct)
    else:
        raise ValueError('probs should be a 1D or 2D numpy array.')



def is_discrete(probs):
    if len(probs.shape) == 1:
        return enough_duplicates(probs)
    elif len(probs.shape) == 2:
        for k in range(probs.shape[1]):
            if not enough_duplicates(probs[:, k]):
                return False
        return True
    else:
        raise ValueError('probs must be a 1D or 2D numpy array.')


def enough_duplicates(array):
    # TODO: instead check that we have at least 2 values in each bin.
    num_bins = get_discrete_bins(array)
    if num_bins.shape[0] < array.shape[0] / 4.0:
        return True
    return False


# Functions that bin data.

def get_bin_all(pred_probs, bins):
    """Get the index of the bin that pred_probs belongs in."""
    return searchsorted(bins, pred_probs)

def tensor_searchsorted(sorted_seq, values_):
    #print('sortedsearch', sorted_seq.shape, values.shape)
    indices = []
    for values in values_.permute(1, 0):
        indices.append(searchsorted(sorted_seq, values.reshape(-1,)).reshape(-1, 1))
    return torch.cat(indices, dim=1)

def searchsorted(sorted_seq, values):
    #print('sortedsearch', sorted_seq.shape, values.shape)
    idx = torch.nonzero(sorted_seq.reshape(1, -1).to(torch.float32) >= values.reshape(-1, 1).to(torch.float32))
    dups = torch.nonzero(idx[:-1, 0] - idx[1:, 0] != 0).reshape(-1,)
    return torch.cat([idx[0,:].reshape(-1,2), idx[dups+1]], dim=0)[:, 1]


def bin(data, bins):
    return fast_bin(data, bins)


def fast_bin(prob_label, bins):
    #print('fast')
    bin_indices = searchsorted(bins, prob_label[:, 0])
    bin_sort_indices = torch.argsort(bin_indices)
    sorted_bins = bin_indices[bin_sort_indices]
    #print(sorted_bins.shape, bins.shape, prob_label.shape)
    splits = searchsorted(sorted_bins, torch.arange(1, bins.shape[0]).cuda())
    first_bunch = splits[0]
    last_bunch = sorted_bins.shape[0] - splits[-1]
    splits = splits[1:] - splits[:-1]
    splits = splits.tolist()
    splits.insert(0, first_bunch)
    splits.append(last_bunch)
    #print('splits', splits)
    binned_data = torch.split(prob_label[bin_sort_indices], splits)
    #print('binned_data', len(binned_data))
    return binned_data


def equal_bin(prob_label, num_bins):
    sorted_probs, indices = torch.sort(prob_label[:,0])
    return split(sorted_probs, num_bins)


def difference_mean(prob_label):
    """Returns average pred_prob - average label."""
    return prob_label[:, 0].mean() - prob_label[:, 1].mean()


def get_bin_probs(binned_data):
    bin_sizes = torch.tensor(list(map(len, binned_data))).to(torch.float32)
    num_data = bin_sizes.sum()
    bin_probs = list(map(lambda b: b * 1.0 / num_data, bin_sizes))
    return torch.tensor(bin_probs).cuda()


def plugin_ce(binned_data, power=2):
    def bin_error(data):
        if len(data) == 0:
            return 0.0
        return torch.abs(difference_mean(data)) ** power
    bin_probs = get_bin_probs(binned_data)
    bin_errors = torch.tensor(list(map(bin_error, binned_data))).cuda()
    return torch.matmul(bin_probs, bin_errors) ** (1.0 / power)


def unbiased_square_ce(binned_data):
    # Note, this is not the l2 CE. It does not take the square root.
    def bin_error(data):
        if len(data) < 2:
            return 0.0
            # raise ValueError('Too few values in bin, use fewer bins or get more data.')
        biased_estimate = torch.abs(difference_mean(data)) ** 2
        label_values = torch.tensor(list(map(lambda x: x[1], data))).cuda()
        mean_label = label_values.mean()
        variance = mean_label * (1.0 - mean_label) / (len(data) - 1.0)
        #print(mean_label, variance)
        return biased_estimate - variance
    bin_probs = get_bin_probs(binned_data)
    #print('bin_probs', bin_probs)
    bin_errors = torch.tensor(list(map(bin_error, binned_data))).cuda()
    #print('err', bin_errors)
    return torch.matmul(bin_probs, bin_errors)

def unbiased_l2_ce(binned_data) -> float:
    return torch.relu(unbiased_square_ce(binned_data)) ** 0.5


def normal_debiased_ce(binned_data, power=1, resamples=1000):
    bin_sizes = torch.tensor(list(map(len, binned_data))).cuda()
    if bin_sizes.min() <= 1:
        raise ValueError('Every bin must have at least 2 points for debiased estimator. '
                         'Try adding the argument debias=False to your function call.')
    label_means = torch.tensor(list(map(lambda b: b[:, 1].mean() for b in binned_data))).cuda()
    label_stddev = torch.sqrt(label_means * (1 - label_means) / bin_sizes)
    model_vals = torch.tensor(list(map(lambda b: b[:, 0].mean() for b in binned_data))).cuda()
    assert(label_means.shape == (len(binned_data),))
    assert(model_vals.shape == (len(binned_data),))
    ce = plugin_ce(binned_data, power=power)
    bin_probs = get_bin_probs(binned_data)
    resampled_ces = []
    for i in range(resamples):
        label_samples = torch.normal(mean=label_means, std=label_stddev)
        # TODO: we can also correct the bias for the model_vals, although this is
        # smaller.
        diffs = torch.pow(torch.abs(label_samples - model_vals), power)
        cur_ce = torch.pow(torch.matmul(bin_probs, diffs), 1.0 / power)
        resampled_ces.append(cur_ce)
    mean_resampled = torch.cat(resampled_ces, dim=0).mean()
    bias_corrected_ce = 2 * ce - mean_resampled
    return bias_corrected_ce


def eval_top_mse(calibrated_probs, probs, labels):
    correct = (get_top_predictions(probs) == labels)
    return torch.mean((calibrated_probs - correct)**2)


def eval_marginal_mse(calibrated_probs, probs, labels):
    assert calibrated_probs.shape == probs.shape
    k = probs.shape[1]
    labels_one_hot = get_labels_one_hot(labels, k)
    return ((calibrated_probs - labels_one_hot)**2).mean() * calibrated_probs.shape[1] / 2.0


def resample(data):
    indices = torch.tensor(np.random.choice(np.arange(data.shape[0]), size=data.shape[0], replace=True)).cuda()
    return data[indices]


def bootstrap_uncertainty(data, functional, estimator=None, alpha=10.0, 
                          num_samples=1000):
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    if estimator is None:
        estimator = functional
    estimate = estimator(data)
    plugin = functional(data)
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    return (plugin + estimate - np.percentile(bootstrap_estimates, 100 - alpha / 2.0),
            plugin + estimate - np.percentile(bootstrap_estimates, 50),
            plugin + estimate - np.percentile(bootstrap_estimates, alpha / 2.0))


def precentile_bootstrap_uncertainty(data, functional, estimator=None, alpha=10.0, num_samples=1000):
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    if estimator is None:
        estimator = functional
    plugin = functional(data)
    estimate = estimator(data)
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    bias = 2 * np.percentile(bootstrap_estimates, 50) - plugin - estimate
    return (np.percentile(bootstrap_estimates, alpha / 2.0) - bias,
            np.percentile(bootstrap_estimates, 50) - bias,
            np.percentile(bootstrap_estimates, 100 - alpha / 2.0) - bias)


def bootstrap_std(data, estimator=None, num_samples=100):
    """Return boostrap uncertained for 1 - alpha percent confidence interval."""
    bootstrap_estimates = []
    for _ in range(num_samples):
        bootstrap_estimates.append(estimator(resample(data)))
    return np.std(bootstrap_estimates)

def get_platt_scaler(model_probs, labels):
    clf = LogisticRegression(C=1e10, solver='lbfgs')
    eps = 1e-12
    #print(model_probs.shape, labels.shape)
    #model_probs = model_probs.reshape(-1,)
    
    model_probs = model_probs.cpu().numpy().astype(dtype=np.float64)
    labels = labels.cpu().numpy().reshape(-1,)
    if len(model_probs.shape) == 1:
        model_probs = np.expand_dims(model_probs, axis=-1)
    model_probs = np.clip(model_probs, eps, 1 - eps)
    model_probs = np.log(model_probs / (1 - model_probs))
    clf.fit(model_probs, labels)
    def calibrator(probs):
        x = probs
        x = torch.clamp(x, eps, 1 - eps)
        x = torch.log(x / (1 - x))
        x = x * torch.tensor(clf.coef_[0]).to(torch.float32).cuda() + torch.tensor(clf.intercept_).to(torch.float32).cuda()
        output = torch.sigmoid(x)
        return output.reshape(-1, 1)
    return calibrator

def gettorch_platt_scaler(model_probs, labels):
    class LogisticRegression(torch.nn.Module):
        def __init__(self, input_dim, output_dim):
            super(LogisticRegression, self).__init__()
            self.linear = torch.nn.Linear(input_dim, output_dim)

        def forward(self, x):
            outputs = self.linear(x)
            return outputs    
    input_dim = model_probs.shape[1]
    output_dim = 1
    model = LogisticRegression(input_dim, output_dim)    
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-10)
    epochs = 10000
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(model_probs)
        loss = criterion(torch.sigmoid(outputs), labels)
        loss.backward()
        optimizer.step()
        #if epoch % 1000 == 0:
        #    print(epoch, loss)
    model.eval()
    def calibrator(probs):
        return torch.sigmoid(model(probs)).reshape(-1,)
    return calibrator
        
        
def get_histogram_calibrator(model_probs, values, bins):
    binned_values = [[]] * len(bins)
    bin_idx = get_bin_all(model_probs, bins).reshape(-1, 1)
    binned_values = torch.zeros(model_probs.shape[0], bins.shape[0]).cuda()
    binned_values.scatter_(1, bin_idx, values.to(torch.float32).reshape(-1, 1) + 1)
    count = get_labels_one_hot(bin_idx, bins.shape[0]).sum(dim=0)
    safe_mean = ((bins[1:] + bins[:-1]) / 2.0).to(torch.float32)
    safe_mean = torch.cat([torch.tensor([bins[0] / 2.0]).to(torch.float32).cuda(), safe_mean], dim=0)
    bin_means = (count > 0.).to(torch.float32) * (binned_values.sum(dim=0) / count - 1.) + (count == 0).to(torch.float32) * safe_mean
    def calibrator(probs):
        indices = searchsorted(bins, probs)
        return bin_means[indices].reshape(-1,1)
    return calibrator


def get_discrete_calibrator(model_probs, bins):
    return get_histogram_calibrator(model_probs, model_probs, bins)


def get_top_predictions(probs):
    return torch.argmax(probs, dim=1)


def get_top_probs(probs):
    return torch.max(probs, dim=1)[0]


def get_accuracy(probs, labels):
    return (labels == predictions).mean()


def get_labels_one_hot(labels, k):
    return torch.eye(k).cuda()[labels.reshape(-1,)]
