import numpy as np


def truncated_normal_(tensor, mean=0, std=1):
    """
    Implemented by @ruotianluo
    See https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor


def get_fixed_temperature(temper, i, N, adapt):
    """A function to set up different temperature control policies"""
    # N = 5000
    if adapt == 'no':
        temper_var_np = 1.0  # no increase, origin: temper
    elif adapt == 'lin':
        temper_var_np = 1 + i / (N - 1) * (temper - 1)  # linear increase
    elif adapt == 'exp':
        temper_var_np = temper ** (i / N)  # exponential increase
    elif adapt == 'log':
        temper_var_np = 1 + (temper - 1) / np.log(N) * np.log(i + 1)  # logarithm increase
    elif adapt == 'sigmoid':
        temper_var_np = (temper - 1) * 1 / (1 + np.exp((N / 2 - i) * 20 / N)) + 1  # sigmoid increase
    elif adapt == 'quad':
        temper_var_np = (temper - 1) / (N - 1) ** 2 * i ** 2 + 1
    elif adapt == 'sqrt':
        temper_var_np = (temper - 1) / np.sqrt(N - 1) * np.sqrt(i) + 1
    else:
        raise Exception("Unknown adapt type!")
    return temper_var_np


def write_tokens(filename, tokens):
    """Write word tokens to a local file (For Real data)"""
    with open(filename, 'w') as f:
        for sent in tokens:
            f.write(' '.join(sent))
            f.write('\n')
