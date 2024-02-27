# Import Libraries
from math import log, sqrt
import numpy as np
from dp_accounting import dp_event
from dp_accounting.rdp import rdp_privacy_accountant


def bit_count(quants):
    """ Tool function to count number of bits """
    N1 = np.floor(np.log2(quants[quants > 1]))
    quants_prime = quants[quants > 1] - 2 ** N1
    N2 = np.ceil(np.log2(quants_prime[quants_prime > 1]))
    sr = np.sum(N1) + np.sum(N2) + np.sum(quants <= 1) + np.sum(quants_prime <= 1)
    return sr


def variable_multishift_quantization(x, sigma):   
    u1 = np.random.uniform(size=x.shape)
    u2 = np.random.uniform(size=x.shape)
    n1 = sigma * np.random.normal(size=x.shape)
    # Stepsize calculations
    y_val = np.exp(-((n1 / sigma) ** 2) / 2) * u1
    y_val[n1 <= 0] = 1 - y_val[n1 <= 0]
    left = -sigma * np.sqrt(-2 * np.log(1 - y_val))
    right = sigma * np.sqrt(-2 * np.log(y_val))
    step_size = right - left
    bias = (right + left) / 2
    # Quantization
    x_prime = np.abs(x)
    s = np.sign(x)
    s[s == 0] = 1
    quants = np.ceil(x_prime / step_size + u2 - 0.5)
    bits = bit_count(quants) + np.sum(np.zeros_like(x) + 1)
    dequants = (quants - u2) * step_size * s + bias
    return dequants, bits


def multishift_quantization(x, s):
    """ Shifted Layered Quantizer
    Params:
    @x  (array): vector to quantize
    @s    (int): compression parameter, default s=2^3
    """
    max_x = np.max(np.abs(x))
    if np.isclose(max_x, 0):
        return x
    # Convert to sigma
    s_prime = s - 2
    sigma = 1 / (2 * (s_prime) * sqrt(log(4)))
    # Sampling
    u1 = np.random.uniform(size=x.shape)
    u2 = np.random.uniform(size=x.shape)
    n1 = sigma * np.random.normal(size=x.shape)
    # Stepsize calculations
    y_val = np.exp(-((n1 / sigma) ** 2) / 2) * u1
    y_val[n1 <= 0] = 1 - y_val[n1 <= 0]
    left = -sigma * np.sqrt(-2 * np.log(1 - y_val))
    right = sigma * np.sqrt(-2 * np.log(y_val))
    step_size = right - left
    bias = (right + left) / 2
    # Quantization
    x_prime = np.abs(x) / max_x
    quants = np.round(x_prime / step_size - u2)
    bits = bit_count(quants) + np.sum(np.zeros_like(x) + 1)
    s = np.sign(x)
    s[s == 0] = 1
    dequants = (quants + u2) * step_size * max_x * s + bias
    return dequants, sigma * max_x, bits


def SQuantization(x, s):
    """ Simple Randomized Quantizer
    Params:
    @x  (array): vector to quantize
    @s    (int): compression parameter, default s=2^3
    """
    if s == 0:
        return x
    norm_x = np.linalg.norm(x, 2)
    if norm_x == 0:
        return x
    max_x = np.max(np.abs(x))
    #from -1 to 1 
    ratio = x / max_x
    #quant
    step_size = 2/s
    quants  = np.floor(ratio/step_size + np.random.uniform(size=ratio.shape))
    bits = bit_count(np.abs(quants)) + np.sum(np.zeros_like(x) + 1)
    qtzt = quants*step_size*max_x
    return qtzt, bits


def l2_projection(X, norm=1.0):
    return norm * X / max(np.norm(X), norm)


RDP_ORDERS = np.array(list(range(2, 129)) + [256.0])


def SGM_rdp(n, d, gamma, sigma, T=1, c=1.0):
    """
  Compute the overall Renyi DP eps(alpha).
  """
    noise_multiplier = sigma / (c / (n * gamma))
    # print(f'noise multiplier: %f' %noise_multiplier)
    sampling_probability = gamma
    count = T * d
    event = dp_event.SelfComposedDpEvent(
        dp_event.PoissonSampledDpEvent(
            sampling_probability, dp_event.GaussianDpEvent(noise_multiplier)
        ),
        count,
    )
    accountant = rdp_privacy_accountant.RdpAccountant(orders=RDP_ORDERS)
    accountant.compose(event)

    return accountant._rdp


def get_SGM_sigma_from_rdp(n, d, gamma, eps_target, T=1, 
                           delta=10 ** -6, c=1.0, itr=40):
    """
  Compute `sigma' for a given (eps, delta) requirement.
  """
    # Perform binary search for the optimal sigma
    sigma_lo = 0.0
    sigma_high = 50.0

    for _itr in range(itr):
        sigma = (sigma_lo + sigma_high) / 2
        rdps = SGM_rdp(n, d, gamma, sigma, T, c)
        eps, _ = rdp_privacy_accountant.compute_epsilon(RDP_ORDERS, rdps, delta)
        if eps <= eps_target:
            sigma_high = sigma
        else:
            sigma_lo = sigma
    return sigma_high
