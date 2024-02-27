import numpy as np
from math import log, sqrt, exp


def bit_count(quants):
    N1 = np.floor(np.log2(quants[quants > 1]))
    quants_prime = quants[quants > 1] - 2**N1
    N2 = np.ceil(np.log2(quants_prime[quants_prime > 1]))
    sr = np.sum(N1) + np.sum(N2) + np.sum(quants <= 1) + np.sum(quants_prime <= 1)
    return sr


def variable_multishift_quantization(sigma, x):

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
    quants = np.round(x_prime / step_size - u2)
    bits = bit_count(quants) + np.sum(np.zeros_like(x) + 1)
    dequants = (quants + u2) * step_size * np.sign(x) + bias
    return dequants, sigma, bits


def multishift_quantization(s, x):
    max_x = np.max(np.abs(x))
    if np.isclose(max_x, 0):
        return x
    # convert to sigma
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
    dequants = (quants + u2) * step_size * max_x * np.sign(x) + bias
    return dequants, sigma * max_x, bits


def SQuantization(s, x):

    if s == 0:
        return x
    norm_x = np.linalg.norm(x, 2)
    if norm_x == 0:
        return x
    ratio = np.abs(x) / norm_x
    l = np.floor(ratio * s)
    p = ratio * s - l
    sampled = np.random.binomial(1, p)
    quants = l + sampled
    bits = bit_count(quants) + np.sum(np.zeros_like(x) + 1)
    qtzt = np.sign(x) * norm_x * quants / s
    return qtzt, bits


def generate_data(b, d, n, iid, iid_level, sigma, mu):
    y = []
    labels = []
    if iid == "True":
        for i in range(b):
            y.append(np.random.normal(mu, sigma, size=(n[i], d)))
            labels.append(i * np.ones(n[i]))
    else:
        for i in range(b):
            mu_i = np.random.normal(mu, iid_level)
            y.append(np.random.normal(mu_i, sigma, size=(n[i], d)))
            labels.append(i * np.ones(n[i]))

    return y, labels


def com_bits(s, d):
    if s <= np.sqrt(d / 2 - np.sqrt(d)):
        return (3 + (3 / 2) * np.log(2 * (s**2 + d) / (s * (s + np.sqrt(d))))) * s * (
            s + np.sqrt(d)
        ) + 32
    elif s == np.sqrt(d):
        return 2.8 * d + 32
    else:
        return (
            (1 / 2) * (np.log(1 + (s**2 + np.minimum(d, s * np.sqrt(d))) / d) + 1) + 2
        ) * d + 32


def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum(np.exp(x))
    return ex / sum_ex


def generate_synthetic(alpha, beta, iid, d, nb_class, b):

    np.random.seed(1994)
    # samples_per_user = np.random.lognormal(4, 2, (b)).astype(int) + 50
    samples_per_user = np.random.randint(low=10, high=50, size=b).astype(int)
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(b)]
    y_split = [[] for _ in range(b)]

    #### define some eprior ####
    np.random.seed(1994)
    mean_W = np.random.normal(0, alpha, b)
    mean_b = mean_W
    np.random.seed(1994)
    B = np.random.normal(0, beta, b)
    mean_x = np.zeros((b, d))

    diagonal = np.zeros(d)
    for j in range(d):
        diagonal[j] = np.power((j + 1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(b):
        if iid == 1:
            mean_x[i] = np.ones(d) * B[i]  # all zeros
        else:
            np.random.seed(1994)
            mean_x[i] = np.random.normal(B[i], 1, d)

    if iid == 1:
        np.random.seed(1994)
        W_global = np.random.normal(0, 1, (d, nb_class))
        np.random.seed(1994)
        b_global = np.random.normal(0, 1, nb_class)

    for i in range(b):
        np.random.seed(1994)
        W = np.random.normal(mean_W[i], 1, (d, nb_class))
        np.random.seed(1994)
        b = np.random.normal(mean_b[i], 1, nb_class)

        if iid == 1:
            W = W_global
            b = b_global
        np.random.seed(1994)
        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])

        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))

        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

    return X_split, y_split, samples_per_user
