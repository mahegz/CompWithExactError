# Import Libraries
import numpy as np
from math import sqrt
from utils import SQuantization
from utils import multishift_quantization
from utils import variable_multishift_quantization


def multishift_subsampled_gaussian(X, sigma, gamma, compression_param,c):
    """ Simple individual multishift scheme
    Params:
    @X (n x d array): matrix of n clients local data
    @seed      (int): random seed for reproducibility
    @sigma   (float): noise magnitude
    @gamma   (float): subsampling rate (so that d*gamma will be the averaged communication cost)
    @s         (int): compression parameter
    Returns:
    @mu_estimate (n x d array): Distributed Mean Estimate
    @bits              (float): communicated bits
    """

    # Get (n,d) shape parameters
    (n, d) = X.shape
    # Draw the sampling mask for active clients
    sampling_mat = np.random.binomial(1, gamma * np.ones(X.shape))
    # Select active clients
    clients_vecs = [X[i, :][sampling_mat[i, :] == 1] for i in range(n)]
    # Run multishift compression
    clients_quants = [
        multishift_quantization(x, compression_param) for x in clients_vecs
    ]
    sigma_arr = np.array([i[1] for i in clients_quants])
    # Compute number of communicated bits
    bits = sum([i[2] for i in clients_quants])
    # Mean estimate and privacy
    X_reconst = np.zeros((n, d))
    for i in range(n):
        X_reconst[i, :][sampling_mat[i, :] == 1] = clients_quants[i][0]
    mu_estimate = np.mean(X_reconst, axis=0) / gamma
    # Ensuring Privacy
    for i in range(d):
        active = np.sum(sampling_mat[:, i] == 1)
        if active == 0:
            mu_estimate[i] = sigma * np.random.normal()
        else:
            sigma_eff = np.sqrt(np.sum(sigma_arr[sampling_mat[:, i] == 1] ** 2)) / (
                n * gamma
            )
            if sigma > sigma_eff:
                mu_estimate[i] += (
                    np.sqrt(sigma ** 2 - sigma_eff ** 2) * np.random.normal()
                )
    return mu_estimate, bits


def CSGM(X, sigma=1, gamma=0.1, s=2 ** 4):
    """ Coordinate Subsampling Gaussian Mechanism from (Chen et al.2023)
    Params:
    @X (n x d array): matrix of n clients local data
    @seed      (int): random seed for reproducibility
    @gamma   (float): subsampling rate (so that d*gamma will be the averaged communication cost)
    @sigma   (float): noise magnitude
    @s         (int): compression parameter
    Returns:
    @X_reconst_mean (n x d array): Distributed Mean Estimate
    @bits                 (float): communicated bits
    """
    # Random seed for reproducibility
    # Get (n,d) shape parameters
    (n, d) = X.shape
    # Draw the sampling mask for active clients
    sampling_mat = np.random.binomial(1, gamma * np.ones(X.shape))
    # Compression step
    clients_vecs = [X[i, :][sampling_mat[i, :] == 1] for i in range(n)]
    clients_quants = [SQuantization(x, s) for x in clients_vecs]
    # Compute number of bits communicated
    bits = 0
    for i in range(n):
        if len(clients_quants[i]) > 0:
            bits += clients_quants[i][1]
    # Mean estimate and Privacy
    X_reconst = np.zeros((n, d))
    for i in range(n):
        if np.sum(sampling_mat[i, :]) == 0:
            continue
        else:
            X_reconst[i, :][sampling_mat[i, :] == 1] = clients_quants[i][0]
    mu_estimate = (np.mean(X_reconst, axis=0) / gamma) + sigma * np.random.normal(
        size=(d,)
    )
    return mu_estimate, bits


def SIGM(X, sigma, gamma):
    """ Subsampled Individual Gaussian Mechanism 
    Params:
    @X (n x d array): matrix of n clients local data
    @seed      (int): random seed for reproducibility
    @sigma   (float): noise magnitude
    @gamma   (float): subsampling rate (so that d*gamma will be the averaged communication cost)
    Returns:
    @X_priv (n x d array): Distributed Mean Estimate
    @bits         (float): communicated bits
    """
    # Random seed for reproducibility
    bits = 0
    # Get (n,d) shape parameters
    (n, d) = X.shape
    # Draw the sampling mask for active clients
    sampling_mat = np.random.binomial(1, gamma * np.ones(X.shape))
    clients_cordinates = [X[:, i][sampling_mat[:, i] == 1] for i in range(d)]
    # Mean estimate and Privacy
    X_priv = np.zeros((d,))
    for i in range(d):
        active_num = np.sum(sampling_mat[:, i] == 1)
        if active_num == 0:
            X_priv[i] = sigma * np.random.normal()
        else:
            comp_x = clients_cordinates[i]
            r ,b = variable_multishift_quantization(
                x=comp_x, sigma=n * gamma * sigma / sqrt(active_num)
            )
            X_priv[i] = np.sum(r) / (n * gamma)
            bits += b
    return X_priv, bits


def subsampled_gaussian(X, sigma, gamma, c=1.0):
    """ Subsampled Gaussian Mechanism from (Chen et al.2023)
    Params:
    @X (n x d array): matrix of n clients local data
    @seed      (int): random seed for reproducibility
    @sigma   (float): noise magnitude
    @gamma   (float): subsampling rate (so that d*gamma will be the averaged communication cost)
    @c       (float): clipping parameter
    Returns:
    @X_priv (n x d array): Distributed Mean Estimate
    """
    # Get (n,d) shape parameters
    (n, d) = X.shape
    # Randomized rounding
    X_rounded = (
        (np.random.binomial(1, (np.clip(X, -c, c) + c) / (2 * c)) - 1 / 2) * 2 * c
    )
    # Poisson sampling
    sampling_mat = np.random.binomial(1, gamma * np.ones(X.shape))
    X_sampled = X_rounded * sampling_mat
    # Perturbation
    X_priv = np.mean(X_sampled, axis=0) / gamma + np.random.normal(0, sigma, size=d)
    return X_priv
