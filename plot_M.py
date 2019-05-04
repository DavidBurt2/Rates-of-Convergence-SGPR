import numpy as np


def KL_bound(k_var, k_ls, sigma_n, N, p_sd, p_success, bound_y, M):
    """
    Bound from theorem 3 for 1D normally distributed data with a SE-kernel

    :param k_var: kernel variance
    :param k_ls: kernel lengthscale
    :param sigma_n: likelihood noise standard deviation
    :param N: number of training data
    :param p_sd: input standard devation
    :param p_success: probability with which the bound holds (1-delta)
    :param bound_y: An upper bound on the magnitude of \|y\|^2
    :param M: number of inducing points
    :return: Bound from theorem 3
    """
    a = 1. / (4 * np.square(p_sd))
    b = 1. / (2 * np.square(k_ls))
    c = np.sqrt(np.square(a) + 2 * a * b)
    A = a + b + c
    B = b / A
    delta = 1 - p_success
    eigenvalues_sum = k_var * np.sqrt(2 * a / A) * np.power(B, M) / (1 - B)
    two_delta_sn_sq = 2 * delta * np.square(sigma_n)
    first_term = (M+1) * N * eigenvalues_sum / two_delta_sn_sq
    second_term = 1 + bound_y / np.square(sigma_n)
    return first_term * second_term

def KL_bound2(k_var, k_ls, sigma_n, N, p_sd, p_success, M):
    """
    Bound from theorem 4 for 1D normally distributed data with a SE-kernel
    :param k_var: kernel variance
    :param k_ls: kernel lengthscale
    :param sigma_n: noise standard deviation
    :param N: Number of training points
    :param p_sd: measure according to which data is distributed
    :param p_success: 1 - delta
    :param M: number of inducing points
    :return: Bound from theorem 4
    """
    # calculations
    a = 1. / (4 * np.square(p_sd))
    b = 1. / (2 * np.square(k_ls))
    c = np.sqrt(np.square(a) + 2 * a * b)
    A = a + b + c
    B = b / A
    delta = 1 - p_success
    delta_sn_sq = delta * np.square(sigma_n)
    eigenvalues_sum = k_var * np.sqrt(2 * a / A) * np.power(B, M) / (1 - B)
    return (M+1) * N * eigenvalues_sum / delta_sn_sq


