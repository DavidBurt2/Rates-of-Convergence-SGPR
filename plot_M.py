# define model,input distribution,amount of data, bound on norm of y, and M, this returns the corresponding bound on the KL-divergence that holds with probability
# success
import gpflow
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from determinental_sample_GP import det_sample_GP as sample

matplotlib.rcParams.update({'font.size': 30})

jitter = 1e-10
low_jitter = gpflow.settings.get_settings()
low_jitter.numerics.jitter_level = jitter


# TODO: Import KL_Bounds from other file, redo this plot with less noise maybe.
def KL_bound(k_var, k_ls, sigma_n, N, p_sd, p_success, bound_y, M):
    # calculations
    a = 1. / (4 * np.square(p_sd))
    b = 1. / (2 * np.square(k_ls))
    c = np.sqrt(np.square(a) + 2 * a * b)
    A = a + b + c
    B = b / A
    delta = 1 - p_success
    first_term = (M + 1) * np.power(B, M) * N * k_var * np.sqrt(2 * a) / (
            2 * np.sqrt(A) * np.square(sigma_n) * delta * (1 - B))
    second_term = 1 + bound_y / np.square(sigma_n)
    return first_term * second_term


np.random.seed(1)
lengthscale = .3

# this is high for illustrative purposes, with low variance can't get enough numerical precision
# for L_upper-L_lower to go to zero and still compute Cholesky, in general this bound is very
# sensitive to jitter. T should be monotonically decreasing, but we basically add, hitter*M to it
sn = .3
k = gpflow.kernels.RBF(1)
k.lengthscales = lengthscale
N = 1000
M = 100
X = np.random.randn(N, 1)
Kff = k.compute_K_symm(X)
Y = np.random.multivariate_normal(mean=np.zeros(N), cov=Kff + sn * np.eye(N))[:, None]

# compute the full marginal likelihood
full = gpflow.models.GPR(X, Y, k)
full.likelihood.variance = np.square(sn)
full.kern.lengthscales = lengthscale
ml = full.compute_log_likelihood()

# draw a sample of M_max inducing points, we reuse these to ensure monotonicity
Zs, ind = sample(X, k, M)

gaps = list()
avg_50 = list()
avg_99 = list()
bound_50 = list()
bound_99 = list()
titsias = list()
ms = np.arange(10, 100, 2)
for m in ms:
    # low jitter, otherwise hard to see asymptotic behavior
    with gpflow.settings.temp_settings(low_jitter):
        # fir sparse model
        Det_Init_M = gpflow.models.SGPR(X, Y, kern=k, Z=Zs[:m, :])
        Det_Init_M.likelihood.variance = np.square(sn)
        Det_Init_M.kern.lengthscales = lengthscale
        # compute elbo
        elbo = Det_Init_M.compute_log_likelihood()
        # true KL-divergence
        gaps.append(ml - elbo)
        # bounds from theorem 3
        bound_50.append(
            KL_bound(k_var=1, k_ls=lengthscale, sigma_n=sn, N=1000, p_sd=1, p_success=.5, bound_y=2 * N, M=m))
        bound_99.append(
            KL_bound(k_var=1, k_ls=lengthscale, sigma_n=sn, N=1000, p_sd=1, p_success=.99, bound_y=2 * N, M=m))
        titsias.append(Det_Init_M.compute_upper_bound() - elbo)
        # bounds from theorem 4
        avg_50.append(2 * KL_bound(k_var=1, k_ls=lengthscale, sigma_n=sn, N=1000, p_sd=1, p_success=.5, bound_y=0, M=m))
        avg_99.append(
            2 * KL_bound(k_var=1, k_ls=lengthscale, sigma_n=sn, N=1000, p_sd=1, p_success=.99, bound_y=0, M=m))

# plotting
plt.rc('font', size=18)
plt.rc('text', usetex=True)
plt.plot(ms, gaps)
plt.plot(ms, titsias)
plt.plot(ms, avg_50, ls=":")
plt.plot(ms, avg_99, ls=":")

plt.plot(ms, bound_50, ls="--")
plt.plot(ms, bound_99, ls="--")

plt.ylim([0., 10.])
plt.xlim([10, 100])
plt.legend(["Actual KL Divergence", "$\mathcal{L}_{upper}-\mathcal{L}_{Lower}$", "Theorem 4, p=.5", "Theorem 4, p=.99",
            "Theorem 3, p=.5", "Theorem 3, p=.99"])
plt.ylabel("KL Divergence")
plt.xlabel("Number of Inducing points")
plt.show()
