import gpflow
import matplotlib.pyplot as plt
import numpy as np
import scipy
from determinental_sample_GP import det_sample_GP as sample_points
import matplotlib

np.random.seed(0)
jitter = 1e-10
low_jitter = gpflow.settings.get_settings()
low_jitter.numerics.jitter_level = jitter

num_trials = 3

N_sequence = np.array([int(100 * 1.3 ** pow) for pow in range(15)])
sn = .1
k = gpflow.kernels.RBF(1)
lengthscale = .5
M = 15

all_gaps = np.zeros((num_trials, len(N_sequence)))
all_ts = np.zeros_like(all_gaps)
for i in range(num_trials):
    gaps = []
    ts = []
    X = 5 * np.random.rand(N_sequence[-1])[:, None]
    Kff = k.compute_K_symm(X)
    Y = np.random.multivariate_normal(mean=np.zeros(N_sequence[-1]), cov=Kff + sn * np.eye(N_sequence[-1]))[:, None]
    for N in N_sequence:
        X_cur = X[:N, :]
        Z_cur, ind = sample_points(X_cur, k, M)
        Y_cur = Y[:N, :]
        # If jitter isn't very low, difficult for t->0 (we basically add M*jitter to t) M*jitter *\|y\|^2 may be large
        with gpflow.settings.temp_settings(low_jitter):
            Kuu = k.compute_K_symm(Z_cur)
            Kuf = k.compute_K(Z_cur, X_cur)
            L = np.linalg.cholesky(Kuu + jitter * np.eye(len(Z_cur)))
            LinvKuf = scipy.linalg.solve_triangular(L, Kuf, lower=True)
            # t= tr(Kff-Qff)
            t = N * k.variance.value - np.sum(np.square(LinvKuf))
            ts.append(t / 2 / np.square(sn))
            # Fit full model and compute ML
            full_m = gpflow.models.GPR(X_cur, Y_cur, k)
            full_m.likelihood.variance = np.square(sn)
            full_m.kern.lengthscales = lengthscale
            ml = full_m.compute_log_likelihood()
            # Fit sparse model and compute ELBO
            m = gpflow.models.SGPR(X_cur, Y_cur, kern=k, Z=Z_cur)
            m.likelihood.variance = np.square(sn)
            m.kern.lengthscales = lengthscale
            elbo = m.compute_log_likelihood()
            gap = ml - elbo
        gaps.append(gap)
    all_gaps[i, :] = gaps
    all_ts[i, :] = ts

# Make plot
matplotlib.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)
plt.figure()
plt.plot(N_sequence, np.mean(all_gaps, axis=0))
plt.plot(N_sequence, np.mean(all_ts, axis=0))
plt.xlabel("Number of Data Points")
plt.ylabel("KL divergence")
plt.legend(["Actual KL divergence", "$t/(2\sigma^2)$"])
plt.tight_layout()
plt.show()


