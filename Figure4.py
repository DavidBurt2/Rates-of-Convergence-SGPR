import gpflow
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from dppy.finite_dpps import FiniteDPP

#from determinental_sample_GP import det_sample_GP as sample_points
from KL_Bounds import KL_bound,KL_bound2

# set seed for reproducibility
np.random.seed(3)


# We reduce the jitter, note that the jitter will at some point have a large impact on the magnitude of the trace term
jitter = 1e-10
low_jitter = gpflow.settings.get_settings()
low_jitter.numerics.jitter_level = jitter

#dimensions, trials, data set sizes
num_trials = 1
N_min = 10000
N_max = 150000
N_step = 10000
N_sequence = np.array([int(100 * 1.3 ** pow) for pow in range(15)])

# noise standard deviation, noise, kernel, lengthscale, we chose a large noise for illustrative purposes
sn = .5
epsilon = np.random.randn(N_sequence[-1], 1) * sn
k = gpflow.kernels.RBF(1)
lengthscale = .5

# number of inducing points to use for each data set
M_sequence = np.linspace(5, 33, 15)
M_sequence = [int(M) for M in M_sequence]

all_gaps = np.zeros((num_trials, len(N_sequence)))
all_t_gaps = np.zeros((num_trials, len(N_sequence)))
avg_kls = np.zeros((num_trials, len(N_sequence)))
for i in range(num_trials):
    gaps = []
    t_gap = []
    avgs = []
    # Generate a dataset of size Max N, Y is sampled from the prior
    X = np.random.randn(N_sequence[-1],1)
    Kff = k.compute_K_symm(X)
    Y = np.random.multivariate_normal(mean=np.zeros(N_sequence[-1]), cov=Kff + np.square(sn) * np.eye(N_sequence[-1]))[:, None]

    for N, M in zip(N_sequence, M_sequence):
        X_cur = X[:N, :]
        kff= k.compute_K_symm(X_cur)
        DPP = FiniteDPP('likelihood', **{'L': kff+np.eye(N)*1e-6})
        DPP.flush_samples()
        DPP.sample_exact_k_dpp(size=M)
        ind = DPP.list_of_samples[0]
        Z_cur = X_cur[ind]
        Y_cur = Y[:N, :]
        with gpflow.settings.temp_settings(low_jitter):
            # bound from theorem 4
            avg_kl = KL_bound2(k_var=k.variance.value, k_ls=lengthscale, sigma_n=sn, N=N, p_sd=1, p_success=0.5, M=M)
            # We set the GP to have the parameters used in generating data
            full_m = gpflow.models.GPR(X_cur, Y_cur, k)
            full_m.likelihood.variance = np.square(sn)
            full_m.kern.lengthscales = lengthscale
            ml = full_m.compute_log_likelihood()
            # We set the sparse GP to have the parameters used in generating data
            m = gpflow.models.SGPR(X_cur, Y_cur, kern=k, Z=Z_cur)
            m.likelihood.variance = np.square(sn)
            m.kern.lengthscales = lengthscale
            # Titsias upper bound
            upper = m.compute_upper_bound()
            # ELBO
            elbo = m.compute_log_likelihood()
            # Actual Gap
            gap = ml - elbo
            # upper - lower
            titsias_gap = upper - elbo
        gaps.append(gap)
        t_gap.append(titsias_gap)
        avgs.append(avg_kl)

    all_gaps[i, :] = gaps
    all_t_gaps[i, :] = t_gap
    avg_kls[i, :] = avgs

#Make plot
matplotlib.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)
plt.figure(figsize=(10, 5))
ax1 = plt.subplot(2, 1, 1)
ax1.plot(N_sequence, all_gaps.mean(0))
ax1.plot(N_sequence, all_t_gaps.mean(0))
ax1.plot(N_sequence, avg_kls.mean(0))
ax1.set_ylim([0, 40])
ax1.set_ylabel("KL Divergence")
ax1.legend(["Actual KL Divergence", "$\mathcal{L}_{upper}-\mathcal{L}_{lower}$" ,"Theorem 4, p=.5"])

ax2 = plt.subplot(2, 1, 2,sharex = ax1)
ax2.plot(N_sequence, M_sequence)
ax2.set_xlabel("Number of Data Points")
ax2.set_ylabel("M")
ax2.set_xlim([0, 3900])
plt.tight_layout()
plt.subplots_adjust(top=0.966,
bottom=0.158,
left=0.075,
right=0.984,
hspace=0.234,
wspace=0.22)
plt.savefig("Figure4.pdf")

