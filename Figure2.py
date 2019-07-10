import matplotlib.pyplot as plt
import numpy as np
import gpflow
from dppy.finite_dpps import FiniteDPP


# X \sim N(0, I)
X = np.random.randn(5000, 2)
kern = gpflow.kernels.RBF(2)
kern.lengthscales = 2
kff = kern.compute_K_symm(X)

kern2 = gpflow.kernels.RBF(2)
kern2.lengthscales = .5
kff2 = kern2.compute_K_symm(X)

M = 50
# Sample points from K-dpp determined by first kernel
DPP = FiniteDPP('likelihood', **{'L': kff})
DPP.flush_samples()
DPP.sample_exact_k_dpp(size=M)
DPP.sample_exact_k_dpp(size=M)
DPP.sample_exact_k_dpp(size=M)
DPP2 = FiniteDPP('likelihood', **{'L': kff2})
DPP2.flush_samples()
DPP2.sample_exact_k_dpp(size=M)
DPP2.sample_exact_k_dpp(size=M)
DPP2.sample_exact_k_dpp(size=M)

samp1 = X[DPP.list_of_samples[0]]#determinental_sample_GP.det_sample_GP(X, kern, M)
samp2 = X[DPP.list_of_samples[1]]#determinental_sample_GP.det_sample_GP(X, kern, M)
samp3 = X[DPP.list_of_samples[2]]#determinental_sample_GP.det_sample_GP(X, kern, M)
# Sample points from K-dpp determined by second kernel
samples1 = X[DPP2.list_of_samples[0]]#determinental_sample_GP.det_sample_GP(X, kern2, M)
samples2 = X[DPP2.list_of_samples[1]]#determinental_sample_GP.det_sample_GP(X, kern2, M)
samples3 = X[DPP2.list_of_samples[2]]# determinental_sample_GP.det_sample_GP(X, kern2, M)

# sample uniformly
samples_unif1 = np.random.choice(1000, M, replace=False)
samples_unif1 = X[samples_unif1, :]
samples_unif2 = np.random.choice(1000, M, replace=False)
samples_unif2 = X[samples_unif2, :]
samples_unif3 = np.random.choice(1000, M, replace=False)
samples_unif3 = X[samples_unif3, :]



#plotting
plt.figure(figsize=(8, 6))
plt.subplot(331)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samp1[:, 0], samp1[:, 1], c='r', marker="x")
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
plt.ylabel("k-DPP $\ell=2$")

plt.subplot(332)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samp2[:, 0], samp2[:, 1], c='r', marker="x")
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

plt.subplot(333)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samp3[:, 0], samp3[:, 1], c='r', marker="x")
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

plt.subplot(334)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samples1[:, 0], samples1[:, 1], c='orange', marker="x")
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
plt.ylabel("k-DPP $\ell=0.5$")

plt.subplot(335)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samples2[:, 0], samples2[:, 1], c='orange', marker="x")
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

plt.subplot(336)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samples3[:, 0], samples3[:, 1], c='orange', marker="x")
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

plt.subplot(337)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samples_unif1[:, 0], samples_unif1[:, 1], c='y', marker='x')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)
plt.ylabel("Unif. Sample")

plt.subplot(338)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samples_unif2[:, 0], samples_unif2[:, 1], c='y', marker='x')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)

plt.subplot(339)
plt.xlim([-4.5, 4.5])
plt.ylim([-4.5, 4.5])
plt.scatter(X[:, 0], X[:, 1], alpha=0.05, marker=".")
plt.scatter(samples_unif3[:, 0], samples_unif3[:, 1], c='y', marker='x')
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, labelbottom=False, labelleft=False)


plt.subplots_adjust(top=0.975, bottom=0.025, left=0.04, right=0.975,hspace=0.03,wspace=0.03)
#plt.show()
plt.savefig("Figure2.png")