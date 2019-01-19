import gpflow
import numpy as np
import matplotlib.pyplot as plt
from determinental_sample_GP import det_sample_GP as sample_points
import scipy

np.random.seed(0)
jitter = 1e-14
low_jitter = gpflow.settings.get_settings()
low_jitter.numerics.jitter_level = jitter

num_trials = 10

D = 1
N_min = 10000
N_max = 150000
N_step = 10000
N_sequence = np.array([int(5000*1.3**pow) for pow in range(10)])#np.arange(N_min,N_max,N_step)#

X = np.random.randn(D,N_sequence[-1])*3
X = np.atleast_1d(X)
X = X.T
sn = .2
epsilon = np.random.randn(N_sequence[-1],1)*sn
Y =  np.sin(X)+epsilon

k = gpflow.kernels.RBF(D)
lengthscale = 1


Min_M = -49
M_sequence = np.linspace(32,50,10)#Min_M + np.ceil(9*np.power(np.log(N_sequence), D)) #times a constant?
M_sequence = [int(M) for M in M_sequence]

print(len(M_sequence),len(N_sequence))
all_gaps = np.zeros((num_trials,len(N_sequence)))
for i in range(num_trials):
    gaps = []
    for N,M in zip(N_sequence, M_sequence):
        X_cur = X[:N, :]
        Y_cur = Y[:N, :]
        Z_cur, ind = sample_points(X_cur, k, M)
        with gpflow.settings.temp_settings(low_jitter):
            Kuu = k.compute_K_symm(Z_cur)
            Kuf = k.compute_K(Z_cur, X_cur)
            L = np.linalg.cholesky(Kuu + jitter * np.eye(len(Z_cur) ))
            LinvKuf = scipy.linalg.solve_triangular(L, Kuf, lower=True)
            t = N-np.sum(np.square(LinvKuf))
            LB = np.linalg.cholesky(Kuu + np.square(sn) ** -1.0 * np.matmul(Kuf, Kuf.T))
            print("t",t,"N",N,"M",M)
            m = gpflow.models.SGPR(X_cur,Y_cur, kern = k, Z = Z_cur)
            m.likelihood.variance = np.square(sn)
            upper = m.compute_upper_bound()
            gap_2 = np.sum(np.square(Y_cur))*t/np.square(sn) +t
            elbo = m.compute_log_likelihood()
            gap =upper - elbo
        print(gap)
        gaps.append(gap)
    all_gaps[i,:] = gaps

mean_gap = np.median(all_gaps,axis=0)
min_gap = np.quantile(all_gaps,0.25 ,axis=0)
max_gap = np.quantile(all_gaps,0.75 ,axis=0)
print(min_gap)
print(max_gap)
plt.figure()

plt.errorbar(N_sequence, mean_gap,  yerr=[mean_gap-min_gap,max_gap-mean_gap])

plt.xlabel("Number of Data Points")
plt.ylabel("Gap Between ELBO and Titsias Upper Bound")
plt.savefig(fname='/scratch/drb62/inducingpoint_plots/continual_learning.eps')
plt.show()

