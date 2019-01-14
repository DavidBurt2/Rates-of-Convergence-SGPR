import gpflow
import numpy as np
import matplotlib.pyplot as plt
from determinental_sample_GP import det_sample_GP as sample_points
from scipy.cluster.vq import kmeans
import scipy

np.random.seed(1)
lengthscale = .3

D = 1
sn = .1
k = gpflow.kernels.RBF(D)
k.lengthscales = lengthscale
M = 100
a = 1/ 4.
b = 1 / (2 * np.square(lengthscale))
c = np.sqrt(np.square(a)+2*a*b)
B = b/ (a+b+c)
jitter = 1e-14
for N_p in [12]:
    N = int(100 * 2 ** N_p)

    # normally distributed data
    X = np.random.randn(N, D)
    #Kff = k.compute_K_symm(X)
    #Y = np.random.multivariate_normal(mean=np.zeros(N), cov=Kff + sn * np.eye(N))[:, None]
    sampled_Zs, ind = sample_points(X, k, M)
    # plt.scatter(X,np.zeros_like(X))
    # plt.scatter(sampled_Zs,np.zeros_like(sampled_Zs),c='red')
    # plt.show()
    m_error = []
    for m in range(10,M,2):
        Kuu = k.compute_K_symm(sampled_Zs[:m,:])
        Kuf = k.compute_K(sampled_Zs[:m,:], X)
        L= np.linalg.cholesky(Kuu+jitter*np.eye(m))
        LinvKuf = scipy.linalg.solve_triangular(L, Kuf,lower=True)
        kff_diag = k.compute_Kdiag(X)
        qff_diag = np.sum(np.square(LinvKuf), 0)
        V = kff_diag - qff_diag
        m_error.append(np.sum(V))
    plt.plot(np.arange(10,M,2),m_error)
    plt.show()
    #because of jitter, unlikely to go below N*jitter
    plt.semilogy(np.arange(10,M,2),m_error)
    plt.semilogy(np.arange(10, M, 2), (np.arange(10, M, 2)+1)*N*np.sqrt(2*a/(a+b+c))*np.power(B,np.arange(10, M, 2)))
    plt.show()
    # for M in range(10, 100, 10):
    #     #fit full model as a baseline
    #     full_model = gpflow.models.GPR(X, Y, kern=k)
    #     full_model.likelihood.variance = sn
    #     full_model.kern.lengthscales = lengthscale
    #     full_ML = full_model.compute_log_likelihood()
    #
    #     #initialize inducing points
    #
    #     #fit sparse model
    #     Zs = sampled_Zs[0:M, :]
    #     Det_Init_M = gpflow.models.SGPR(X, Y, kern=k, Z=Zs)
    #     Det_Init_M.likelihood.variance = sn
    #     Det_Init_M.kern.lengthscales = lengthscale
    #     Det_Init_M.kern.set_trainable(False)
    #     Det_Init_M.likelihood.set_trainable(False)
    #     Det_Init_M.compile()
    #     #optimize?
    #     opt = gpflow.train.ScipyOptimizer()
    #     opt.minimize(Det_Init_M)
    #     M_ELBO = Det_Init_M.compute_log_likelihood()
    #     Gap = full_ML - M_ELBO
    #     print(Gap)
    #
    #     #fit model with random init
    #     Rand_Init_M = gpflow.models.SGPR(X, Y, kern=k, Z=X[0:M,:])
    #     Rand_Init_M.likelihood.variance = sn
    #     Rand_Init_M.kern.lengthscales = lengthscale
    #     Rand_Init_M.kern.set_trainable(False)
    #     Rand_Init_M.likelihood.set_trainable(False)
    #     Rand_Init_M.compile()
    #     opt = gpflow.train.ScipyOptimizer()
    #     opt.minimize(Rand_Init_M)
    #     M_ELBO = Rand_Init_M.compute_log_likelihood()
    #     Gap = full_ML - M_ELBO
    #     print(Gap)
    #
    #     #fit model with k-means
    #     means_Init_M = gpflow.models.SGPR(X, Y, kern=k, Z= kmeans(X, M, iter=50)[0])
    #     means_Init_M.likelihood.variance = sn
    #     means_Init_M.kern.lengthscales = lengthscale
    #     means_Init_M.kern.set_trainable(False)
    #     means_Init_M.likelihood.set_trainable(False)
    #     means_Init_M.compile()
    #     opt = gpflow.train.ScipyOptimizer()
    #     opt.minimize(means_Init_M)
    #     M_ELBO = means_Init_M.compute_log_likelihood()
    #     Gap = full_ML - M_ELBO
    #     print(Gap)