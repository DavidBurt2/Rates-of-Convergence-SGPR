import numpy as np
import gpflow
import scipy
jitter = 1e-12
no_jitter = gpflow.settings.get_settings()
no_jitter.numerics.jitter_level = 0

def det_sample_GP(X,kern,M):
    N = X.shape[0]
    Zs = list()
    Zs.append(np.random.randint(N))
    for m in range(M-1):
    #    Y = np.zeros((len(Zs), 1))
        Kuu = kern.compute_K_symm(X[Zs,:])
       # mask = np.ones(X.shape[0], dtype=bool)
        #mask[Zs] = False
        Kuf = kern.compute_K(X[Zs,:], X)
        L = np.linalg.cholesky(Kuu+jitter*np.eye(m+1))
        LinvKuf = scipy.linalg.solve_triangular(L, Kuf,lower=True)
        V = kern.compute_Kdiag(X)-np.sum(np.square(LinvKuf), 0)
        # with gpflow.settings.temp_settings(no_jitter):
        #      gpr = gpflow.models.GPR(X[Zs, :], Y, kern)
        #      gpr.likelihood.variance = 0
        #      V = gpr.predict_f(X)[1]
        CMF = np.cumsum(V)
        CMF = CMF/CMF[-1]
        U = np.random.rand()
        i = 0
        while U > CMF[i]:
            i += 1
        Zs.append(i)
        if m==M-1:
            Kuf = kern.compute_K(X[Zs, :], X)
            L, low = scipy.linalg.cho_factor(Kuu + jitter * np.eye(m + 1))
            LinvKuf = scipy.linalg.cho_solve((L, low), Kuf)
            V = kern.compute_Kdiag(X) - np.sum(np.square(LinvKuf), 0)

    return X[Zs,:],Zs
