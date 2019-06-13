import numpy as np
import scipy
jitter = 1e-12


# We implement this algorithm the naive way, without the rank one updates needed to make this algorithm O(NM^2)
def det_sample_GP(X, kern, M):
    """

    :param X: Dataset [N,D]
    :param kern: GPflow kernel
    :param M: Number of inducing points to select
    :return: A set of M inducing points sampled according to Algorithm 1, as well as the associated indices
    """
    N = X.shape[0]
    Zs = list()
    Zs.append(np.random.randint(N))  # the first point is randomly selected
    for m in range(M-1): # with this naive implementation the complexity of the inner loop is O(NM^2+M^3)
        Kuu = kern.compute_K_symm(X[Zs, :])
        Kuf = kern.compute_K(X[Zs, :], X)
        # LL^T = Kuu
        L = np.linalg.cholesky(Kuu+jitter * np.eye(m+1)) # jitter for numerical stability, this is O(M^3)

        LinvKuf = scipy.linalg.solve_triangular(L, Kuf, lower=True) # This is O(NM^2)
        # posterior varaince of GP
        V = kern.compute_Kdiag(X) - np.sum(np.square(LinvKuf), 0)
        CMF = np.cumsum(V)
        CMF = CMF/CMF[-1]
        U = np.random.rand()
        i = 0
        while U > CMF[i]:
            i += 1
        Zs.append(i)

    return X[Zs, :], Zs
