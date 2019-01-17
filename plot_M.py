#define model,input distribution,amount of data, bound on norm of y, and M, this returns the corresponding bound on the KL-divergence that holds with probability
#success
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
#kernel params
k_var = 1
k_ls = .1

#likelihood noise
sigma_n = .1

#Number of data points
N = 100000
M = 500
# input density parameters
p_sd = 1

# #Kl-threshold
# epsilon = .1

#success probability
p_success = 0.9

#bound on \|y\|^2
bound_y = 2 * N


def KL_bound(k_var,k_ls,sigma_n,N,p_sd,p_success,bound_y, M):
    # calculations
    a = 1. / (4 * np.square(p_sd))
    b = 1. / (2 * np.square(k_ls))
    c = np.sqrt(np.square(a) + 2 * a * b)
    A = a + b + c
    B = b / A
    delta = 1 - np.sqrt(p_success)
    first_term = (M+1)*np.power(B,M)*N * k_var*np.sqrt(2*a) / (2*np.sqrt(A)*np.square(sigma_n)*np.square(delta)*(1-B))
    second_term = 1 +  bound_y/np.square(sigma_n)
    return first_term*second_term

N= 1e7
KLs_50 = []
KLs_99 = []
Ms = np.arange(200,500,5)
for m in Ms:
    KL_50 = KL_bound(k_var, k_ls, sigma_n, N, p_sd, p_success=.5, bound_y=2, M=m)
    KL_99 = KL_bound(k_var,k_ls,sigma_n,N,p_sd,p_success = .99 ,bound_y=2, M=m)
    KLs_99.append(KL_99)
    KLs_50.append(KL_50)
plt.plot(Ms,KLs_99)
plt.plot(Ms,KLs_50)
plt.ylim([0,100])
plt.legend(["Probability of Sucess=.99","Probability of Sucess=.5"])

plt.xlabel("Number of Inducing Points")
plt.ylabel("Bounds on KL-divergence")
plt.show()

print(KL_bound(k_var,k_ls,sigma_n,N,p_sd,p_success,bound_y, 514))

#NOTE THIS CODE IS SIMPLY ILLUSTRATIVE, i have made the very loose upper bound, M+1 < N. An improved approach is to use a root solver

def func(M,B,c):
    return (M+1) * np.power(B,M) - c


def smallest_m(k_var,k_ls,sigma_n,N,p_sd,p_success,bound_y, KL_threshold):
    a = 1. / (4 * np.square(p_sd))
    b = 1. / (2 * np.square(k_ls))
    c = np.sqrt(np.square(a) + 2 * a * b)
    A = a + b + c
    B = b / A
    delta = 1 - np.sqrt(p_success)
    first_term = N * k_var * np.sqrt(2 * a) / (2 * np.sqrt(A) * np.square(sigma_n) * np.square(delta) * (1 - B))
    second_term = 1 +  bound_y/np.square(sigma_n)
    KL_bound_constant = first_term * second_term
    M_guess = np.log(KL_bound_constant/KL_threshold)/np.log(1/B) -2
    fun = func
    M = scipy.optimize.bisect(fun,a = M_guess,b=M_guess*2,args=(B,KL_threshold/KL_bound_constant))
    return np.ceil(M)

print(smallest_m(k_var,k_ls,sigma_n,N,p_sd,p_success,bound_y, .1))
Ns = np.power(2, np.arange(20,30))
Ms_99 = []
Ms_50 = []
for N in Ns:
    M_50 = smallest_m(k_var, k_ls, sigma_n, N, p_sd, p_success=.5, bound_y=2, KL_threshold=1)
    M_99 = smallest_m(k_var,k_ls,sigma_n,N,p_sd,p_success = .99,bound_y=2, KL_threshold =1)
    Ms_99.append(np.ceil(M_99))
    Ms_50.append(np.ceil(M_50))
plt.plot(Ns,Ms_99)
plt.plot(Ns,Ms_50)
plt.legend(["Probability of Sucess=.99","Probability of Sucess=.5"])
plt.xlabel("Number of Data Points")
plt.ylabel("Number of Inducing Points")
plt.show()