from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
#Problem 1

def mc_estimate(mu0,sigma0, n):
    '''
    SMC method for predicting E[sqrt(x^2+y^2)] 
    '''
    g_function = np.vectorize(lambda x : norm.pdf(x,mu0,sigma0))
    
    samples = np.random.normal(loc=mu0, scale=sigma0, size=[n,2])
    pi_values = pi_function(samples)
    g_values = g_function(samples)
    f_samples = np.sqrt(np.sum(np.square(samples),axis=1))
    pi_values = np.product(pi_values,axis=1)
    g_values = np.product(g_values,axis=1)
    weights = pi_values / g_values
    
    return f_samples,weights
    

#exp set up
n_samples = 10**7
n_points = 100
#problem 1
mu = 2
sigma = 1
pi_function = np.vectorize(lambda x : norm.pdf(x,mu,sigma))
l_n_samples = [i*n_samples//n_points for i in range(1,n_samples//n_points)]
#theta1
mu0 = 2
sigma0 = 1
f_samples,weights = mc_estimate(mu0, sigma0, n_samples)
weighted_fsamples = f_samples*weights
theta1 = [np.mean(weighted_fsamples[:n]) for n in l_n_samples]
ess1 = [n/(1+np.var(weights[:n])) for n in l_n_samples]


#theta2
mu0 = 0
sigma0 = 1
f_samples,weights = mc_estimate(mu0, sigma0, n_samples)
weighted_fsamples = f_samples*weights
theta2 = [np.mean(weighted_fsamples[:n]) for n in l_n_samples]
ess2 = [n/(1+np.var(weights[:n])) for n in l_n_samples]

#theta3
mu0 = 0
sigma0 = 4
f_samples,weights = mc_estimate(mu0, sigma0, n_samples)
weighted_fsamples = f_samples*weights
theta3 = [np.mean(weighted_fsamples[:n]) for n in l_n_samples]
ess3 = [n/(1+np.var(weights[:n])) for n in l_n_samples]




l_log_n_samples = np.log10(l_n_samples)
plt.plot(l_log_n_samples, theta1,label="theta1")
plt.plot(l_log_n_samples, theta2,label="theta2")
plt.plot(l_log_n_samples, theta3,label="theta3")
plt.xlabel("number samples (log10n)")
plt.ylabel("theta_hat")
plt.legend()
plt.show()



#looking at graph, we see 

plt.xlabel("number samples (log10n)")
plt.ylabel("ess(n)/ess*(n)")
plt.legend()
