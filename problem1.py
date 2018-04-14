from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
#Problem 1
mu = 2
sigma = 1
pi_function = np.vectorize(lambda x : norm.pdf(x,mu,sigma))

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
    theta_hat = np.mean(f_samples * weights)
    ess = n/(1+np.var(weights))
    return theta_hat,ess

n_samples = [10**i for i in range(1,7)]
n_samples.append([5e6,1e7])
#theta1
mu0 = 2
sigma0 = 1
theta1 = []
ess1 = []
print("theta1")
for n in n_samples2:
    theta_hat,ess = mc_estimate(mu0, sigma0, n)
    print("n:{} theta:{}".format(n,theta_hat))
    theta1.append(theta_hat)
    ess1.append(ess)
#theta2
mu0 = 0
sigma0 = 1
theta2 = []
ess2 = []
print("theta2")
for n in n_samples:
    theta_hat,ess = mc_estimate(mu0, sigma0, n)
    print("n:{} theta:{}".format(n,theta_hat))
    theta2.append(theta_hat)
    ess2.append(ess)
#theta3
mu0 = 0
sigma0 = 4
theta3 = []
ess3 = []
print("theta3")
for n in n_samples:
    theta_hat,ess = mc_estimate(mu0, sigma0, n)
    print("n:{} theta:{}".format(n,theta_hat))
    theta3.append(theta_hat)
    ess3.append(ess)

log_n = np.log10(n_samples)
plt.plot(log_n, theta1,label="theta1")
plt.plot(log_n, theta2,label="theta2")
plt.plot(log_n, theta3,label="theta3")
plt.xlabel("number samples (log10n)")
plt.ylabel("theta_hat")
plt.legend()
plt.show()



#looking at graph, we see 
plt.plot(log_n,ess2/ess2[5],label="ess theta2")
plt.plot(log_n,ess3/ess3[2],label="ess theta3")

plt.xlabel("number samples (log10n)")
plt.ylabel("ess(n)/ess*(n)")
plt.legend()
