import numpy as np
import matplotlib.pyplot as plt
#Problem 1

def mc_estimate(mu0,sigma0, n):
    '''
    SMC method for predicting E[sqrt(x^2+y^2)] 
    '''
    samples = np.random.normal(loc=mu0, scale=sigma0, size=[n,2])
    pi_values = -1/(2*sigma**2) * np.square((samples-mu))
    g_values = -1/(2*sigma0**2) * np.square((samples-mu0))
    pi_values = np.exp(pi_values)/sigma
    g_values = np.exp(g_values)/sigma0
    pi_values = np.product(pi_values,axis=1)
    g_values = np.product(g_values,axis=1)
    f_samples = np.sqrt(np.sum(np.square(samples),axis=1))
    weights = pi_values / g_values
    
    return f_samples,weights
    
n_samples = 50000000
#problem 1
mu = 2
sigma = 1
pi_function = np.vectorize(lambda x : norm.pdf(x,mu,sigma))
#theta1
mu0 = 2
sigma0 = 1
print("1")
f_samples1,weights1 = mc_estimate(mu0, sigma0, n_samples)
#np.save("problem1/samples1.npy",f_samples1)
#np.save("problem1/weights1.npy",weights1)
weighted_fsamples1 = f_samples1*weights1
#theta2
mu0 = 0
sigma0 = 1
print("2")
f_samples2,weights2 = mc_estimate(mu0, sigma0, n_samples)
weighted_fsamples2 = f_samples2*weights2
#theta3
mu0 = 0
sigma0 = 4
print('3')
f_samples3,weights3 = mc_estimate(mu0, sigma0, n_samples)
weighted_fsamples3 = f_samples3*weights3

# n_points = 100
# l_n_samples = [i*n_samples//n_points for i in range(1,n_points)]
# if l_n_samples[0] > 100:
    # l_n_samples = [10,100,1000] + l_n_samples
l_n_samples = [10,100,1000,10000,100000,1000000,\
        10000000,25000000,50000000]
theta1 = [np.mean(weighted_fsamples1[:n]) for n in l_n_samples]
ess1 = [n/(1+np.var(weights1[:n])) for n in l_n_samples]
theta2 = [np.mean(weighted_fsamples2[:n]) for n in l_n_samples]
ess2 = [n/(1+np.var(weights2[:n])) for n in l_n_samples]
theta3 = [np.mean(weighted_fsamples3[:n]) for n in l_n_samples]
ess3 = [n/(1+np.var(weights3[:n])) for n in l_n_samples]

l_log_n_samples = np.log10(l_n_samples)
plt.plot(l_log_n_samples, theta1,label="theta1")
plt.plot(l_log_n_samples, theta2,label="theta2")
plt.plot(l_log_n_samples, theta3,label="theta3")
plt.xlabel("number samples (log10n)")
plt.ylabel("theta_hat")
plt.legend()
plt.show()


#10^3
#10^7
#looking at graph, we see 
plt.plot(l_log_n_samples,ess2/ess2[2])
plt.plot(l_log_n_samples,ess3/ess3[5])
plt.xlabel("number samples (log10n)")
plt.ylabel("ess(n)/ess*(n)")
plt.show()
plt.legend()
