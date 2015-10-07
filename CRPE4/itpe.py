from __future__ import division
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def sample_phi_int(n=1,t=10000):
    phi_k = np.random.random((n, ))
    return np.ceil(t * phi_k)

def itpe(true_k, s=20, t=10000,agg=1):
    likelihoods = np.empty((s, t))
    
    M=np.random.randint(low=1,high=np.ceil(agg*t)+1,size=(s,1))
    theta = 2 * np.pi * np.random.random((s, 1))
    k = np.arange(t)

    true_likelihood_1 = (
        (1 / 2) * (1 - np.cos(
            2 * np.pi * M * true_k / t + theta))
        )
    outcomes = np.random.random(true_likelihood_1.shape) < true_likelihood_1
    likelihoods[:, :] = (
        (1 / 2) * (1 + (-1) ** outcomes * np.cos(
            2 *np.pi * M * k /t -theta
            ))
        )
    k_est=np.argmax(np.prod(likelihoods, axis=0))
    x_true = np.cos(2 * np.pi * true_k / t)
    y_true = np.sin(2 * np.pi * true_k / t)
    x_est = np.cos(2 * np.pi * k_est / t)
    y_est = np.sin(2 * np.pi * k_est / t)
    
    return (k_est, abs(np.arctan2(x_true-x_est,y_true-y_est)), sum(M))

def kitaev_pe_constant(true_phi, j, s):
    phi = true_phi * 2** j
    # We define theta here slightly differently, factoring out
    # the pi / 2, so that we can use a cute indexing trick later.
    # Using astype(bool) lets us take the logical not with the
    # unary negation operator ~.
    theta = np.random.randint(0, 2, size=(s,)).astype(bool)
    L = -(1 / 2) * np.cos(2 * np.pi * phi + (np.pi / 2) * theta) + (1 / 2)
    samples = np.random.random(L.shape) <= L

    p_cos_star = (np.sum(samples[~theta] == 0) - np.sum(samples[~theta] == 1)) / np.sum(~theta)
    p_sin_star = (np.sum(samples[theta] == 1) - np.sum(samples[theta] == 0)) / np.sum(theta)

    rho = np.mod(np.arctan2(p_sin_star, p_cos_star) / (2 * np.pi), 1)
    return rho


def kitaev_pe(true_phi, m, s):
    rho_j = np.array([
        kitaev_pe_constant(true_phi, j, s)
        for j in xrange(m)
    ])

    alpha_bits = np.empty((m + 2,), dtype=bool)

    beta_m = np.argmin(np.abs(np.arange(8) / 8 - rho_j[-1]))
    alpha_bits[-3:] = map(int, np.binary_repr(beta_m, width=3))

    for j in reversed(xrange(m - 1)):
        candidate_1 = 2**-1 + 2**-2 * alpha_bits[j + 1] + 2**-3 * alpha_bits[j + 2]
        alpha_bits[j] = np.mod(np.abs(candidate_1 - rho_j[j]), 1) < 1 / 4

    return np.sum([2**(-j-1) for j in xrange(len(alpha_bits)) if alpha_bits[j]])

def main(smax=20, t=10000, nRuns=1000, is_real=True, agg=1.0):
    
    data=np.zeros(nRuns)
    dataM=np.zeros(nRuns)
    outData=np.zeros((smax,3))
    
    for sind in xrange(0,smax):
        for k in xrange(0,nRuns):
            k_true=np.random.randint(low=1,high=t+1)+(np.random.rand()-0.5)*is_real
            output=itpe(k_true,sind+1,t,agg)
            data[k]=output[1]
            dataM[k]=output[2]   
        
        outData[sind,0]=np.median(data)
        outData[sind,1]=np.percentile(data,25)
        outData[sind,2]=np.mean(dataM)
    
    np.savetxt("ITPE_s="+str(smax)+"_t="+str(t)+"_agg="+str(agg)+".txt",outData)


if __name__ == "__main__":
    import sys
    smax = int(sys.argv[1])
    t = int(sys.argv[2])
    nRuns = int(sys.argv[3])
    is_real = bool(int(sys.argv[4]))
    agg = 0.01 * float(sys.argv[5])

    main(smax, t, nRuns, is_real, agg)
