import numpy as np
import statsmodels.api as sm
import math


def compute_autocovariance(observable, max_time):
    # Find array of differences from mean for the time series observable
    diff_from_mean = observable - np.mean(observable)
    # Return the sum term of equation (18). This is most conveniently done by dotting the appropriate list ranges together.
    return np.array([np.dot(diff_from_mean[:len(observable)-t],diff_from_mean[t:])/(len(observable)-1) for t in range(max_time)])

def estimate_correlation_time(autocov):
    # estimate the auto correlation time
    # firstly, if all covariances are so small they round to zero, correlation 
    # time is zero as system is equilibrated. For nz covariances, asssume they
    # have a well behaved exponential decay behaviour, and estimate correlation 
    # time as the index of the first value below autocov[0]/e:
    below_e = []
    if np.count_nonzero(autocov) == 0:
        correlation_time = 0
    else:
        for i in range(len(autocov)):
            if autocov[i] < np.exp(-1)*autocov[0]:
                below_e.append(i)
        correlation_time = below_e[0] if len(below_e) > 0 else len(autocov)
        
    return correlation_time

def compute_autocorrelation(observable):
    # use statsmodels.api to calculate autocorrelation as it is more optimised:
    autocorr = sm.tsa.acf(observable, nlags = (len(observable) - 1))
    for i in autocorr:
        if math.isnan(i) == True:
            autocorr[autocorr.index(i)] = 0
    return autocorr