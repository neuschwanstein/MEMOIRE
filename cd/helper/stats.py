import numpy as np

def five_stats(data,x=None):
    min = np.amin(data,axis=1)
    q25 = np.percentile(data,25,axis=1)
    median = np.percentile(data,50,axis=1)
    q75 = np.percentile(data,75,axis=1)
    max = np.amax(data,axis=1)
    if x is None:
        return min,q25,median,q75,max
    else:
        return x,min,x,q25,x,median,x,q75,x,max

def empirical_cdf(xs):
    '''Returns empirical cdf for the given dataset.'''
    xs = np.sort(xs)
    return lambda x: np.searchsorted(xs,x,side='right')/len(xs)
