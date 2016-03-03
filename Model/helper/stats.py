import numpy as np

def five_stats(data,x=None):
    min = np.amin(data,axis=0)
    q25 = np.percentile(data,25,axis=0)
    median = np.percentile(data,50,axis=0)
    q75 = np.percentile(data,75,axis=0)
    max = np.amax(data,axis=0)
    if x is None:
        return min,q25,median,q75,max
    else:
        return x,min,x,q25,x,median,x,q75,x,max
