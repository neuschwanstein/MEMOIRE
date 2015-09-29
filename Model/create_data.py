import numpy as np
from config import *
    
def create_data(n,p):
    t = helper.simplex_sample(p)
    t = np.sqrt(t * r_vol**2)
    t = np.append(r_mean, t)
    
    ones_col = np.ones(n,1)
    S = np.random.randn(n,p)
    S = np.concatenate((ones_col,S), axis=1)

    noise = np.random.randn(n,1)

    r = np.dot(S,t) + noise

    np.save("Data/returns.npy", r)
    np.save("Data/rule.npy", t)
    np.save("Data/dataset.npy", S)

    
if (__name__ == "__main__"):
    reload(cfg)
    create_data(n,p)
