import numpy as np
import helper
from config import *

def create_data(t,n,p):
    ones_col = np.ones((n,1))
    S = np.random.randn(n,p)
    S = np.concatenate((ones_col,S), axis=1)

    noise = np.random.randn(n)

    r = np.dot(S,t) + noise

    np.save("Data/returns.npy", r)
    np.save("Data/dataset.npy", S)

    
def create_rule(p):
    t = helper.simplex_sample(p)
    t = np.sqrt(t * r_vol**2)
    t = np.append(r_mean, t)
    
    return t

    
if (__name__ == "__main__"):
    reload(cfg)
    create_data(n,p)
