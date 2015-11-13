import numpy as np
import pfopt.config
import pfopt.helper as helper

cfg = pfopt.config.config

def create_data(t, save=False):
    n,p = cfg.n,cfg.p
    
    ones_col = np.ones((n,1))
    X = np.random.randn(n,p)
    X = np.concatenate((ones_col,X), axis=1)
    noise = np.random.randn(n)

    r = np.dot(X,t) + noise

    if save:
        np.save("Data/returns.npy", r)
        np.save("Data/dataset.npy", S)
    else:
        return X,r

    
def create_rule():
    t = helper.simplex_sample(cfg.p)
    t = np.sqrt(t * cfg.r_vol**2)
    t = np.append(cfg.r_mean, t)
    
    return t

    
# if (__name__ == "__main__"):
#     reload(cfg)
#     create_data(n,p)
