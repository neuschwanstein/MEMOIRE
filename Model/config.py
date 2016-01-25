from cvxpy import *
import numpy as np

class Config:
    Rf = 0.0
    n,p = 1000,100
    
    mu = 1.0
    beta = 0.8
    Lambda = 0.1
    #rBar = 10.0

    # Choose from ['exp','linear']
    utility_shape = 'linear'
        
    @classmethod
    def cvx_utility(cls,r):
        if cls.utility_shape == 'exp':
            return -exp(-cls.mu*r)
        elif cls.utility_shape == 'linear':
            return min_elemwise(r,cls.beta*r)

    @classmethod
    def utility(cls,r):
        if cls.utility_shape == 'exp':
            return -np.exp(-cls.mu*r)
        elif cls.utility_shape == 'linear':
            return np.amin(np.array([r,cls.beta*r]), axis=0)


# Pointer to static config class.
cfg = Config
