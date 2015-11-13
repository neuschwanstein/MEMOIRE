from cvxpy import *
import numpy as np

class config:
    Rf = 0.0
    n,p = 1000,100

    t_mean = 0.0
    t_vol = 3.0
    noise_mean = 0.0
    noise_vol = 1.0
    r_mean = 2.0+Rf
    r_vol = 2.0
    r_max = r_mean + 3*r_vol
    
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
