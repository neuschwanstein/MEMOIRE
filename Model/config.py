from cvxpy import *
import numpy as np

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
utility_shape = 'exp'

if utility_shape == 'exp':
    cvx_utility = lambda r: -exp(-mu*r)
    utility = lambda r: -np.exp(-mu*r)
    
elif utility_shape == 'linear':
    cvx_utility = lambda r: min_elemwise(r,beta*r)
    utility = lambda r: min(r,beta*r)
