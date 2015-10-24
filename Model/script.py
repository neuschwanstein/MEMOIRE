import config
import create_data
import objective as obj
import numpy as np

cfg = config.config

cfg.n,cfg.p = 2000,100

t = create_data.create_rule()
# create_data.create_data(t)

# X = np.load("Data/dataset.npy")
# r = np.load("Data/returns.npy")

# q,_ = obj.solve_objective(X,r,0.01)

N = 1
results = np.zeros(N);
for i in range(N):
    X,r = create_data.create_data(t)
    q,_ = obj.solve_objective(X,r,1)
    # results[i] = q
    

# lambda0 = 0.01
# nLambda = 10
# endLambda = 0.01

# n = 10000
# for l in np.linspace(lambda0,endLambda,nLambda):
#     results = np.empty(N)   
#     for i in xrange(N):
#         q,_ = obj.solve_objective(X,r,l)
#         results[i] = q
