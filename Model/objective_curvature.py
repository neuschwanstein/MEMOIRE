# For a fixed N we draw a sample of empirical optimal decision vectors q, we study the
# influence of the lambda regularization parameter of the algorithm on the `size' of
# covariance ellipsoid (using the largest eigenvalue as a measure of its size).

import pfopt.config as config
import pfopt.create_data as create_data
import pfopt.objective as obj
import numpy as np

cfg = config.config

n,p = 2000,10
cfg.n,cfg.p = n,p

t = create_data.create_rule()
N = 2000

ls = range(1,22,5)
results = np.empty(len(ls))

for i,l in enumerate(ls):
    qs = np.empty((cfg.p+1,N))
    print("lambda=",l)

    for j in range(N):
        print(j)
        X,r = create_data.create_data(t)
        q,_ = obj.solve_objective(X,r,l)
        qs[:,j] = q

    cov_matrix = np.cov(qs)
    eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

    results[i] = np.sort(eigenvals)[-1]
