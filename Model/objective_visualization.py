# For a fixed N we draw a sample of empirical optimal decision vectors q, we study the
# influence of the lambda regularization parameter of the algorithm on the `size' of
# covariance ellipsoid (using the largest eigenvalue as a measure of its size).

import config
import create_data
import objective as obj
import numpy as np
import matplotlib.pyplot as plt

cfg = config.config

n,p = 100,1
cfg.n,cfg.p = n,p
l = 1

t = create_data.create_rule()
X,r = create_data.create_data(t)

(q1,q2),_ = obj.solve_objective(X,r,l)
q1s = np.arange(-5,5,0.1) + q1
q2s = np.arange(-5,5,0.1) + q2
ys = [obj.objective(X,r,[q1,q2],l) for q2 in q2s]

plt.plot(q2s,ys)
plt.show()

# ls = range(1,22,5)
# results = np.empty(len(ls))

# for i,l in enumerate(ls):
#     qs = np.empty((cfg.p+1,N))
#     print("lambda=",l)

#     for j in range(N):
#         print(j)
#         X,r = create_data.create_data(t)
#         q,_ = obj.solve_objective(X,r,l)
#         qs[:,j] = q

#     cov_matrix = np.cov(qs)
#     eigenvals, eigenvecs = np.linalg.eig(cov_matrix)

#     results[i] = np.sort(eigenvals)[-1]
