import numpy as np
import cvxpy as cvx
import model.distrs as d
from model.distrs import E,Var,Std

Z = lambda p: [d.RademacherDistribution()**2 for _ in range(p)]
# p = 50
# n = 500
# m = 100

# bias = d.DiracDistribution(1)
# X = [bias] + [d.RademacherDistribution() for _ in range(p)]
# R = d.DiscreteDistribution([-1,0,2])
# M = X + [R]
# M = d.IndependantMixedDistribution(M)

# for _ in range(m):
#     ms = M.sample(n)
#     xs,rs = ms[:,:-1],ms[:,-1]

#     q = cvx.Variable(p+1)
#     obj = 1/n * cvx.sum_entries(cvx.mul_elemwise(rs,xs*q)) - cvx.norm(q,2)**2
#     prob = cvx.Problem(cvx.Maximize(obj))
#     v = prob.solve()
#     q = q.value.A1

#     ins = np.mean(rs*(xs@q))
#     oos = E(R)*E(np.inner(q,X))
    


print('Done23')
