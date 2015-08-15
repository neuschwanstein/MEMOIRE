from cvxpy import *
import numpy as np

m,p = 4,4
mu = 1

r = np.random.randn(m)
xs = [100*np.random.randn(p) for _ in xrange(m)]

X = np.zeros((m*p,p))
for i in xrange(m):
    X[p*i:p*(i+1),i] = xs[i]

Q = Variable(m,m*p)
ones = np.ones(m)

objective = Maximize(ones*Q*X*r - mu*norm(Q*X,1))
problem = Problem(objective)

problem.solve()
print "Optimal solution: %f" % problem.value
# print Q.value
# print (norm(Q*X,1))
