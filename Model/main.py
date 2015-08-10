from cvxpy import *
import numpy as np

Rf = np.sqrt(np.e) # sqrt(e) is the mean of the lognormal distribution used to generate rs.

def utility(r):
    return -np.exp(-r)
U = utility

def objective(ps,rs):
    return -U(np.dot(rs,ps) + (1 - np.sum(ps))*Rf)
c = objective

# Let us first consider a single holding period.
# Once it's complete we'll go on and consider many 

p = 10  # Number of features
m = 100 # Number of assets

#xs = [np.random.randn(p) for _ in xrange(m)] # Features of each asset
X = np.random.randn(m,p)
rs = np.random.randn(m)              # Returns for each asset

# If convex optimization is impossible, then set Qs with random values
Qs = [np.random.randn(p,m) for _ in xrange(m)]

ps = [np.trace(np.dot(X,Q)) for Q in Qs]

cost = c(ps,rs)
print cost
    


# m = 30
# n = 20
# np.random.seed(1)
# A = np.random.randn(m,n)
# b = np.random.randn(m)

# # Construct the problem.
# x = Variable(n)
# objective = Minimize(sum_squares(A*x - b))
# constraints = [0 <= x, x <= 1]
# prob = Problem(objective, constraints)

# # The optimal objective is returned by prob.solve().
# result = prob.solve()
# # The optimal value for x is stored in x.value.
# print x.value
# # The optimal Lagrange multiplier for a constraint
# # is stored in constraint.dual_value.
# print constraints[0].dual_value
