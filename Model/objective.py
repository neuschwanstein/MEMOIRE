# from cvxpy import *
import cvxpy as cvx
import numpy as np
from numpy.linalg import norm

class Utility:
    pass


class ExpUtility(Utility):
    def __init__(self,mu):
        self.mu = mu

    def cvx_utility(self,r):
        return -cvx.exp(-self.mu * r)

    def utility(self,r):
        return -np.exp(-self.mu * r)


class LinearUtility(Utility):
    def __init__(self,beta):
        self.beta = beta

    def cvx_utility(self,r):
        return cvx.min_elemwise(r, self.beta * r)

    def utility(self,r):
        # TODO Rewrite the method
        return np.amin(np.array([r,self.beta*r]),axis=0)


class Config:
    # This class is meant to be accessed and changed by the `client.'
    Rf = 0.0
    # r_bar = Solve the problem here. Perhaps with a new Problem class or smthing like that.
    u = LinearUtility(0.8)
    λ = 1.0

cfg = Config


def append_bias(xss):
    n,_ = xss.shape
    bias = np.ones(n)
    return np.c_[bias,xss]

def solve_objective(X,r):
    n,p = X.shape

    cost = lambda p,r: -cfg.u.cvx_utility(cvx.mul_elemwise(r,p) + (1-p)*cfg.Rf)
    total_cost = lambda t: \
                 1/n*cvx.sum_entries(cost(X*t,r)) + cfg.λ * cvx.norm(t)**2

    q = Variable(p)
    objective = Minimize(total_cost(q))
    problem = Problem(objective)
    problem.solve()

    if problem.status == 'unbounded':
        raise Exception(problem.status)
    if problem.status == 'optimal_inaccurate':
        print(problem.status, " with l=", l)
    
    return q.value.A1, problem.value

def regularized_risk(X,r,q):
    n,_ = X.shape
    cost = lambda p,r: -cfg.u.utility(r*p + (1-p)*cfg.Rf)
    total_cost = 1/n * sum(cost(np.dot(X,q),r)) + cfg.λ * norm(q)**2
    return total_cost
