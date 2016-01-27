# from cvxpy import *
import cvxpy as cvx
import numpy as np

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
    u = ExpUtility(0.8)

cfg = Config


def append_bias(xss):
    n,_ = xss.shape
    bias = np.ones(n)
    return np.c_[bias,xss]

def solve_objective(X,r,l):
    n,p = X.shape

    cost = lambda p,r: -cfg.u.cvx_utility(cvx.mul_elemwise(r,p) + (1-p)*cfg.Rf)
    total_cost = lambda t: 1.0/n * cvx.sum_entries(cvx_cost(X*t,r)) + l*cvx.norm(t,2)**2

    q = Variable(p)
    objective = Minimize(total_cost(q))
    problem = Problem(objective)
    problem.solve()

    if problem.status == 'unbounded':
        raise Exception(problem.status)
    if problem.status == 'optimal_inaccurate':
        print(problem.status, " with l=", l)
    
    return q.value.A1, problem.value

def solve_unregularized_objective(X,r):
    return solve_objective(X,r,l=0)

def regularized_risk(X,r,q,l):
    n,_ = X.shape
    cost = lambda p,r: -cfg.utility(r*p + (1-p)*cfg.Rf)
    total_cost = 1/n * sum(cost(np.dot(X,q),r)) + l*sum(q**2)
    return total_cost

def risk(X,r,q):
    n,_ = X.shape
    cost = lambda p,r: -cfg.utility(r*p + (1-p)*cfg.Rf)
    total_cost = 1/n * sum(cost(np.dot(X,q),r))
    return total_cost
