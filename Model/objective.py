from cvxpy import *
import numpy as np

class Config:
    Rf = 0.0
    
    mu = 1.0
    beta = 0.8

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


cfg = Config

def append_bias(xss):
    n,_ = xss.shape
    bias = np.ones(n)
    return np.c_[bias,xss]

def solve_objective(X,r,l):
    n,p = X.shape

    cvx_cost = lambda p,r: -cfg.cvx_utility(mul_elemwise(r,p) + (1-p)*cfg.Rf)
    cvx_total_cost = lambda t: 1.0/n * sum_entries(cvx_cost(X*t,r)) + l*norm(t,2)**2

    q = Variable(p)
    objective = Minimize(cvx_total_cost(q))
    problem = Problem(objective)
    problem.solve()

    if problem.status == 'unbounded':
        raise Exception(problem.status)
    if problem.status == 'optimal_inaccurate':
        print(problem.status, " with l=", l)
    
    return q.value.A1, problem.value

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
