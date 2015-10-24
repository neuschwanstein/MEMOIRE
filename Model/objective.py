from cvxpy import *
import numpy as np
import config
import helper

cfg = config.config

def normalize_data(data):
    mu = data.mean(axis=0)      # axis=0: along column
    vol = data.std(axis=0)
    return (data-mu)/vol

def solve_objective(X,r,l):
    n,p = cfg.n,cfg.p+1

    cvx_cost = lambda p,r: -cfg.cvx_utility(mul_elemwise(r,p) + (1-p)*cfg.Rf)
    cvx_total_cost = lambda t: 1.0/n * (sum_entries(cvx_cost(X*t,r)) + l*norm(t,2)**2)

    q = Variable(p)
    objective = Minimize(cvx_total_cost(q))
    problem = Problem(objective)
    problem.solve()

    return q.value.A1, problem.value


def objective(X,r,q,l):
    # utility = lambda r: -np.exp(-cfg.mu*r)
    cost = lambda p,r: -cfg.utility(r*p + (1-p)*cfg.Rf)
    total_cost = 1.0/n * sum(cost(np.dot(X,t),r)) + l*sum(t**2)
    return total_cost

# if (__name__ == "__main__"):
#     X = np.load("Data/dataset.npy")
#     r = np.load("Data/returns.npy")
#     t = np.load("Data/rule.npy")

#     q,opt = solve_objective(X,r)
