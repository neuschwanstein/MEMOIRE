from cvxpy import *
import numpy as np
import config as cfg

n,p = cfg.n,cfg.p

def solve_objective(X,r):
    cvx_utility = lambda r: -exp(-cfg.mu*r)
    cvx_cost = lambda p,r: -cvx_utility(mul_elemwise(r,p) + (1-p)*cfg.Rf)
    cvx_total_cost = lambda t: 1.0/n * sum_entries(cvx_cost(X*t,r)) + cfg.regul_q_norm*norm(t,2)**2

    q = Variable(p)
    objective = Minimize(cvx_total_cost(q))
    problem = Problem(objective)
    problem.solve()

    return q.value, problem.value

def objective(X,r,q):
    utility = lambda r: -np.exp(-cfg.mu*r)
    cost = lambda p,r: -utility(r*p + (1-p)*cfg.Rf)
    total_cost = 1.0/n * sum(cost(np.dot(X,t),r)) + cfg.regul_q_norm*sum(t**2)
    return total_cost


q = Variable(p)

if (__name__ == "__main__"):
    X = np.load("Data/dataset.npy")
    r = np.load("Data/returns.npy")
    t = np.load("Data/rule.npy")

    q,opt = solve_objective(X,r)




# R_f = np.mean([np.mean(r) - 0.5*np.std(r), 0])
