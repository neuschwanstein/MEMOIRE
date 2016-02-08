import multiprocessing as mp

import cvxpy as cvx
import numpy as np
from numpy.linalg import norm

from utility import *

class BaseProblem(object):
    def __init__(self,λ,u,Rf=0):
        self.λ = λ
        self.u = u
        self.Rf = Rf

    def cost(self,p,r):
        return -self.u.util(p*r + (1-p)*self.Rf)
        

class AbstractProblem(BaseProblem):
    def __init__(self,market,λ,u,Rf=0):
        self.m = market
        super().__init__(λ,u,Rf)
        
    def sigma_admissibility(self):
        k,gamma = self.u.k, self.u.gamma_lipschitz
        return k*gamma*(self.m.r_bar + self.Rf)

    def alpha_stability(self,n):
        sigma = self.sigma_admissibility()
        return (sigma*self.m.X_max)**2 / (2*self.λ*n)

    def outsample_bound(self,n,δ):
        '''The returned bound holds with probability at least 1-δ 
        over a random draw of size n. See Theorem2.
        '''        
        k,gamma = self.u.k,self.u.gamma_lipschitz
        X_max,r_bar = self.m.X_max,self.m.r_bar
        
        α = self.alpha_stability(n)
        p_max = k*gamma*X_max**2*(r_bar-self.Rf) / (2*self.λ)
        B = self.cost(p_max,-r_bar)
        Ω = 2*α + (4*n*α + B)*np.sqrt(np.log(2/δ)/(2*n))
        return Ω

    def outsample_risk(self,n_sample):
        X,r = self.m.sample(n_sample)
        sample_problem = Problem(X,r,self.λ,self.u,self.Rf)
        insample_cost = sample_problem.solve()
        outsample_cost = sample_problem.outsample_cost(self.X_true,self.r_true)
        return np.abs(insample_cost - outsample_cost)

    def risk_distribution(self,n,n_experiments=1000,n_true=100000):
        self.X_true,self.r_true = self.m.sample(n_true)
        
        ctx = mp.get_context('forkserver')
        with ctx.Pool(ctx.cpu_count()) as pool:
            risk_distribution = pool.map(self.outsample_risk, [n]*n_experiments)

        return risk_distribution


class Problem(BaseProblem):
    def __init__(self,X,r,λ,u,Rf=0):
        self.X = X
        self.r = r
        self.n,self.p = X.shape
        super().__init__(λ,u,Rf)        
        
    def cvx_cost(self,p,r):
        return -self.u.cvx_util(cvx.mul_elemwise(r,p) + (1-p)*self.Rf)

    def solve(self):
        n,X,r,λ = self.n,self.X,self.r,self.λ

        def total_cost(q):
            p = X*q
            return 1/n * cvx.sum_entries(self.cvx_cost(p,r)) + λ*cvx.norm(q)**2

        q = cvx.Variable(self.p)
        objective = cvx.Minimize(total_cost(q))
        problem = cvx.Problem(objective)
        problem.solve(solver=cvx.SCS)

        if problem.status == 'unbounded':
            raise Exception(problem.status)
        if problem.status == 'optimal_inaccurate':
            print(problem.status, " with λ =", λ)

        self.q = q.value.A1
        self.insample_cost = problem.value
        return self.insample_cost

    def outsample_cost(self,X,r):
        n,_ = X.shape
        p = np.dot(X,self.q)
        total_cost = 1/n * sum(self.cost(p,r))
        return total_cost
