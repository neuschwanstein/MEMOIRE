import multiprocessing as mp

import cvxpy as cvx
import numpy as np
from numpy.linalg import norm
import scipy.optimize

from utility import *

class BaseProblem(object):
    def __init__(self,u,Rf=0):
        self.u = u
        self.Rf = Rf

    def cost(self,p,r):
        return -self.u.util(p*r + (1-p)*self.Rf)
        

class AbstractProblem(BaseProblem):
    def __init__(self,market,n=None,λ=None,u=LinearUtility(β=0.8),Rf=0):
        self.m = market
        self._n = n
        self._λ = λ
        super().__init__(u,Rf)

    @property
    def n(self):
        if not self._n:
            raise ValueError('n must be initialized')
        return self._n

    @n.setter
    def n(self,n):
        self._n = n

    @property
    def λ(self):
        if not self._λ:
            raise ValueError('λ must be initialized')
        return self._λ

    @λ.setter
    def λ(self,λ):
        self._λ = λ    

    @property
    def p(self):
        return self.m.p

    @property
    def X_max(self):
        return self.m.X_max

    @property
    def r_bar(self):
        return self.m.r_bar

    @property
    def k(self):
        return self.u.k

    @property
    def γ(self):
        return self.u.gamma_lipschitz

    @property
    def σ(self):
        σ = self.k * self.γ * (self.r_bar + self.Rf)
        return σ

    @property
    def α(self):
        α = (self.σ * self.X_max)**2 / (2*self.λ*self.n)
        return α

    @property
    def p_max(self):
        p_max = self.k * self.γ * self.X_max**2 * (self.r_bar - self.Rf) / (2*self.λ)
        return p_max

    @property
    def B(self):
        B = self.cost(self.p_max, -self.r_bar)
        return B

    def outsample_bound(self,δ):
        '''The returned bound holds with probability at least 1-δ 
        over a random draw of size n. See Theorem2.
        '''        
        α = self.α
        n = self.n
        B = self.B
        Ω = 2*α + (4*n*α + B)*np.sqrt(np.log(2/δ)/(2*n))
        return Ω

    def outsample_risk(self,_):
        X,r = self.m.sample(self.n)
        sample_problem = Problem(X,r,self.λ,self.u,self.Rf)
        insample_cost = sample_problem.solve()
        outsample_cost = sample_problem.outsample_cost(self.X_true,self.r_true)
        return np.abs(insample_cost - outsample_cost)

    def outsample_risk_distribution(self,n_experiments=1000,n_true=100000):
        self.X_true,self.r_true = self.m.sample(n_true)
        
        ctx = mp.get_context('forkserver')
        with ctx.Pool(ctx.cpu_count()) as pool:
            risk_distribution = pool.map(self.outsample_risk, [self.n]*n_experiments)

        return risk_distribution

    def find_optimal_λ(self,n,λ0 = 1.0):
        def objective(λ):
            self.λ = λ
            return np.mean(self.risk_distribution(n))
        result = scipy.optimize.minimize(objective,λ0)
        return result


class Problem(BaseProblem):
    def __init__(self,X,r,λ,u,Rf=0):
        self.X = X
        self.r = r
        self.λ = λ
        self.n,self.p = X.shape
        super().__init__(u,Rf)        
        
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
        raise ValueError("no! Must take the empirical lost without λ regularizer!")
        # return self.insample_cost

    def outsample_cost(self,X,r):
        n,_ = X.shape
        p = np.dot(X,self.q)
        total_cost = 1/n * sum(self.cost(p,r))
        return total_cost
