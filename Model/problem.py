import cvxpy as cvx
import numpy as np
from numpy.linalg import norm

from utility import *

class Problem(object):
    def __init__(self,X,r, λ=1.0, u=LinearUtility(0.8), Rf=0):
        self.X = self.__append_bias(X)
        self.r = r
        self.n, self.p = self.X.shape
        self.λ = λ
        self.u = u
        self.Rf = Rf

        self.__initialize_rbar()
        self.__initialize_Xmax()

    @staticmethod
    def __append_bias(X):
        n,_ = X.shape
        bias = np.ones(n)
        return np.c_[bias,X]

    def __initialize_rbar(self):
        # This method is up to you!
        self.r_bar = np.percentile(np.abs(self.r), 95)

    def __initialize_Xmax(self):
        # And so is this one!
        self.X_max = np.percentile(norm(self.X,axis=0), 95)

    def sigma_admissibility(self):
        k,gamma = self.u.k, self.u.gamma_lipschitz
        return k*gamma*(self.r_bar + self.Rf)

    def alpha_stability(self,n):
        sigma = self.sigma_admissibility()
        return (sigma*self.X_max)**2 / (2*self.λ*n)

    def outsample_bound(self,n,δ):
        k,gamma,X_max,r_bar = self.u.k,self.u.gamma,self.X_max,self.r_bar
        alpha = self.alpha_stability(n)
        p_max = k*gamma*X_max**2*(r_bar-self.Rf) / (2*self.λ)
        return cost(p_max,-r_bar)

    def cost(p,r):
        return -self.u.util(p*r + (1-p)*self.Rf)

    def cvx_cost(p,r):
        return -self.u.cvx_util(cvx.mul_elemwise(p,r) + (1-p)*self.Rf)

    def solve(self):
        n,X,r,λ = self.n,self.X,self.r,self.λ

        def total_cost(q):
            return 1/n * cvx.sum_entries(cost(X*q,r)) + λ*cvx.norm(q)**2

        q = cvx.Variable(self.p)
        objective = cvx.Minimize(total_cost(q))
        problem = cvx.Problem(objective)
        problem.solve()

        if problem.status == 'unbounded':
            raise Exception(problem.status)
        if problem.status == 'optimal_inaccurate':
            # print(problem.status, " with reg =", λ)
            print(problem.status)

        self.q = q.value.A1
        self.insample_cost = problem.value
        return self.insample_cost

    def outsample_risk(self,X,r):
        X = self.__append_bias(X)
        n,_ = X.shape
        p = np.dot(X,self.q)
        total_cost = 1/n * sum(self.cost(p,r))
        return total_cost
