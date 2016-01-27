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

    def solve(self):
        def cost(p,r):
            return -self.u.cvx_util(cvx.mul_elemwise(r,p) + (1-p)*self.Rf)

        def total_cost(q):
            n,X,r,λ = self.n,self.X,self.r,self.λ
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
        cost = lambda p,r: -self.u.util(r*p + (1-p)*self.Rf)
        total_cost = 1/n * sum(cost(np.dot(X,self.q),r))
        return total_cost
