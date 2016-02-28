import warnings

import cvxpy as cvx
import numpy as np

from math_ops import *
from lipschitzexp import LipschitzExp

class Utility(Function):
    pass

class ExpUtility(Utility):
    def __init__(self,β):
        self.β = β
        self.k = 1

    def __str__(self):
        return 'u(r) = -exp(-%2.2fr + 1)' % self.β

    def __call__(self,r):
        β = self.β
        exp = np.exp
        return 1/β * (1 - exp(-β*r))

    def cvx_util(self,r):
        β = self.β
        exp = cvx.exp
        return 1/β * (1 - exp(-β*r))

    def _derive(self,r):
        return np.exp(-β*r)

class LipschitzExpUtility(Utility):
    def __init__(self,β,r0):
        self.β = β
        self.r0 = r0
        self.k = 1

    def __call__(self,r):
        β,r0 = self.β,self.r0
        exp = np.exp
        return (r >= r0)* (1/β * (1 - exp(-β*r))) + (r<r0) * (r*exp(-β*r0) + 1/β*(1-(1+β*r0)*exp(-β*r0)))

    def cvx_util(self,r):
        return LipschitzExp(r,self.β,self.r0)
        # β,r0 = self.β,self.r0
        # exp = cvx.exp
        # x,y = cvx.Variable(),cvx.Variable()
        # constraints = [r-r0 == x-y, x>=0, y>=0]
        # obj = cvx.Maximize(1/β * (1-exp(-β*(x+r0))) - y*exp(-β*r0)) # + 1/β*(1-(1+β*r0)*exp(-β*r0)))
        # problem = cvx.Problem(obj,constraints)
        # problem.solve(solver=cvx.ECOS)
        # self.x = x.value
        # self.y = y.value
        # return problem


class LinearUtility(Utility):
    def __init__(self,β):
        self.β = β
        self.k=1
        self.gamma_lipschitz = β

    def __str__(self):
        return 'u(r) = min(r,%2.2fr)' % self.β

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.β * r)

    def __call__(self,r):
        # TODO Rewrite the method
        return np.amin(np.array([r,self.β*r]),axis=0)

    def _derive(self,r):
        return (r<=0)*1 + (r>0)*self.β
