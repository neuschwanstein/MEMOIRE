import warnings

import cvxpy as cvx
import numpy as np

from math_ops import *

class Utility(Function):
    pass

class ExpUtility(Utility):
    def __init__(self,β):
        warnings.warn('r0 is assumed to be = 0', Warning)
        self.β = β
        self.r0 = 0

    def __str__(self):
        return 'u(r) = -exp(-%2.2fr + 1)' % self.β

    def __call__(self,r):
        β,r0  = self.β, self.r0
        exp = np.exp
        pos = lambda x: x*(x >= 0)
        neg = lambda x: -x*(x<0)
        return 1/β * (1-exp(-β*pos(r-r0))) + \
               -neg(r-r0)*exp(-β*r0) + 1/β * (1 - (1+β*r0)*exp(-β*r0))

    def cvx_util(self,r,constraints):
        β,r0 = self.β, self.r0
        exp = cvx.exp
        x = cvx.Variable()
        y = cvx.Variable()

        obj = cvx.Maximize(1/β * (1 - exp(-β*(x+r0))) - y*exp(-β*r0))
        constraints = constraints + [r - r0 == x - y, x >= 0, y >= 0]
        prob = cvx.Problem(obj,constraints)
        prob.solve()
        self._cvx_x,self._cvx_y = x.value,y.value
        return prob.value


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
