import cvxpy as cvx
import numpy as np

from math_ops import *

class Utility(Function):
    pass

class ExpUtility(Utility):
    def __init__(self,μ):
        self.μ = μ

    def cvx_util(self,r):
        return -cvx.exp(-self.μ*r + 1)

    def util(self,r):
        return -np.exp(-self.μ*r + 1)


class LinearUtility(Utility):
    def __init__(self,β):
        self.β = β
        self.k=1
        self.gamma_lipschitz = β

    def __repr__(self):
        return 'u(r) = min(r,%2.2fr)' % self.β

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.β * r)

    def __call__(self,r):
        # TODO Rewrite the method
        return np.amin(np.array([r,self.β*r]),axis=0)

    def _derive(self,r):
        return (r<=0)*1 + (r>0)*self.β
