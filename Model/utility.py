import cvxpy as cvx
import numpy as np

class Utility:
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

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.β * r)

    def util(self,r):
        # TODO Rewrite the method
        return np.amin(np.array([r,self.β*r]),axis=0)
