import cvxpy as cvx
import numpy as np

class Utility:
    pass


class ExpUtility(Utility):
    def __init__(self,mu):
        self.mu = mu

    def cvx_util(self,r):
        return -cvx.exp(-self.mu * r)

    def util(self,r):
        return -np.exp(-self.mu * r)


class LinearUtility(Utility):
    def __init__(self,beta):
        self.beta = beta

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.beta * r)

    def util(self,r):
        # TODO Rewrite the method
        return np.amin(np.array([r,self.beta*r]),axis=0)
