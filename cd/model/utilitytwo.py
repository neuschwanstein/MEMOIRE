# -*- coding: utf-8 -*-

import cvxpy as cvx
import numpy as np

# import cd.model.distrs as ds
# import model.distrs as ds


# from .math_ops import Function
from .lipschitzexp import LipschitzExp


class Utility(object):
    pass


class ExpUtility(Utility):
    def __init__(self,beta):
        self.beta = beta
        self.k = 1

    def __str__(self):
        return 'u(r) = -exp(-%2.2fr + 1)' % self.beta

    def _call(self,r):
        if isinstance(r,ds.Distribution):
            return self._create_distribution(r)
        beta = self.beta
        exp = np.exp
        return 1/beta * (1 - exp(-beta*r))

    def cvx_util(self,r):
        beta = self.beta
        exp = cvx.exp
        return 1/beta * (1 - exp(-beta*r))

    def _derive(self,r):
        return np.exp(-self.beta*r)


class LipschitzExpUtility(Utility):
    def __init__(self,beta):
        self.beta = beta
        self.r0 = 1
        self.k = 1

    def __call__(self,r):
        b = self.beta
        return np.where(r<=0,r,1/b*(1-np.exp(-b*r)))

    def cvx_util(self,r):
        return LipschitzExp(r,self.beta,self.r0)

    @property
    def gamma(self):
        return np.exp(-self.beta * self.r0)


class LinearUtility(Utility):
    def __init__(self,beta):
        self.beta = beta
        self.k=1
        self.gamma_lipschitz = beta

    def __str__(self):
        return 'u(r) = min(r,%2.2fr)' % self.beta

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.beta * r)

    def _call(self,r):
        # TODO Rewrite the method
        return np.amin(np.array([r,self.beta*r]),axis=0)

    def _derive(self,r):
        return (r<=0)*1 + (r>0)*self.beta

    def inverse(self,r):
        return 1/self.beta * r


class RiskNeutralUtility(Utility):
    def __init__(self):
        self.k = 1
        self.gamma_lipschitz = 1

    def __str__(self):
        return 'u(r) = r'

    def cvx_util(self,r):
        return r

    def _call(self,r):
        return r

    def _derive(self,r):
        try:
            return np.ones_like(r)
        except:
            return 1

    def inverse(self,r):
        return r


class LinearPlateauUtility(Utility):
    '''Piecewise linear utility such that ∂u(x) = 1 for x ∈ (-∞,0), ∂u(x) = beta for x ∈ [0,x0]
    and ∂u(x) = 0 for x>x0.
    '''
    def __init__(self,beta,x0):
        '''Instantiates the LinearPlateauUtility class.

        Args:
            beta: second branch slope
            x0: break-off point
        '''
        if beta > 1:
            raise ValueError('beta must be lower than 1.')
        if x0 <= 0:
            raise ValueError('x0 must be higher or equal to 0.')
        self.beta = beta
        self.x0 = x0
        self.k = 1

    def __str__(self):
        s = '$u(r) = \min(r,%.2fr,%.2f)$' % (self.beta,self.x0)
        return s

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.beta*r, self.beta*self.x0)

    def _call(self,r):
        r = np.array(r)
        return np.minimum(np.minimum(self.beta*r,r),self.beta*self.x0)

    def inverse(self,r):
        x0,beta = self.x0,self.beta
        if isinstance(r,list) or isinstance(r,np.ndarray):
            r = np.array(r)
            inv = (r < 0) * r + \
                  (r >= 0) * 1/beta*r
            inv[r>self.x0] = np.infty
            return inv
        else:
            if r < 0:
                return r
            elif r <= self.x0:
                return 1/beta*r
            else:
                return np.infty

    def _derive(self,r):
        r = np.array(r)
        return (r<=0)*1 + (r>0)*(r<=self.x0)*self.beta + (r>self.x0)*0

    @property
    def gamma(self):
        return self.beta
