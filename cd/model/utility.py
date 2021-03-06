import cvxpy as cvx
import numpy as np

import cd.model.distrs as ds
# import model.distrs as ds


from .math_ops import Function
from .lipschitzexp import LipschitzExp


class Utility(Function):

    def inverse(self,r):
        raise NotImplementedError

    def subinverse(self,r):
        raise NotImplementedError


class ExpUtility(Utility):
    def __init__(self,β):
        self.β = β
        self.k = 1

    def __str__(self):
        return 'u(r) = -exp(-%2.2fr + 1)' % self.β

    def _call(self,r):
        if isinstance(r,ds.Distribution):
            return self._create_distribution(r)
        β = self.β
        exp = np.exp
        return 1/β * (1 - exp(-β*r))

    def cvx_util(self,r):
        β = self.β
        return 1/β * (1 - cvx.exp(-β*r))

    def _derive(self,r):
        return np.exp(-self.β*r)


class LipschitzExpUtility(Utility):
    def __init__(self,β):
        self.β = β
        self.k = 1

    def __str__(self):
        if isinstance(self.β,int):
            return 'u(r) = -\exp(-%dr + 1)' % self.β
        else:
            return 'u(r) = -\exp(-%2.2fr + 1)' % self.β

    def _call(self,r):
        b = self.β
        return np.where(r<=0,r,b*(1-np.exp(-r/b)))

    def inverse(self,r):
        b = self.β
        return np.where(r<=0,r,-b*np.log(1-r/b))

    def subinverse(self,r):
        return np.where(r<=0,1,np.exp(r/self.β))
        # return np.where(r<=0,1,self.β/(self.β - r))

    def cvx_util(self,r):
        return LipschitzExp(r,self.β)

    @property
    def γ(self):
        return np.exp(-self.β * self.r0)


class LinearUtility(Utility):
    def __init__(self,β):
        self.β = β
        self.k=1
        self.gamma_lipschitz = β

    def __str__(self):
        return 'u(r) = min(r,%2.2fr)' % self.β

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.β * r)
        # return cvx.minimum(r, self.β * r)

    def _call(self,r):
        # TODO Rewrite the method
        return np.amin(np.array([r,self.β*r]),axis=0)

    def _derive(self,r):
        return (r<=0)*1 + (r>0)*self.β

    def inverse(self,r):
        return 1/self.β * r


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

    def subinverse(self,r):
        return 1


class LinearPlateauUtility(Utility):
    '''Piecewise linear utility such that ∂u(x) = 1 for x ∈ (-∞,0), ∂u(x) = β for x ∈ [0,x0]
    and ∂u(x) = 0 for x>x0.
    '''
    def __init__(self,β,x0):
        '''Instantiates the LinearPlateauUtility class.

        Args:
            β: second branch slope
            x0: break-off point
        '''
        if β > 1:
            raise ValueError('β must be lower than 1.')
        if x0 <= 0:
            raise ValueError('x0 must be higher or equal to 0.')
        self.β = β
        self.x0 = x0
        self.k = 1

    def __str__(self):
        s = '$u(r) = \min(r,%.2fr,%.2f)$' % (self.β,self.x0)
        return s

    def cvx_util(self,r):
        return cvx.min_elemwise(r, self.β*r, self.β*self.x0)

    def _call(self,r):
        r = np.array(r)
        return np.minimum(np.minimum(self.β*r,r),self.β*self.x0)

    def inverse(self,r):
        x0,β = self.x0,self.β
        if isinstance(r,list) or isinstance(r,np.ndarray):
            r = np.array(r)
            inv = (r < 0) * r + \
                  (r >= 0) * 1/β*r
            inv[r>self.x0] = np.infty
            return inv
        else:
            if r < 0:
                return r
            elif r <= self.x0:
                return 1/β*r
            else:
                return np.infty

    def _derive(self,r):
        r = np.array(r)
        return (r<=0)*1 + (r>0)*(r<=self.x0)*self.β + (r>self.x0)*0

    @property
    def γ(self):
        return self.β
