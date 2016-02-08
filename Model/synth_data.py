import multiprocessing

import numpy as np
import scipy as sp
import scipy.special
from scipy.stats import norm

class Copula(object):
    def __init__(self,p=None):
        self.p = p if p else 2

    def sample(self,n):
        raise NotImplementedError


class IndependanceCopula(Copula):
    def __init__(self,p=None):
        super().__init__(p)

    def sample(self,n):
        return np.random.uniform(0,1,(n,self.p))


class ClaytonCopula(Copula):
    def __init__(self,t,p=None):
        if t <= 0:
            raise ValueError('θ must lie in (0, +∞)')
        super().__init__(p)
        self.t = t

    def sample(self,n):
        '''Partly implemented for θ>0. See Stats. Methd. for Fin. Eng. p.305 for more details.

        '''
        t = self.t
        s = np.random.gamma(1/t,1,(n,1))
        e = np.random.exponential(1,(n,self.p))
        return (1 + e/s)**(-1/t)


class GaussianCopula(Copula):
    def __init__(self,Σ):
        Σ = np.array(Σ)
        self.p,_ = Σ.shape
        self.Σ = Σ

    def sample(self,n):
        μ = np.zeros(self.p)
        z = np.random.multivariate_normal(μ,self.Σ,n)
        return norm.cdf(z)


class Distribution(object):
    def inverse(self,p):
        raise NotImplemented


class UniformDistribution(Distribution):
    def __init__(self,a=None,b=None):
        self.a = a if not a else 0
        self.b = b if not b else 1
        
    def inverse(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')
        return self.a + p*(self.b - p)


class NormalDistribution(Distribution):
    def __init__(self, μ=0, σ=1):
        self.μ = μ
        self.σ = σ

    def __repr__(self):
        return 'N(μ=%2.2f,σ=%2.2f)' % (self.μ, self.σ)

    def inverse(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')
        return self.μ + self.σ*np.sqrt(2)*sp.special.erfinv(2*p - 1)


class Market(object):
    def __init__(self,r_distr,x_distrs,cop):
        self.r_distr = r_distr
        self.x_distrs = x_distrs
        self.p = len(x_distrs) + 1 # +1: bias feature
        self.cop = cop

        highest_quantile = \
            lambda d: np.max(np.abs([d.inverse(0.02),d.inverse(.98)]))
        self.r_bar = highest_quantile(r_distr)
        self._X_max = None
        self.X_max = np.linalg.norm([highest_quantile(x_d) for x_d in x_distrs])

    @property
    def X_max(self):
        '''Or implement theoretical value?'''
        if self._X_max:
            return self._X_max
        X,_ = self.sample(100000)
        self._X_max = np.percentile(np.linalg.norm(self.X,axis=1),98)
        return self._X_max

    @X_max.setter
    def X_max(self,val):
        self._X_max = val

    def sample(self,n):
        distrs = tuple(self.x_distrs) + (self.r_distr,)
        unif_sample = self.cop.sample(n)
        sample = np.array([d.inverse(us) for d,us in zip(distrs,unif_sample.T)]).T
        X = sample[:,0:-1]
        r = sample[:,-1]
        bias = np.ones(n)
        X = np.c_[bias,X]
        return X,r


class GaussianMarket(Market):
    def __init__(self,r_distr,x_distrs):
        p = len(x_distrs)
        ε = 0.001
        α = (1-ε)/p                     # Information from every feature
        v = [α]*p + [1]
        Σ = np.empty((p+1,p+1))
        Σ[:-1,:-1] = np.eye(p)
        Σ[-1,:] = v
        Σ[:,-1] = v
        cop = GaussianCopula(Σ)
        super().__init__(r_distr,x_distrs,cop)
