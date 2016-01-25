import numpy as np
import scipy as sp
from copulalib.copulalib import Copula

class Config(object):
    instance = None
    def __new__(cls):
        if cls.instance is None:
            cls.instance = object.__new__(cls)
        return cls.instance

    Rf = 0.0
    n,p = 1000,100


class Copula(object):
    def __init__(self,p=None):
        self.p = p if p else 2


class IndependanceCopula(Copula):
    def __init__(self,p=None):
        super().__init__(p)

    def sample(self,n):
        return np.random.uniform(0,1,(n,self.p))


class ClaytonCopula(Copula):
    def __init__(self,t,p=None):
        if t <= 0:
            raise ValueError('theta must lie in (0,+infty)')
        super().__init__(p)
        self.t = t

    def sample(self,n):
        '''Partly implemented for theta>0. See Stats. Methd. for Fin. Eng. p.305 for more details.

        '''
        t = self.t
        ss = np.random.gamma(1/t,1,n)
        ess = np.random.exponential(1,(n,self.p))
        return [[(1+e/s)**(-1/t) for e in es] for es,s in zip(ess,ss)] # Matlab style?


class UniformDistribution(object):
    def __init__(self,a=None,b=None):
        self.a = a if not a else 0
        self.b = b if not b else 1
        
    def inverse(self,p):
        return self.a + p*(self.b - p)


class NormalDistribution(object):
    def __init__(self, mu=None, vol=None):
        self.mu = mu if  mu else 0
        self.vol = vol if vol else 1

    def inverse(self,p):
        return self.mu + self.vol*np.sqrt(2)*sp.special.erfinv(2*p - 1)


def market_sample(xs,r,cop,n):
    '''For specified marginal distributions for features and market return, and a copula, this
    function returns a sample (matrix) of the features and a sample vector of the returns.

    '''
    distrs = tuple(xs) + (r,)
    cop.p = len(distrs)
    unif_sample = cop.sample(n)
    sample = [[d.inverse(u) for d,u in zip(distrs,us)] for us in unif_sample]
    sample = np.asarray(sample)
    return sample[:,0:-1], sample[:,-1]
    

x1 = NormalDistribution()
x2 = NormalDistribution()
r = NormalDistribution(7,8)

cop = IndependanceCopula()
xss_sample, rs_sample  = market_sample([x1,x2],r,cop,10000)

