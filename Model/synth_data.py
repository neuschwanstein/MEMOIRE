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
        '''Partly implemented for theta>0. See Stats. Methd. for Fin. Eng. p.305 for more details.

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


class UniformDistribution(object):
    def __init__(self,a=None,b=None):
        self.a = a if not a else 0
        self.b = b if not b else 1
        
    def inverse(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')
        return self.a + p*(self.b - p)


class NormalDistribution(object):
    def __init__(self, μ=0, σ=1):
        self.μ = μ
        self.σ = σ

    def __repr__(self):
        return 'N(μ=%2.2f,σ=%2.2f)' % (self.μ, self.σ)

    def inverse(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')
        return self.μ + self.σ*np.sqrt(2)*sp.special.erfinv(2*p - 1)


def market_sample(xs,r,cop,n):
    '''For specified marginal distributions for features and market return, and a copula, this
    function returns a sample (matrix) of the features and a sample vector of the returns.

    '''
    distrs = tuple(xs) + (r,)
    cop.p = len(distrs)
    unif_sample = cop.sample(n)
    sample = np.array([d.inverse(us) for d,us in zip(distrs,unif_sample.T)]).T
    return sample[:,0:-1], sample[:,-1]
    
if (__name__ == '__main__'):
    p = 100
    n_true = 100000
    x_distrs = [NormalDistribution() for _ in range(p)]
    r_distr = NormalDistribution(8,10)
    cop = ClaytonCopula(10) # TODO Investigate meaning of the argument.

    xss,rs = market_sample(x_distrs,r_distr,cop,n_true)

    print('Done.')
