import multiprocessing

import numpy as np
import scipy as sp
import scipy.special
from scipy.stats import norm

from math_ops import *

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


class UniformDistribution(Distribution):
    def __init__(self,a=0,b=1):
        if b <= a:
            raise ValueError('Must have b > a')
        self.a = a
        self.b = b
        
    def inverse(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')
        return self.a + p*(self.b - p)

    def E(self):
        return 0.5*(self.a + self.b)

    def var(self):
        return 1/12 * (self.b - self.a)**2


class NormalDistribution(Distribution):
    def __init__(self, μ=0, σ=1):
        self.μ = μ
        self.σ = σ

    def __str__(self):
        return 'N(μ=%2.2f,σ=%2.2f)' % (self.μ, self.σ)

    def E(self):
        return self.μ

    def __add__(self,μ0):
        return NormalDistribution(self.μ+μ0,self.σ)

    def __sub__(self,μ0):
        return self.__add__(-μ0)

    def __sub__(self,a):
        return NormalDistribution(self.μ,a*self.σ)

    def var(self):
        return self.σ**2
    
    def inverse(self,p):
        self._inverse_check(p)
        p = np.array(p)
        inv = self.μ + self.σ*np.sqrt(2)*sp.special.erfinv(2*p - 1)
        return inv


class KumaraswamyDistribution(Distribution):
    def __init__(self,α,β,x_min=0,x_max=1):
        if α <= 0 or β <= 0:
            raise ValueError('α and β must be higher than 0')
        self.α = α
        self.β = β
        self.x_min = x_min
        self.x_max = x_max

    def __repr__(self):
        return 'Kumaraswamy(α=%2.2f,β=%2.2f) on domain [%2.2f,%2.2f]' \
            % (self.α,self.β,self.x_min,self.x_max)

    def __add__(self,μ):
        return KumaraswamyDistribution(self.α,self.β,self.x_min+μ,self.x_max+μ)

    def __sub__(self,μ):
        return self.__add__(-μ)

    def raw_moment(self,n):
        '''Returns E[X^n]'''
        Γ = scipy.special.gamma
        α,β = self.α, self.β
        m = β*Γ(1 + n/α)*Γ(b) / Γ(1+β+n/α)
        return m

    def __rmul__(self,σ):
        return KumaraswamyDistribution(self.α,self.β,σ*self.x_min,σ*self.x_max)

    def E(self):
        std_mean = self.raw_moment(1)
        return std_mean*(self.x_max - self.x_min) + self.x_min

    def var(self):
        

    def inverse(self,p):
        self._inverse_check(p)
        p = np.array(p)
        α,β = self.α,self.β        
        std_inv = (1 - (1-p)**(1/β))**(1/α)
        return std_inv*(self.x_max - self.x_min) + self.x_min

    def var(self):
        return NotImplementedError


class Market(object):
    def __init__(self,r_distr,x_distrs,cop):
        self.r_distr = r_distr
        self.x_distrs = x_distrs
        self.p = len(x_distrs) + 1 # +1: bias feature
        self.cop = cop

        X,r = self.sample(100000)
        self.X_max = np.percentile(np.linalg.norm(X,axis=1),98) # Not quite...
        self.r_bar = np.max(np.abs([np.percentile(r,2),np.percentile(r,98)]))

    def feature_correlation(i):
        '''Returns empirical correlation of returns with feature i'''
        X,r = self.sample(100000)
        corr_mat = np.corrcoef(X[:,i],r)
        return corr_mat[0,1]
        
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
    def __init__(self,r_distr,x_distrs,corr_vector=None):
        p = len(x_distrs)
        if corr_vector:
            if len(corr_vector) is not p:
                raise ValueError('Invalid correlation vector. Must have one entry for each feature.')
            if sum(v) >= 1.0:
                raise ValueError('Must have elements summing to less than 1.')
            v = corr_vector
        else:
            ε = 0.001
            α = (1-ε)/p                     # Information from every feature
            v = [α]*p + [1]
        Σ = np.empty((p+1,p+1))
        Σ[:-1,:-1] = np.eye(p)
        Σ[-1,:] = v
        Σ[:,-1] = v
        cop = GaussianCopula(Σ)
        super().__init__(r_distr,x_distrs,cop)
