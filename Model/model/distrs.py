import random as rm
import warnings

import numpy as np
import scipy as sp

class Distribution(object):
    def inverse(self,p):
        raise NotImplementedError

    def E(self):
        raise NotImplementedError

    def var(self):
        raise NotImplementedError

    def _inverse_check(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')

    def sample(self,n):
        unif_sample = np.random.uniform(0,1,n)
        return self.inverse(unif_sample)
        

def E(d):
    try:
        return d.E()
    except AttributeError as e:
        print('Object {} is not a distribution.'.format(d))
        raise e

def Var(d):
    try:
        return d.var()
    except AttributeError as e:
        print('Object {} is not a distribution.'.format(d))

def Std(d):
    try:
        return np.sqrt(d.var())
    except AttributeError as e:
        print('Object {} is not a distribution.'.format(d))


class DiscreteDistribution(Distribution):
    def __init__(self,points):
        self.n = len(points)
        self.points = points

    def E(self):
        return np.mean(self.points,axis=0)

    def var(self):
        return np.var(self.points,axis=0)

    def sample(self,k):
        ks = rmx.sample(range(self.n),k)
        return np.take(self.points,ks,axis=0)


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
        Γ = sp.special.gamma
        α,β = self.α, self.β
        m = β*Γ(1 + n/α)*Γ(β) / Γ(1+β+n/α)
        return m

    def __rmul__(self,σ):
        return KumaraswamyDistribution(self.α,self.β,σ*self.x_min,σ*self.x_max)

    def __mul__(self,σ):
        return self.__rmul__(σ)

    def __truediv__(self,σ):
        return self.__rmul__(1/σ)

    def E(self):
        std_mean = self.raw_moment(1)
        return std_mean*(self.x_max - self.x_min) + self.x_min

    def inverse(self,p):
        self._inverse_check(p)
        p = np.array(p)
        α,β = self.α,self.β        
        std_inv = (1 - (1-p)**(1/β))**(1/α)
        return std_inv*(self.x_max - self.x_min) + self.x_min

    def var(self):
        std_var = self.raw_moment(2) - self.raw_moment(1)**2
        σ = self.x_max - self.x_min
        var = σ**2 * std_var
        return var
