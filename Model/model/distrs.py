import random as rm
import warnings

import numpy as np
import scipy as sp
import scipy.special

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

def RandomSample(X,n):
    try:
        return X.random_sample(n)
    except NotImplementedError:
        raise NotImplementedError

    
class Distribution(object):
    def _inverse_check(self,p):
        if np.min([p])<0 or np.max([p])>1:
            raise ValueError('p must lie in (0,1)')

    def sample(self,n):
        unif_sample = np.random.uniform(0,1,n)
        return self.inverse(unif_sample)

    def __pow__(self,n):
        return PowerRandomVariable(self,n)

    def __add__(self,x):
        if isinstance(x,(int,float)):
            return self.scalar_add(x)
        elif isinstance(x,Distribution):
            return self.distr_add(x)
        else:
            raise TypeError('Unsupported operation.')

    def distr_add(self,X):
        return SumRandomVariable(self,X)

    def __radd__(self,x):
        return self.__add__(x)

    def __sub__(self,x):
        return self.add(-x)

    def __rsub__(self,x):
        return self.add(-x)

    def random_sample(self,n):
        raise NotImplementedError


class UnknownDistribution(Distribution):
    pass


class FunctionDistribution(Distribution):
    def __init__(self,X,f):
        self.X = X
        self.f = f

    def sample(self,n):
        return self.f(self.X.sample(n))


class PowerRandomVariable(UnknownDistribution):
    def __init__(self,X,n):
        self.X = X
        self.n = n

    def __str__(self):
        return '%s**%d' % (str(self.X),self.n)

    def sample(self,n):
        return self.X.sample(n)**2

    def general_sum(self,μ):
        if μ is not 0:
            raise NotImplemented('Only works for +0')
        else:
            return self

class SumRandomVariable(UnknownDistribution):
    def __init__(self,X1,X2):
        self.Xs = [X1,X2]

    def __add__(self,X):
        return SumRandomVariable(self,X)

    def __str__(self):
        return ' + '.join([str(X) for X in self.Xs])

    def E(self):
        sum = 0
        for X in self.Xs:
            try:
                sum += E(X)
            except:
                raise ValueError('Expectation not defined for',X)
        return sum

    def add_variable(self,Xi):
        return self.Xs.append(Xi)

    def sample(self,n):
        sums = [X.sample(n) for X in self.Xs]
        return np.sum(sums,axis=0)


class DiscreteDistribution(Distribution):
    def __init__(self,points):
        self.n = len(points)
        self.points = np.array(points)

    def __call__(self):
        return self.points

    def E(self):
        return np.mean(self.points,axis=0)

    def var(self):
        return np.var(self.points,axis=0)

    def sample(self,k):
        ks = np.random.choice(range(self.n),k)
        return np.take(self.points,ks,axis=0)

    def __rmul__(self,a):
        return DiscreteDistribution(a*self.points)

    @property
    def min(self):
        return np.min(self(),axis=0)

    @property
    def max(self):
        return np.max(self(),axis=0)

    @property
    def support(self):
        return np.array([self.min,self.max]).T


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

    @property
    def support(self):
        return [self.a,self.b]


class NormalDistribution(Distribution):
    def __init__(self, μ=0, σ=1):
        self.μ = μ
        self.σ = σ

    def __str__(self):
        return 'N(μ=%2.2f,σ=%2.2f)' % (self.μ, self.σ)

    def E(self):
        return self.μ

    def scalar_add(self,μ0):
        return NormalDistribution(self.μ+μ0,self.σ)

    def distr_add(self,X):
        if isinstance(X,NormalDistribution):
            return NormalDistribution(self.μ + X.μ, np.sqrt(self.σ**2 + X.σ**2))
        else:
            return super().distr_add(X)

    def var(self):
        return self.σ**2
    
    def inverse(self,p):
        self._inverse_check(p)
        p = np.array(p)
        inv = self.μ + self.σ*np.sqrt(2)*sp.special.erfinv(2*p - 1)
        return inv

    @property
    def support(self):
        return [-np.infty,+np.infty]


class StudentTDistribution(Distribution):
    def __init__(self,ν,μ=0,σ=1):
        self.ν = ν
        self.μ = μ
        self.σ = σ

    def __str__(self):
        if self.μ is not 0 and self.σ is not 1: 
            return '%2.2f + %2.2f*Student(ν=%2.2f)' % (self.μ,self.σ,self.ν)
        elif self.μ is 0 and self.σ is not 1:
            return '%2.2f*Student(ν=%2.2f)' % (self.σ,self.ν)
        elif self.μ is not 0 and self.σ is 1:
            return '%2.2f + Student(ν=%2.2f)' % (self.μ,self.ν)
        else:
            return 'Student(ν=%2.2f)' % self.ν

    def E(self):
        if self.ν <= 1:
            raise ValueError('Undefined expected value')
        return self.μ

    def var(self):
        ν = self.ν
        σ = self.σ
        if ν <= 1:
            raise ValueError('Undefined variance')
        elif ν <= 2:
            return np.infty
        else:
            return σ**2 * ν/(ν - 2)

    def scalar_add(self,μ):
        return StudentTDistribution(self.ν,self.μ + μ,self.σ)

    def __rmul__(self,a):
        return StudentTDistribution(self.ν,self.μ,a*self.σ)

    def __mul__(self,a):
        return self.__rmul__(a)

    def inverse(self,p):
        self._inverse_check(p)
        p = np.array(p)
        ν,μ,σ = self.ν,self.μ,self.σ
        π = np.pi

        if ν not in {1,2,4}:
            raise NotImplementedError('Only implemented for ν=1,2 or 4')
        # https://en.wikipedia.org/wiki/Quantile_function#Student.27s_t-distribution
        α = 4*p*(1-p)
        q = np.cos(1/3 * np.arccos(np.sqrt(α))) / np.sqrt(α)
        if ν is 1:
            std_inverse = np.tan(π*(p-1/2))
        elif ν is 2:
            std_inverse = 2*(p-1/2)*np.sqrt(2/α)
        if ν is 4:
            std_inverse = np.sign(p - 1/2)*2*np.sqrt(q - 1)
        return μ + σ * std_inverse

    @property
    def support(self):
        return [-np.infty,+np.infty]


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

    def general_sum(self,μ):
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

    @property
    def support(self):
        return [self.x_min,self.x_max]
