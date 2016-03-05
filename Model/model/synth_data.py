import random as rm
import multiprocessing

import numpy as np
import scipy as sp
import scipy.special
import scipy.stats

from .distrs import *

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
        return scipy.stats.norm.cdf(z)


class Market(object):
    def __init__(self,x_distrs,r_distr,cop):
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
        '''Returns X,r tuple sampled from market distribution.'''
        distrs = tuple(self.x_distrs) + (self.r_distr,)
        unif_sample = self.cop.sample(n)
        sample = np.array([d.inverse(us) for d,us in zip(distrs,unif_sample.T)]).T
        X = sample[:,0:-1]
        r = sample[:,-1]
        bias = np.ones(n)
        X = np.c_[bias,X]
        return X,r


class GaussianMarket(Market):
    '''Represent synthetic market distribution where dependance between features and returns
    stems from a gaussian copula. It is assumed that features are jointly
    independant. Features and return distributions must be provided from the user.
    '''
    def __init__(self,Xs,R,corr_vector=None):
        '''Instantiates the market distribution using user supplied distributions. if no
        correlation vector is present, then Corr(Xᵢ,R) = Corr(Xⱼ,R) = (1-ε)/p.

        Args:
            r_distr return distribution. Must support the inverse() method.
            x_distrs: list of features distribution, ie. [Xᵢ]
            corr_vector [optional]: Vector [Corr(Xᵢ,R)]
        '''
        p = len(Xs)
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
        super().__init__(Xs,R,cop)


class MarketDiscreteDistribution(DiscreteDistribution):
    def __init__(self,X,R):
        self.X = DiscreteDistribution(X)
        self.R = DiscreteDistribution(R)
        self.n,self.p = self.X.points.shape

    def sample(self,k,p=None):
        """Samples from the discrete market distribution

        :param k: int - Size of the sample 
        :param p: list - of features to be returned. If not present, all the features are returned
        :returns: X and r, where X is a matrix of features size k x len(p) and r is size k
        :rtype: tuple

        """
        if p is None:
            p = range(self.p)
        ks = np.random.choice(self.n,k)
        X = np.take(self.X.points,ks,axis=0)
        r = np.take(self.R.points,ks,axis=0)
        X = X[:,p]
        return X,r

    @property
    def X_max(self):
        return max(np.linalg.norm(self.X.points,axis=1))

    @property
    def r_max(self):
        return max(self.R.points)

    @property
    def r_min(self):
        return min(self.R.points)

    @property
    def r_bar(self):
        return max(np.abs([self.r_max,self.r_min]))
