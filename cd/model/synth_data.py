import numpy as np
import scipy.special
import scipy.stats

from .distrs import Distribution,DiscreteDistribution,Var,Std


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
    def __init__(self,Xs,R,cop,**kwargs):
        self.R = R
        self.Xs = Xs
        self.bias = 0 if 'bias' not in kwargs else int(kwargs['bias'])
        self.p = len(Xs) + self.bias
        self.cop = cop

    def feature_correlation(self,i):
        '''Returns empirical correlation of returns with feature i'''
        X,r = self.sample(100000)
        corr_mat = np.corrcoef(X[:,i],r)
        return corr_mat[0,1]

    def sample(self,n):
        '''Returns X,r tuple sampled from market distribution.'''
        distrs = tuple(self.Xs) + (self.R,)
        unif_sample = self.cop.sample(n)
        sample = np.array([d.inverse(us) for d,us in zip(distrs,unif_sample.T)]).T
        X = sample[:,0:-1]
        r = sample[:,-1]
        if self.bias:
            bias = np.ones(n)
            X = np.c_[bias,X]
        return X,r

    def sample_t(self,n):
        X,r = self.sample(n)
        return X*r[:,None]


class GaussianMarket(Market):
    '''Represent synthetic market distribution where dependance between features and returns
    stems from a gaussian copula. It is assumed that features are jointly
    independant. Features and return distributions must be provided from the user.
    '''
    def __init__(self,Xs,R,**kwargs):
        '''Instantiates the market distribution using user supplied distributions. if no
        correlation vector is present, then Corr(Xᵢ,R) = Corr(Xⱼ,R) = (1-ε)/p.

        Args:
            r_distr return distribution. Must support the inverse() method.
            x_distrs: list of features distribution, ie. [Xᵢ]
            corr_vector [optional]: Vector [Corr(Xᵢ,R)]
        '''
        p = len(Xs)
        if 'sigma' in kwargs:
            Σ = kwargs['sigma']
        else:
            if 'corr_vector' in kwargs:
                # Independent features with provided correlation with R
                corr_vector = kwargs['corr_vector']
            else:
                # Independent features with 1/sqrt(p) influence on R
                corr_vector = 0.95/np.sqrt(p) * np.ones(p)
            Σ = np.eye(p+1,p+1)
            Σ[:-1,-1] = corr_vector
            Σ[-1,:-1] = corr_vector

        self.Σ = Σ
        self.corr_vector = corr_vector
        cop = GaussianCopula(Σ)
        super().__init__(Xs,R,cop,**kwargs)

    def feature_correlation(self,i):
        return self.corr_vector[i]

    def qstar(self,lamb):
        q = np.empty(self.p)
        stdr = Std(self.R)
        for i,X in enumerate(self.Xs):
            q[i] = self.corr_vector[i]*stdr*Std(X)
        return 1/(2*lamb) * q


class MarketDiscreteDistribution(DiscreteDistribution):
    def __init__(self,X,R):
        self.X = DiscreteDistribution(X)
        self.R = DiscreteDistribution(R)
        self.n,self.p = self.X.points.shape

    def sample(self,k):
        """Samples from the discrete market distribution

        :param k: int - Size of the sample 
        :param p: list - of features to be returned. If not present, all the features are returned
        :returns: X and r, where X is a matrix of features size k x len(p) and r is size k
        :rtype: tuple

        """
        ks = np.random.choice(self.n,k)
        X = np.take(self.X.points,ks,axis=0)
        r = np.take(self.R.points,ks,axis=0)
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


class Feature(Distribution):
    def __init__(self,X):
        self.X = X

    def test(self):
        return 'hello'

    def __dir__(self):
        # return dir(self) + dir(self.X)
        raise NotImplementedError

    def __getattr__(self,arg):
        return getattr(self.X,arg)

