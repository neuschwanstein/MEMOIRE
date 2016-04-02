import multiprocessing as mp

import cvxpy as cvx
import numpy as np
from numpy.linalg import norm

from helper.stats import empirical_cdf

from .utility import *
from .math_ops import *


class BaseProblem(object):
    def __init__(self,u,Rf=0):
        self.u = u
        self.Rf = Rf

    def cost(self,p,r):
        return -self.u(p*r + (1-p)*self.Rf)

    def total_cost(self,X,r,q,λ):
        n = len(r)
        p = X@q
        return 1/n * sum(self.cost(p,r)) + λ*norm(q)**2


class AbstractProblem(BaseProblem):
    '''Abstract utility maximization problem exposing theoretical properties
    of the problem.

    '''
    def __init__(self,M,n=None,λ=None,u=None,Rf=0):
        '''Instantiates an abstract problem.

        Args:
            m: Market object
            n [optional]: Number of (theoretical) samples
            λ [optional]: Theoretical regularizer
            u: [optional]: utility function
            Rf [optional]: risk free rate (Default 0)
        '''
        self.M = M
        self._n = n
        self._λ = λ
        super().__init__(u,Rf)

    @property
    def n(self):
        if self._n is None:
            raise ValueError('n must be initialized')
        return self._n

    @n.setter
    def n(self,n):
        self._n = n

    @property
    def λ(self):
        if self._λ is None:
            raise ValueError('λ must be initialized')
        return self._λ

    @λ.setter
    def λ(self,λ):
        self._λ = λ

    @property
    def p(self):
        return self.M.p

    @property
    def r_max(self):
        t = 0.0001
        R = self.M.R
        if R.support[1] == np.infty:
            return R.inverse(1-t)
        else:
            return R.support[1]

    @property
    def r_min(self):
        t = 0.0001
        R = self.M.R
        if R.support[0] == -np.infty:
            return R.inverse(t)
        else:
            return R.support[0]

    @property
    def r_bar(self):
        return max(np.abs([self.r_min,self.r_max]))

    @property
    def k(self):
        return self.u.k

    @property
    def γ(self):
        return self.u.γ

    @property
    def X_max(self):
        raise NotImplementedError
        # return self.M.X_max

    @property
    def σ(self):
        '''Returns the σ Lipschitz attribute, ie. |c(p₁,r) - c(p₂,r)| ≤ σ|p₁ - p₂|.

        O(1)'''
        k,γ,r_bar,Rf = self.k,self.γ,self.r_bar,self.Rf
        σ = k*γ*(r_bar + Rf)
        return σ

    @property
    def α(self):
        '''Returns the α stability of the algorithm, ie. for two samples different by a single
        entry i, |ℓ(m) - ℓ'(m)| ≤ α. Here ℓ refers to the unregularized loss.

        O(X_max²/(λn))

        '''
        σ,X_max,λ,n = self.σ,self.X_max,self.λ,self.n
        α = (σ * X_max)**2 / (2*λ*n)
        return α

    @property
    def q_max(self):
        '''Maximum amplitude of the decision vector q, ie. ∥q∥₂ ≤ q_max.

        O(X_max/λ)'''
        k,Rf,γ,λ,r_max,X_max = self.k,self.Rf,self.γ,self.λ,self.r_max,self.X_max
        q_max = k*γ*(r_max-Rf)*X_max / (2*λ)
        return q_max

    @property
    def p_max(self):
        '''Maximum amplitude of the allocation scalar, ie. |p| ≤ p_max.

        O(X_max²/λ)'''
        p_max = self.q_max * self.X_max
        return p_max

    @property
    def ℓ_max(self):
        '''Returns the maximum loss produced by the algorithm, ie. ℓ_max ≥ ℓ(m,q) ∀ S, m ~ M.

        O(X_max²/λ)
        '''
        ℓ_max = self.cost(self.p_max, self.r_min)
        return ℓ_max

    @property
    def ℓ_min(self):
        '''Returns the minimum loss produced by the algorithm, ie. ℓ_min ≤ ℓ(m,q) ∀ S, m ~ M.

        O(X_max²/λ) if u keeps increasing;
        O(1) otherwise
        '''
        ℓ_min = self.cost(self.p_max, self.r_max)
        return ℓ_min

    def outsample_bound(self,δ,n=None):
        '''The returned bound holds with probability at least 1-δ over a random draw of size
        n. See Theorem2.

        O(X_max²/(λ√n))
        '''
        self.n = n = np.array(n) if n is not None else self.n
        α = self.α
        B = self.ℓ_max - self.ℓ_min

        # Ω = 2*α + (4*n*α + B)* np.sqrt(np.log(2/δ)/(2*n))
        first_part = 2*α
        second_part = 4*n*α + B
        third_part = np.sqrt(np.log(2/δ)/(2*n))
        Ω = first_part + second_part*third_part
        return Ω

    def Ω(self,δ,n=None):
        return self.outsample_bound(δ,n)

    def find_optimal_λ(self,n,λ0=1.0):
        raise NotImplementedError
        # def objective(λ):
        #     self.λ = λ
        #     return np.mean(self.risk_distribution(n))
        # result = scipy.optimize.minimize(objective,λ0)
        # return result


class Problem(BaseProblem):
    '''Realized instance of a utility maximization problem.'''

    def __init__(self,X,r,λ,u,Rf=0):
        '''Instantiates the Problem with features and returns samples, as well as regularization
        and utility function.

        Args:
            X: Numeric features matrix of size [n, (p+1)]
            r: Numeric returns vector of size [n]
            λ: Numeric regularizater constant
            u: utility function
            Rf (optional): risk-free rate
        '''
        self.X = X
        self.r = r
        self.λ = λ
        self.n,self.p = X.shape
        # self.solver = cvx.ECOS
        self.solver = None
        super().__init__(u,Rf)

    def _cvx_cost(self,p,r):
        '''[Internal] Returns the cvx cost for cvx Variable position p and numeric return r.

        Args:
            p: Scalar or vector of portfolio composition
            r: Scalar or vector of asset return
        '''
        return -self.u.cvx_util(cvx.mul_elemwise(r,p) + (1-p)*self.Rf)

    def insample_cost(self,q=None,f_list=None,M_square=None):
        '''Average insample cost using decision vector q.

        Args:
            q [optional]: decision vector q
        '''
        q = q if q is not None else self.q
        n,X,r = self.n,self.X,self.r
        # if M_square is not None:
        #     X = np.sign(X)*np.abs(np.maximum(X,np.sqrt(M_square)))
        p = X@q
        return 1/n * sum(self.cost(p,r))

    def insample_CE(self,q=None,f_list=None,M_square=None):
        return self.u.inverse(-self.insample_cost(q,f_list,M_square))

    def solve(self):
        """Determines optimal decision vector q of the regularized problem at hand and returns in
        sample cost, ie. R'(q').

        :returns: average in-sample cost
        :rtype: double

        """
        n,X,r,λ = self.n,self.X,self.r,self.λ

        def total_cost(q):
            p = X*q
            return 1/n * cvx.sum_entries(self._cvx_cost(p,r)) + λ*cvx.norm(q)**2

        q = cvx.Variable(self.p)
        objective = cvx.Minimize(total_cost(q))
        problem = cvx.Problem(objective)
        problem.solve(solver=self.solver)

        if problem.status == 'unbounded':
            raise Exception(problem.status)

        self.q = q.value.A1
        return self.insample_cost()

    def outsample_cost(self,X,R):
        """R_true(q_hat), where R_true is determined using (large) sample of features and
        returns. The value will be exact if X and r represent all possible outcomes of a
        discrete distribution.

        :param X: X features matrix
        :param R: R returns vector
        :returns: outsample cost according to market distribution
        :rtype: float

        """
        n,_ = X.shape
        p = np.dot(X,self.q)
        total_cost = 1/n * sum(self.cost(p,R))
        return total_cost


class MaskedProblem(Problem):
    '''Represents problem with hidden features.'''

    def __init__(self,fs,X,r,λ,u,Rf=0):
        """Instantiates a masked problem with features and returns samples, the list of hidden
        features, regularization and so on.
        """
        self.fs = fs
        X = X[:,fs]
        super().__init__(X,r,λ,u,Rf)

    def outsample_cost(self,X,R):
        X = X[:,self.fs]
        return super().outsample_cost(X,R)


class SaturatedFeaturesProblem(Problem):

    def outsample_cost(self,X,R):
        M = np.max(np.abs(self.X),axis=0)
        X = np.sign(X)*np.abs(np.maximum(X,M))
        return super().outsample_cost(X,R)


class SaturatedFeaturesMaskedProblem(MaskedProblem,SaturatedFeaturesProblem):
    pass


class SaturatedNormProblem(Problem):
    '''Applies empirical cdf to its in-sample to perform an size-independant training.'''

    def __init__(self,X,r,λ,u,Rf=0):
        self.ρs = norm(X,axis=1)
        X = self.τ(X)
        super().__init__(X,r,λ,u,Rf)

    @property
    def cdf(self):
        return empirical_cdf(self.ρs)

    def τ(self,X):
        ρs = norm(X,axis=1)
        X = (X.T/ρs).T                # Unit vector
        k = self.cdf(ρs)
        X = (k*X.T).T
        return X

    def outsample_cost(self,X,R):
        X = self.τ(X)
        return super().outsample_cost(X,R)


class SaturatedNormMaskedProblem(MaskedProblem,SaturatedNormProblem):
    pass


class ProblemsDistribution(BaseProblem):
    '''Set of routines for parallel sampling of variables of interest.'''

    def __init__(self,M,n,λ,u,Rf=0,problem_t=Problem):
        '''Instantiates a distributed problem class using a market distribution M.'''
        self.M = M
        self.X = M.X.points
        self.R = M.R.points
        self.n = n
        self.λ = λ
        self.ps = None
        self.problem_t = problem_t
        super().__init__(u,Rf)

    def sample(self,m,specific_args={},par=True):
        '''Samples m problems.'''
        _ = 0
        self.specific_args = specific_args
        if par:
            ctx = mp.get_context('forkserver')
            with ctx.Pool(ctx.cpu_count()) as pool:
                ps = pool.map(self._p_hat,[_]*m)
        else:
            ps = [self._p_hat(_) for _ in range(m)]
        self.ps = ps
        return ps

    def _p_hat(self,_):
        '''Samples a problem p_hat using n market observations.'''
        X,R = self.M.sample(self.n)
        default_args = { 'X':X,'r':R,'λ':self.λ,'u':self.u,'Rf':self.Rf }
        default_args = { **default_args, **self.specific_args }
        p_hat = self.problem_t(**default_args)
        p_hat.solve()
        return p_hat

    @property
    def ps(self):
        '''Returns a list of the samples of the problem with specified parameters.'''
        if not self._ps:
            raise ValueError('The problems must be sampled first!')
        return self._ps

    @ps.setter
    def ps(self,ps):
        self._ps = ps

    @property
    def qs(self):
        return np.array([p.q for p in self.ps])

    @property
    def Rs_ins(self):
        '''Returns a list of the insample risks R_hat, with λ=0.

        R'(q') = 1/n ∑ℓ(mᵢ,q')
        '''
        return np.array([p.insample_cost() for p in self.ps])

    @property
    def Rs_oos(self):
        '''Returns a list of the outsample risks R∗(^q), with λ=0.

        R∗(q') = E[ℓ(M,q')]
        '''
        return np.array([p.outsample_cost(self.X,self.R) for p in self.ps])

    @property
    def CEs_ins(self):
        '''Returns a list of the insample certainty equivalents.

        CE'(q') = u^{-1}(-R'(q'))
        '''
        return self.u.inverse(-self.Rs_ins)

    @property
    def CEs_oos(self):
        '''Returns a list of the outsample CE CE∗(^q) with λ=0.

        CE∗(q') = u^{-1}(-R∗(q'))
        '''
        return self.u.inverse(-self.Rs_oos)
