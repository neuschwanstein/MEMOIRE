import multiprocessing as mp
import warnings

import cvxpy as cvx
import numpy as np
from numpy.linalg import norm
import scipy.optimize

from utility import *
from math_ops import *

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
    '''Abstract utility maximization problem exposing theoretical properties of the problem.'''
    
    def __init__(self,market,n=None,λ=None,u=ExpUtility(0.8),Rf=0):
        '''Instiates an abstract problem.

        Args:
            m: Market object
            n [optional]: Number of (theoretical) samples
            λ [optional]: Theoretical regularizer
            u: utility function
            Rf [optional]: risk free rate (Default 0)
        '''
        self.m = market
        self._n = n
        self._λ = λ
        super().__init__(u,Rf)

    @property
    def n(self):
        if not self._n:
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
        return self.m.p

    @property
    def X_max(self):
        return self.m.X_max

    @property
    def r_sup(self):
        return self.m.r_sup

    @property
    def r_inf(self):
        return self.m.r_inf

    @property
    def k(self):
        return self.u.k

    @property
    def γ(self):
        p = self.p_max
        r_inf = -self.r_inf
        return D(self.u)(p*(r_inf - self.Rf) + Rf)
        # return self.u.gamma_lipschitz

    @property
    def σ(self):
        σ = self.k * self.γ * (self.r_bar + self.Rf)
        return σ

    @property
    def α(self):
        α = (self.σ * self.X_max)**2 / (2*self.λ*self.n)
        return α

    @property
    def p_max(self):
        p_max = self.k * self.γ * self.X_max**2 * (self.r_bar - self.Rf) / (2*self.λ)
        return p_max

    @property
    def B(self):
        B = self.cost(self.p_max, -self.r_bar)
        return B

    def outsample_bound(self,δ,n):
        '''The returned bound holds with probability at least 1-δ over a random draw of size
        n. See Theorem2.
        '''        
        α = self.α
        n = self.n
        B = self.B
        Ω = 2*α + (4*n*α + B)*np.sqrt(np.log(2/δ)/(2*n))
        return Ω

    def find_optimal_λ(self,n,λ0 = 1.0):
        raise NotImplementedError
        def objective(λ):
            self.λ = λ
            return np.mean(self.risk_distribution(n))
        result = scipy.optimize.minimize(objective,λ0)
        return result


class ProblemsDistribution(BaseProblem):
    '''Set of routines for parallel sampling of variables of interest.'''
    
    @property
    def ps_hat(self):
        if not self._ps_hat:
            raise ValueError('The problems must be sampled first!')
        return self._ps_hat

    @ps_hat.setter
    def ps_hat(self,ps_hat):
        self._ps_hat = ps_hat

    def __init__(self,M,n,λ,u,Rf=0):
        '''Instantiates a distributed problem class using a market distribution M.'''
        self.M = M
        self.n = n
        self.λ = λ
        self.ps_hat = None
        super().__init__(u,Rf)

    def _p_hat(self,_):
        '''Samples a problem p_hat using n market observations.''' 
        x_hat,r_hat = self.M.sample(self.n)
        p_hat = Problem(x_hat,r_hat,self.λ,self.u,self.Rf)
        try:
            p_hat.solve()
        # You dont wanna fuck it up too much as we're in parallel when calling
        except cvx.SolverError:
            print('Failed with %s solver. Retrying with CVXOPT solver...' % p_hat.solver)
            try:
                p_hat.solver = cvx.CVXOPT
                p_hat.solve()
            except cvx.SolverError:
                print('Failing with CVXOPT solver. Giving up...')
                return None
        return p_hat

    def sample(self,m):
        self.m = m
        '''Samples m problems.'''
        ctx = mp.get_context('forkserver')
        _ = 0
        with ctx.Pool(ctx.cpu_count()) as pool:
            ps_hat = pool.map(self._p_hat,[_]*m)
        self.ps_hat = ps_hat
        return ps_hat

    @property
    def qs(self):
        return [p_hat.q for p_hat in self.ps_hat]

    @property
    def Rs(self):
        '''Returns a list of the insample risks R_hat, with λ=0'''
        return np.array([p_hat.insample_cost() for p_hat in self.ps_hat])
        

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
        self.solver = cvx.ECOS
        super().__init__(u,Rf)
        
    def _cvx_cost(self,p,r):
        '''[Internal] Returns the cvx cost for cvx Variable position p and numeric return r.
        
        Args:
            p: Scalar or vector of portfolio composition
            r: Scalar or vector of asset return
        '''
        return -self.u.cvx_util(cvx.mul_elemwise(r,p) + (1-p)*self.Rf)

    def insample_cost(self,q=None):
        '''Average insample cost using decision vector q.

        Args:
            q [optional]: decision vector q
        '''
        if not q:
            q = self.q
        n,X,r,λ = self.n,self.X,self.r,self.λ
        p = X@q
        return 1/n * sum(self.cost(p,r))

    def solve(self):
        '''Determines optimal decision vector q of the regularized problem at hand and returns in
        sample cost, ie. R_hat(q_hat)
        '''
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
        # if problem.status == 'optimal_inaccurate':
            # print(problem.status, " with λ =", λ)

        self.q = q.value.A1
        return self.insample_cost()

    def outsample_cost(self,X,r):
        '''R_true(q_hat), where R_true is determined using (large) sample of features and
        returns. The value will be exact if X and r represent all possible outcomes of a
        discrete distribution.

        Args:
            X: Large (ideally true) sample of the features
            r: Large (ideally true) sample of the returns
        '''
        n,_ = X.shape
        p = np.dot(X,self.q)
        total_cost = 1/n * sum(self.cost(p,r))
        return total_cost
