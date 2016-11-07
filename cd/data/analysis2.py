import cvxpy as cvx
import numpy as np

from cd.datasets.newsmarket import NewsMarket


class NewsMarketAnalyzer(object):

    def __init__(self,newsmarket):
        self.newsmarket = newsmarket

        # Create train,test sets
        sz = int(0.8*len(newsmarket))
        train,test = newsmarket[:sz],newsmarket[sz:]

        mean = train.X.mean(axis=0)
        std = train.X.std(axis=0)
        train['X'] = (train['X'] - mean)/std
        test['X'] = (test['X'] - mean)/std

        # add biases
        train['f_bias'] = np.ones(shape=len(train))
        test['f_bias'] = np.ones(shape=len(test))

        self.train = train
        self.test = test

    def solve(self,u=None,λ=None,**kwargs):
        if u is None:
            u = self.u
        if λ is None:
            λ = self.λ

        n,p = self.train.X.shape
        q = cvx.Variable(p)
        r = self.train.r.values
        X = self.train.X.values

        objective = cvx.Maximize(
            1/n * cvx.sum_entries(u.cvx_util(cvx.mul_elemwise(r,X*q))) -
            λ*cvx.norm(q)**2)

        problem = cvx.Problem(objective)
        problem.solve(**kwargs)

        self.q = q.value.A1
        return self.q

    @classmethod
    def CE(newsmarket,q,u):
        X = newsmarket.X
        r = newsmarket.r
        return u.inverse(np.mean(u(r*(X@q))))

    def train_CE(self,q=None,u=None):
        if q is None:
            q = self.q
        if u is None:
            u = self.u
        return NewsMarketAnalyzer.CE(self.train,q,u)

    def test_CE(self,q=None,u=None):
        if q is None:
            q = self.q
        if u is None:
            u = self.u
        return NewsMarketAnalyzer.CE(self.test,q,u)


# def get_q_scale(newsmarket,u,n=100):
#     '''If under no regularization a covariate implies a very large decision in qᵢ, then we
#     want to make it easier for the regularized variable to also have a large value, thus
#     we decrease its value by its inverse mean size.

#     '''
#     qs = [solve(newsmarket.sample(100,replace=True)) for _ in range(n)]
#     qs = np.array(qs)
#     q_mean = np.mean(qs,axis=0)
#     result = 1/q_mean
#     return result
