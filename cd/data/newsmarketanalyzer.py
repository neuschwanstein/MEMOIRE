import cvxpy as cvx
import numpy as np


class NewsMarketAnalyzer(object):

    def __init__(self,newsmarket,shuffle=True):
        if shuffle:
            self.newsmarket = newsmarket.sample(len(newsmarket))
        else:
            self.newsmarket = newsmarket

        # Create train,test sets
        sz = int(0.8*len(self.newsmarket))
        train,test = self.newsmarket[:sz],self.newsmarket[sz:]

        mean = train.X.mean(axis=0)
        std = train.X.std(axis=0)
        train['X'] = (train['X'] - mean)/std
        test['X'] = (test['X'] - mean)/std

        # add biases
        train.insert(loc=len(train.columns),column='f_bias',value=1)
        test.insert(loc=len(test.columns),column='f_bias',value=1)

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

        if p == 1:              # q is a scalar
            self.q = np.array([q.value])
        if p > 1:               # q is a vector
            self.q = q.value.A1
        return self.q

    @staticmethod
    def CE(newsmarket,q,u):
        X = newsmarket.X
        r = newsmarket.r
        return u.inverse(np.mean(u(r*(X@q))))

    def train_CE(self,q=None,u=None):
        if q is None:
            q = self.q
        if u is None:
            u = self.u
        return self.CE(self.train,q,u)

    def test_CE(self,q=None,u=None):
        if q is None:
            q = self.q
        if u is None:
            u = self.u
        return self.CE(self.test,q,u)

    def cross_val(self,λs,u=None,**kwargs):
        if u is None:
            u = self.u

        res = []
        for λ in λs:
            q = self.solve(u,λ,**kwargs)
            in_ce = self.train_CE(q,u)
            out_ce = self.test_CE(q,u)
            res.append((in_ce,out_ce))

        return res


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