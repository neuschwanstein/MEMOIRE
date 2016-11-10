import numpy as np
import cvxpy as cvx
import pandas as pd

from . import market,news as ns


class NewsMarket(pd.DataFrame):

    @property
    def _constructor(self):
        return NewsMarket

    @property
    def X(self):
        return self.filter(regex='^f_')

    def during(self,bool):
        return self.xs(bool,level='during')

    def __getitem__(self,key):
        if key is 'X':
            return self.X
        else:
            return super().__getitem__(key)

    def __setitem__(self,key,val):
        if key is 'X':
            cols = self.columns[self.columns.str.contains('f_')]
            try:
                self.loc[:,cols] = val.values
            except:
                self.loc[:,cols] = val
        else:
            super().__setitem__(key,val)

    @classmethod
    def load(cls,*years,features=[]):
        newsmarket = market.load(*years,what='r')
        newsmarket = newsmarket[['r']]

        if 'vol' in features:
            vol = market.load(*years,what='vol')
            newsmarket = newsmarket.join(vol,how='inner')

        if 'news' in features:
            news = ns.load(*years)
            news = news.reset_index(level=1)
            newsmarket = newsmarket.merge(news,left_index=True,right_index=True)
            newsmarket = newsmarket.set_index('during',append=True)

        return NewsMarket(newsmarket)


class NewsMarketAnalyzer(object):

    def __init__(self,newsmarket,shuffle=True,full=False,**kwargs):
        if shuffle:
            self.newsmarket = newsmarket.sample(len(newsmarket))
        else:
            self.newsmarket = newsmarket

        # Create train,test sets
        if not full:
            sz = int(0.8*len(self.newsmarket))
        else:
            sz = len(self.newsmarket)
        train,test = self.newsmarket[:sz].copy(),self.newsmarket[sz:].copy()

        mean = train.X.mean(axis=0)
        std = train.X.std(axis=0)
        train['X'] = (train['X'] - mean)/std
        test['X'] = (test['X'] - mean)/std

        # add biases
        train.insert(loc=len(train.columns),column='f_bias',value=1)
        test.insert(loc=len(test.columns),column='f_bias',value=1)

        self.train = train
        self.test = test

        self.params = kwargs

    def solve(self,verbose=False,solver=None,**kwargs):
        params = {**self.params, **kwargs}
        u = params['u']
        λ = params['λ']

        n,p = self.train.X.shape
        q = cvx.Variable(p)
        r = self.train.r.values
        X = self.train.X.values

        objective = cvx.Maximize(
            1/n * cvx.sum_entries(u.cvx_util(cvx.mul_elemwise(r,X*q))) -
            λ*cvx.norm(q)**2)

        problem = cvx.Problem(objective)
        problem.solve(solver=solver,verbose=verbose)

        if p == 1:              # q is a scalar
            self.q = np.array([q.value])
        if p > 1:               # q is a vector
            self.q = q.value.A1
        self.params['q'] = self.q
        return self.q

    def CE(self,newsmarket,**kwargs):
        params = {**self.params, **kwargs}
        u = params['u']
        try:
            q = params['q']
        except KeyError:
            q = self.solve(**params)

        X = newsmarket.X
        r = newsmarket.r
        return u.inverse(np.mean(u(r*(X@q))))

    def train_CE(self,**kwargs):
        return self.CE(self.train,**kwargs)

    def test_CE(self,**kwargs):
        return self.CE(self.test,**kwargs)

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
