import cvxpy as cvx
import numpy as np


def split_and_normalize(newsmarket,sz=None):
    if sz is None:
        sz = int(0.8*len(newsmarket))

    train,test = newsmarket[:sz],newsmarket[sz:]
    mean = train.X.mean(axis=0)
    std = train.X.std(axis=0)

    train['X'] = (train['X'] - mean)/std
    test['X'] = (test['X'] - mean)/std

    return train,test


def add_bias(newsmarket,bias=1):
    newsmarket['bias'] = bias*np.ones(shape=len(newsmarket))
    return newsmarket


def solve(train,u,λ,**kwargs):
    n,p = train.X.shape
    q = cvx.Variable(p)
    r = train.r.values
    X = train.X.values

    objective = cvx.Maximize(
        1/n * cvx.sum_entries(u.cvx_util(cvx.mul_elemwise(r,X*q))) -
        λ*cvx.norm(q)**2)

    problem = cvx.Problem(objective)
    problem.solve(**kwargs)

    q = q.value.A1
    return q


def CE(newsmarket,q,u,λ):
    X = newsmarket.X
    r = newsmarket.r
    return u.inverse(np.mean(u(r*(X@q))))


def get_q_scale(newsmarket,u,n=100):
    '''If under no regularization a covariate implies a very large decision in qᵢ, then we
    want to make it easier for the regularized variable to also have a large value, thus
    we decrease its value by its inverse mean size.

    '''
    qs = [solve(newsmarket.sample(100,replace=True)) for _ in range(n)]
    qs = np.array(qs)
    q_mean = np.mean(qs,axis=0)
    result = 1/q_mean
    return result
