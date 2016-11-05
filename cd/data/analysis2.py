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


def solve(train,u,λ):
    n,p = train.X.shape
    q = cvx.Variable(p)
    r = train.r.values
    X = train.X.values

    objective = cvx.Maximize(
        1/n * cvx.sum_entries(u.cvx_util(cvx.mul_elemwise(r,X*q))) -
        λ*cvx.norm(q)**2)

    problem = cvx.Problem(objective)
    problem.solve(verbose=True)

    q = q.value.A1
    return q


def CE(train,q,u,λ):
    X = train.X
    r = train.r
    return u.inverse(np.mean(u(r*(X@q))))
