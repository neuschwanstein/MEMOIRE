import numpy as np
import cvxpy as cvx

import model.distrs as d
from model.distrs import E,Var

## Sampling and solving functions

def get_X(p):
    '''List of features marginal distribution'''
    bias = d.DiracDistribution(1)
    X = [bias] + [d.RademacherDistribution() for _ in range(p)]
    return X

def get_R():
    '''Market return marginal distribution'''
    return d.DiscreteDistribution([-1,0,2])

def get_M(p):
    '''Market (X,R) full distribution'''
    X = get_X(p)
    R = get_R()
    M = d.IndependantMixedDistribution(X+[R])
    return M

def sample_market(M,n):
    '''(X,R) tuple of samples from the market'''
    Z = M.sample(n)
    X,r = Z[:,:-1],Z[:,-1]
    return X,r

def solve(X,r):
    '''optimal q' from a sample X,r'''
    X = np.array(X); r = np.array(r);
    n,p = X.shape
    q = cvx.Variable(p)
    obj = 1/n * cvx.sum_entries(cvx.mul_elemwise(r,X*q)) - cvx.norm(q)**2
    prob = cvx.Problem(cvx.Maximize(obj))
    prob.solve()
    q = q.value.A1
    return q

def ins_loss(X,r,q):
    '''In-sample loss using decision q applied to sample X,r'''
    ins_loss = np.mean(r*(X@q))
    return ins_loss

def oos_loss(q):
    '''Out-sample loss using decision q'''
    p = len(q)
    X = X(p)
    R = R()
    oos_loss = E(R) * E(np.inner(q,X)) # Independance assumption
    return oos_loss

def loss(X,r,q):
    '''Relative loss'''
    return ins_loss(X,r,q) - oss_loss(q)


## Beta functions

def new_sample(M,X,r):
    '''From a sample X,r returns an identical sample except in its first position where it has
    been resampled from M'''
    X2 = np.copy(X); r2 = np.copy(r)
    X_new,r_new = sample_market(M,1)
    X2[0] = X_new; r2[0] = r_new
    return X2,r2

def beta(p,n,m1,m2=1):
    '''Empirical beta stability using p features and sample of size n. Max taken over m1*m2 trials'''
    p = int(p); n = int(n);
    M = get_M(p)
    loss = np.empty(m1*m2)

    for i in range(m1):
        X1,r1 = sample_market(M,n)
        X2,r2 = new_sample(M,X1,r1)
        q1 = solve(X1,r1)
        q2 = solve(X2,r2)
        X,r = sample_market(M,m2)
        for j,(X,r) in enumerate(zip(X,r)):
            diff_loss = np.abs(ins_loss(X,r,q1) - ins_loss(X,r,q2))
            loss[j+i*m2] = diff_loss

    return loss

def art_beta(p,n):
    p = int(p); n = int(n);
    first_X = [1] + [1 for _ in range(p)]
    first_R = -1
    opp_X = [1] + [-1 for _ in range(p)]
    opp_R = -1
    X1 = [first_X for _ in range(n)]; R1 = [first_R for _ in range(n)]
    X2 = [opp_X for _ in range(n)]; R2 = [opp_R for _ in range(n)]
    q1 = solve(X1,R1)
    q2 = solve(X2,R2)
    return q1,q2

def th_beta(p,n):
    '''Theoretical beta stability bound'''
    σ = 2; λ = 1;
    β = ((p+1)*σ**2)/(λ*n)
    return β
