'''Creates a market with p_star features, each with the same dependance to the market,
ie. the correlation vector is constant for each of these. Then sample the market using
only the first feature, the first and the second, etc. 

Do this for mutliple utility functions.'''

import random as rn

import numpy as np
import cvxpy as cvx

from model.distrs import KumaraswamyDistribution,DiscreteDistribution,NormalDistribution
from model.distrs import E,Var,Std
import model.synth_data as synth
import model.utility as ut
import model.problem as pr

from helper.state import saver,loader
from helper.plotting import plt

p_real = 1
n_true = 50000


def make_X():
    '''Returns list of bounded features with EXᵢ = 0 and Var[Xᵢ] = 1'''
    α = lambda: rn.uniform(0.1,10)
    β = lambda: rn.uniform(0.1,10)
    Xs = [KumaraswamyDistribution(α(),β()) for _ in range(p_real)]
    Xs = [(X - E(X))/Std(X) for X in Xs]
    return Xs

R = NormalDistribution(8,10)
X = make_X()
M = synth.GaussianMarket(X,R)

X,R = M.sample(n_true)
M = synth.MarketDiscreteDistribution(X,R)

X = DiscreteDistribution(X)
R = DiscreteDistribution(R)

n_experiments = 800
λ = 3
n = 500
δ = 0.2
# βs = [1,0.99,0.5,0.1,0.01]
βs = [1]
# ps = range(1,p_real+1)
ps = [1]

for β in βs:
    u = ut.LinearPlateauUtility(β,60) 
    p_star = pr.Problem(X.points,R.points,λ=0,u=u)

    print('Computing q⋆ for the discretized problem...')
    # p_star.solver = cvx.ECOS_BB
    p_star.solver = cvx.SCS
    p_star.solve()
    q_star = p_star.q
    print('Done.')

    def R_star(q,fs=None):
        if fs is None:
            fs = range(M.p)
        return p_star.total_cost(X.points[:,fs],R.points,q,λ=0)

    def CE(q,fs=None):
        return u.inverse(-R_star(q,fs))

    R_star_q_star = R_star(q_star)

    oos = np.empty(shape=(n_experiments,len(ps)))
    ins = np.empty(shape=(n_experiments,len(ps)))
    sbpt = np.empty(shape=(n_experiments,len(ps)))
    qs = np.zeros(shape=(len(ps),n_experiments,max(ps)+1))

    CE_ins = np.empty(shape=(n_experiments,len(ps)))
    CE_star = np.empty(shape=(n_experiments,len(ps)))
    CE_lb = np.empty(shape=(n_experiments,len(ps)))

    for i,p in enumerate(ps):
        print('Computing with features up to %d' % p)
        fs = range(p+1)
        pr_th = pr.AbstractProblem(M,n,λ,u,Rf=0)

        prs = pr.ProblemsDistribution(M,n,λ,u,Rf=0)
        prs.sample(n_experiments,fs,no_par=False)

        qs[i,:,:p+1] = prs.qs

        ins[:,i] = prs.Rs
        Rs_oos = [R_star(q_hat,fs) for q_hat in prs.qs]
        oos[:,i] = np.abs(Rs_oos - prs.Rs)
        sbpt[:,i] = np.abs(R_star_q_star - prs.Rs)

        CE_ins[:,i] = prs.CEs
        CE_star[:,i] = [CE(q_hat,fs) for q_hat in prs.qs]
        CE_lb[:,i] = CE_star[:,i] - u.inverse(-pr_th.Ω(δ,n) - prs.Rs)


    # datas = ['CE_ins','CE_star','CE_lb','oos','sbpt','q_star','ins','qs']
    # datas = { data: eval(data) for data in datas }

    # params = ['λ','p_real','ps','n','δ','n_experiments','u','n_true']
    # params = { param: eval(param) for param in params }

    # saver(datas,params,'ut%2.2f' % β)
