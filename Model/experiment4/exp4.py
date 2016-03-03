'''experiment with synthetic data with fixed n,p and λ varies.'''

import random as rn

import numpy as np
import matplotlib.pyplot as plt

from model.distrs import KumaraswamyDistribution,DiscreteDistribution,NormalDistribution
from model.distrs import E,Var,Std
import model.synth_data as synth
import model.utility as ut
import model.problem as pr

from helper.state import saver,loader

p = 10
n_true = 50000


def make_X():
    '''Returns list of bounded features with EXᵢ = 0 and Var[Xᵢ] = 1'''
    α = lambda: rn.uniform(0.1,10)
    β = lambda: rn.uniform(0.1,10)
    Xs = [KumaraswamyDistribution(α(),β()) for _ in range(p)]
    Xs = [(X - E(X))/Std(X) for X in Xs]
    return Xs

R = NormalDistribution(8,10)
X = make_X()
M = synth.GaussianMarket(X,R)

X,R = M.sample(n_true)
M = synth.MarketDiscreteDistribution(X,R)

X = DiscreteDistribution(X)
R = DiscreteDistribution(R)

# u = LipschitzExpUtility(1,-1)
u = ut.LinearPlateauUtility(0.7,60) 
p = pr.Problem(X.points,R.points,λ=0,u=u)

print('Computing q⋆ for the discretized problem...')
p.solver = None
p.solve()
q_star = p.q
print('Done.')

def R_star(q,λ=0):
    return p.total_cost(X.points,R.points,q,λ)

def CE(q,λ=0):
    return u.inverse(-R_star(q,λ))

R_star_q_star = R_star(q_star)

n_experiments = 20
λs = np.arange(13)
n = 200
δ = 0.2

oos = np.empty(shape=(n_experiments,len(λs)))
sbpt = np.empty(shape=(n_experiments,len(λs)))

CE_ins = np.empty(shape=(n_experiments,len(λs)))
CE_star = np.empty(shape=(n_experiments,len(λs)))
CE_lb = np.empty(shape=(n_experiments,len(λs)))

for i,λ in enumerate(λs):
    print('Computing with λ=%d' % λ)
    pr_th = pr.AbstractProblem(M,n,λ,u,Rf=0)
    
    prs = pr.ProblemsDistribution(M,n,λ,u,Rf=0)
    prs.sample(n_experiments)
    
    Rs_oos = [R_star(q_hat) for q_hat in prs.qs]
    oos[:,i] = np.abs(Rs_oos - prs.Rs)
    sbpt[:,i] = np.abs(R_star_q_star - prs.Rs)

    CE_ins[:,i] = prs.CEs
    CE_star[:,i] = [CE(q_hat) for q_hat in prs.qs]
    CE_lb[:,i] = CE_star[:,i] - u.inverse(-pr_th.Ω(δ,n) - prs.Rs)


datas = ['CE_ins','CE_star','CE_lb','oos','sbpt']
datas = { data: eval(data) for data in datas }

params = ['λs','n','δ','n_experiments','u']
params = { param: eval(param) for param in params }

saver(datas,params,'graph1')
