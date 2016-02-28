'''experiment with synthetic data with fixed λ,p and n varies.'''

import random as rn

import numpy as np
import matplotlib.pyplot as plt

from distrs import KumaraswamyDistribution,DiscreteDistribution,NormalDistribution
from distrs import E,Var,Std
import synth_data as synth
from utility import *
import problem as pr

p = 50
n_true = 50000

def make_X():
    '''Returns list of features with EXᵢ = 0 and Var[Xᵢ] = 1'''
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

u = ExpUtility(β=0.4)
p = pr.Problem(X.points,R.points,λ=0,u=u)

print('Computing q⋆ for the discretized problem...')
p.solve()
q_star = p.q
print('Done.')

def R_star(q,λ=0):
    return p.total_cost(X.points,R.points,q,λ)

R_star_q_star = R_star(q_star)

# We want the ratio p/sqrt(n) to remain the same
p = range(1,51)
n = np.empty(len(p))
n[0] = 10
for k in range(1,len(n)):
    n[k] = n[k-1]*(p[k]/p[k-1])**2
