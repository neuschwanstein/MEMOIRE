'''experiment with synthetic data with fixed λ,p and n varies.'''

import random as rn

import numpy as np
# import matplotlib.pyplot as plt

from distrs import KumaraswamyDistribution,DiscreteDistribution,NormalDistribution
from distrs import E,Var,Std
import synth_data as synth
from utility import *
import problem as pr

p = 10
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

# u = LipschitzExpUtility(1,-1)
u = LinearPlateauUtility(4,60) 
p = pr.Problem(X.points,R.points,λ=0,u=u)

print('Computing q⋆ for the discretized problem...')
p.solver = None
p.solve()
q_star = p.q
print('Done.')

def R_star(q,λ=0):
    return p.total_cost(X.points,R.points,q,λ)

R_star_q_star = R_star(q_star)

n_experiments = 400
λ = 3
ns = np.arange(100,1050,50)

oos = np.empty(shape=(n_experiments,len(ns)))
sbpt = np.empty(shape=(n_experiments,len(ns)))

for i,n in enumerate(ns):
    print('Computing with n=%d' % n)
    prs = pr.ProblemsDistribution(M,n,λ,u,Rf=0)
    prs.sample(n_experiments)
    
    Rs_oos = [R_star(q_hat) for q_hat in prs.qs]
    oos[:,i] = np.abs(Rs_oos - prs.Rs)
    sbpt[:,i] = np.abs(R_star_q_star - prs.Rs)

# oos = np.ma.masked_invalid(oos)
# sbpt = np.ma.masked_invalid(sbpt)

# # Plotting analysis
# def rate_sqrtn(x,a,b): return a * 1/np.sqrt(x-b)
# def rate_1n(x,a,b): return a * 1/(x-b)

# from scipy.optimize import curve_fit
# params_sqrtn,cov_sqrtn = curve_fit(rate_sqrtn,ns,mean_oos)
# params_1n,cov_1n = curve_fit(rate_1n,ns,mean_oos)

# plt.plot(ns,mean_oos,ns,rate_sqrtn(ns,*params_sqrtn),ns,rate_1n(ns,*params_1n))
# plt.legend(['Empirical data','1/sqrt(n) fit','1/n fit'])
# plt.show()
