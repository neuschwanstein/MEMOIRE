'''Creates a market with p_star features, each with the same dependance to the market,
ie. the correlation vector is constant for each of these. Then sample the market using
only the first feature, the first and the second, etc. 

Do this for mutliple utility functions.'''

import random as rn

import numpy as np
import cvxpy as cvx

from attrdict import AttrDict

from model.distrs import StudentTDistribution,DiscreteDistribution,NormalDistribution
from model.distrs import E,Var,Std
import model.synth_data as synth
import model.utility as ut
import model.problem as pr

from helper.state import saver,loader
from helper.plotting import plt

l = AttrDict()

l.p = 25
l.n_true = 50000

l.n_experiments = 500
l.λ = 3
l.δ = 0.2
l.ns = np.arange(25,5000,50)
l.ps = np.floor(np.sqrt(l.ns)).astype(int)
l.p_max = max(l.ps)
l.Rf = 0

# Continuous market distribution
R_true = NormalDistribution(8,10)
X_true = [StudentTDistribution(ν=4) for _ in range(l.p_max)] # EXᵢ = 0, Var(Xᵢ) = 1
l.M_true = synth.GaussianMarket(X_true,R_true)           # constant corr(Xᵢ,R) = 1/p - ε

# Discretized sampled distribution in order to have real q⋆
X,R = l.M_true.sample(l.n_true)
l.M = synth.MarketDiscreteDistribution(X,R)

β = 1
r_threshold = 60
l.u = ut.LinearPlateauUtility(β,r_threshold)

print('Computing q⋆ for the discretized problem...')
p_star = pr.Problem(X,R,λ=0,u=l.u)
p_star.solver = cvx.SCS
R_star_q_star = p_star.solve()
q_star = p_star.q

R_star = p_star.insample_cost
R_star_q_star = R_star(q_star)
CE_star = p_star.insample_CE
CE_star_q_star = CE_star(q_star)
print('Done.')

for m_square in [True,False]:
    qs = np.zeros(shape=(len(l.ns),l.n_experiments,l.p_max+1))

    Rs_ins = np.empty(shape=(l.n_experiments,len(l.ns)))
    Rs_oos = np.empty(shape=(l.n_experiments,len(l.ns)))
    # Rs_lb = np.empty(shape=(len(l.ns)))

    CEs_ins = np.empty(shape=(l.n_experiments,len(l.ns)))
    CEs_oos = np.empty(shape=(l.n_experiments,len(l.ns)))

    for i,n in enumerate(l.ns):
        p = l.ps[i]
        fs = range(p+1)
        print('Computing using sample of size %d' % n)

        pr_th = pr.AbstractProblem(l.M_true,n,l.λ,l.u,l.Rf)

        prs = pr.ProblemsDistribution(l.M,n,l.λ,l.u,l.Rf)
        prs.sample(l.n_experiments,f_list=fs,par=True)

        qs[i,:,:p+1] = prs.qs

        Rs_ins[:,i] = prs.Rs
        CEs_ins[:,i] = prs.CEs
        if m_square:
            Rs_oos[:,i] = [R_star(q=p.q,f_list=fs,M_square=p.M_square) for p in prs()] # Fixme make iterable
            CEs_oos[:,i] = [CE_star(q=p.q,f_list=fs,M_square=p.M_square) for p in prs()] # Fixme make iterable
        else:
            Rs_oos[:,i] = [R_star(q=p.q,f_list=fs,M_square=None) for p in prs()]
            CEs_oos[:,i] = [CE_star(q=p.q,f_list=fs,M_square=None) for p in prs()]

    datas = ['q_star','R_star_q_star','CE_star_q_star','qs','Rs_ins','Rs_oos','CEs_ins','CEs_oos']
    datas = { data: eval(data) for data in datas }

    if m_square:
        name = 'Student-Saturated-p_over_sqrtn'
    else:
        name = 'Student-Unsaturated-p_over_sqrtn'
    saver(datas,l,name)
