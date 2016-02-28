import os

import numpy as np
# import matplotlib as mpl; mpl.use('pdf')
# mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import synth_data as synth
from utility import *
from problem import *


# filename = 'fig/plot.pdf'

n_experiments = 800
λ = 0
p = 10
n_true = 100000
u = ExpUtility(0.8)

R = synth.NormalDistribution(8,10)
X = [synth.NormalDistribution() for _ in range(p)]
market = synth.GaussianMarket(R,X)

problem = AbstractProblem(market,u=u)
problem.λ = λ
problem.n = 100

# n = 100
# λs = np.arange(0.5,5,0.1)
# mean_risk = np.empty(len(λs))
# median_risk = np.empty(len(λs))
# for i,λ in enumerate(λs):
#     problem.λ = λ
#     h = problem.risk_distribution(n)
#     mean_risk[i] = np.mean(h)
#     median_risk[i] = np.median(h)

# plt.plot(λs,mean_risk,label='Mean risk')
# plt.plot(λs,median_risk,label='Median risk')
# plt.xlabel('$\lambda$')
# plt.legend()
# title = 'Out of sample Risk Histogram ($p={},n={}$)'.format(p,n)
# plt.title(title)

# plt.savefig(filename)
# os.system('open ' + filename)
# plt.clf()


# ns = [50,100,200]

# for n in ns:
#     h = problem.risk_distribution(n)
#     density = gaussian_kde(h)
#     x = np.linspace(np.min(h),np.max(h),num=40)
#     plt.plot(x,density(x), label='$n=%d$' % n)

# plt.legend()
# title = 'Out of Sample Risk Histogram ($p={},\lambda={}$)'.format(p,λ)
# plt.title(title)

# plt.savefig(filename)
# os.system('open ' + filename)
# plt.clf()
