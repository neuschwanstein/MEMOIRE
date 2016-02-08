import os
import multiprocessing as mp

import numpy as np
import numpy.random as rm
from scipy.stats import gaussian_kde
import scipy.optimize as opt

import matplotlib as mpl; mpl.use('pdf')
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt

import synth_data as synth
from utility import *
import problem


filename = 'fig/plot.pdf'

n_experiments = 800
λ = 1
p = 10
n_true = 100000
u = LinearUtility(0.8)

r_distr = synth.NormalDistribution(8,10)
x_distrs = [synth.NormalDistribution() for _ in range(p)]
market = synth.GaussianMarket(r_distr,x_distrs)

problem = problem.AbstractProblem(market,λ,u)
X_true,r_true = market.sample(n_true)

ns = [50,100,200]

for n in ns:
    h = problem.risk_distribution(n)
    density = gaussian_kde(h)
    x = np.linspace(np.min(h),np.max(h),num=40)
    plt.plot(x,density(x), label='$n=%d$' % n)

plt.legend()
title = 'Out of Sample Risk Histogram ($p={},\lambda={}$)'.format(p,λ)
print(title)
plt.title(title)

plt.savefig(filename)
os.system('open ' + filename)
plt.clf()
