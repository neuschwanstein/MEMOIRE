import os
import multiprocessing

import numpy as np
import numpy.random as rm
from scipy.stats import gaussian_kde
import matplotlib as mpl
mpl.use('pdf')
import matplotlib.pyplot as plt

import synth_data as synth
from utility import *
from problem import Problem

n_experiments = 500
reg = 0.9
p = 100
# n_sample = 100
n_true = 100000
u = LinearUtility(0.8)


x_distrs = [synth.NormalDistribution() for _ in range(p)]
r_distr = synth.NormalDistribution(8,10)
cop = synth.ClaytonCopula(10) # TODO Investigate meaning of the argument.
X_true,r_true = synth.market_sample(x_distrs,r_distr,cop,n_true)

def abs_risk_deviation(n_sample):
    X,r = synth.market_sample(x_distrs,r_distr,cop,n_sample)
    sample_problem = Problem(X,r,reg,u,Rf=0)
    # print(problem.outsample_bound(n_sample,0.1))
    insample_cost = sample_problem.solve()
    outsample_cost = sample_problem.outsample_risk(X_true,r_true)
    return np.abs(insample_cost - outsample_cost)

def get_outsample_distribution(n_sample):
    n_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(n_cpus) as pool:
        outsample_distribution = \
            pool.map(abs_risk_deviation, [n_sample]*n_experiments)
    return outsample_distribution

if (__name__ == '__main__'):
    n_samples = [50,100,200,500,1000]
    results = dict()

    risk_deviation = get_outsample_distribution(100)
    density = gaussian_kde(risk_deviation)
    x = np.linspace(np.min(risk_deviation), np.max(risk_deviation), num=40)
    plt.plot(x, density(x))

    filename = 'plot.pdf'
    filename = 'fig/' + filename
    plt.savefig('fig/plot.pdf')
    os.system('open fig/plot.pdf')
    plt.clf()
