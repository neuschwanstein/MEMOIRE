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
from problem import Problem


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


# def find_optimal_lambda(n_sample):
#     median = lambda d: np.median(
#     opt.minimize_scalar(median,


def abs_risk_deviation(n_sample):
    X,r = synth.market_sample(x_distrs,r_distr,cop,n_sample)
    sample_problem = Problem(X,r,λ,u,Rf=0)
    insample_cost = sample_problem.solve()
    outsample_cost = sample_problem.outsample_risk(X_true,r_true)
    return np.abs(insample_cost - outsample_cost)

def get_outsample_distribution(n_sample):
    outsample_distribution = \
        pool.map(abs_risk_deviation, [n_sample]*n_experiments)
    return outsample_distribution

global pool

if (__name__ == '__main__'):
    ctx = mp.get_context('forkserver')
    pool = ctx.Pool(ctx.cpu_count())
    n_samples = [50,100,200]

    for n_sample in n_samples:
        risk_deviation = get_outsample_distribution(n_sample)
        density = gaussian_kde(risk_deviation)
        x = np.linspace(np.min(risk_deviation), np.max(risk_deviation), num=40)
        plt.plot(x, density(x), label='$n=%d$'%n_sample)

    plt.legend()
    title = 'Out of Sample Risk Histogram ($p={},\lambda={}$)'.format(p,λ)
    print(title)
    plt.title(title)

    plt.savefig(filename)
    os.system('open ' + filename)
    plt.clf()

    pool.close()
    pool.join()
