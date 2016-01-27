import multiprocessing
import importlib

import numpy as np
import numpy.random as rm
from scipy.stats import gaussian_kde

import synth_data as synth
from utility import *
from problem import Problem

n_experiments = 500
reg = 0.9
p = 100
n_sample = 100
n_true = 100000
u = LinearUtility(0.8)

x_distrs = [synth.NormalDistribution() for _ in range(p)]
r_distr = synth.NormalDistribution(8,10)
cop = synth.ClaytonCopula(10) # TODO Investigate meaning of the argument.

X_true,r_true = synth.market_sample(x_distrs,r_distr,cop,n_true)

# i is dummy and is only there to multiprocess the task.
def abs_risk_deviation(i):
    X,r = synth.market_sample(x_distrs,r_distr,cop,n_sample)
    sample_problem = Problem(X,r,reg,u,Rf=0)
    insample_cost = sample_problem.solve()
    outsample_cost = sample_problem.outsample_risk(X_true,r_true)
    return np.abs(insample_cost - outsample_cost)

n_cpus = multiprocessing.cpu_count()
# n_cpus = 4
with multiprocessing.Pool(n_cpus) as pool:
    risk_deviation = pool.map(abs_risk_deviation, range(n_experiments))

import matplotlib.pyplot as plt
density = gaussian_kde(risk_deviation)
x = np.linspace(np.min(risk_deviation), np.max(risk_deviation), num=40)
plt.plot(x, density(x))
plt.show()

print("Done.")
