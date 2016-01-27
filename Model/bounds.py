import multiprocessing

import numpy as np
import numpy.random as rm
# import matplotlib.pyplot as plt

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
with multiprocessing.Pool(n_cpus) as pool:
    risk_deviation = pool.map(abs_risk_deviation, range(n_experiments))

# plt.hist(risk_sample)

print("Done.")
