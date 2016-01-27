import multiprocessing

import numpy as np
import numpy.random as rm
# import matplotlib.pyplot as plt

import synth_data as synth
import objective as obj

n_experiments = 500
l = 0.9
p = 100
n_sample = 100
n_true = 100000

x_distrs = [synth.NormalDistribution() for _ in range(p)]
r_distr = synth.NormalDistribution(8,10)
cop = synth.ClaytonCopula(10) # TODO Investigate meaning of the argument.

xss_true,rs_true = synth.market_sample(x_distrs,r_distr,cop,n_true)
xss_true = obj.append_bias(xss_true)

# i is dummy and is only there to multiprocess the task.
def abs_risk_deviation(i):
    xss,rs = synth.market_sample(x_distrs,r_distr,cop,n_sample)
    xss = obj.append_bias(xss)
    q_sample,risk_sample = obj.solve_objective(xss,rs,l)
    true_risk = obj.risk(xss_true,rs_true,q_sample)
    return np.abs(risk_sample - true_risk)

n_cpus = multiprocessing.cpu_count()
with multiprocessing.Pool(n_cpus) as pool:
    risk_sample = pool.map(abs_risk_deviation, range(n_experiments))

# plt.hist(risk_sample)

print("Done.")


















