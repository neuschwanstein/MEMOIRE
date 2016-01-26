import numpy.random as rm

import synth_data as synth
import objective as obj
from config import cfg

n_experiments = 500
λ = 0.9
p = 100
n_sample = 100
n_true = 100000

x_distrs = [synth.NormalDistribution() for _ in range(p)]
r_distr = synth.NormalDistribution(8,10)
cop = synth.ClaytonCopula(10) # TODO Investigate meaning of the argument.

xss_true,rs_true = synth.market_sample(x_distrs,r_distr,cop,n_true)

i=1
def sample_risk():
    print("Processing ",i)
    xss,rs = synth.market_sample(x_distrs,r_distr,cop,n_sample)
    q_sample,risk_sample = obj.solve_objective(xss,rs,λ)
    true_risk = obj.risk(xss_true,rs_true,q_sample)
    return np.abs(risk_sample - true_risk)

risk_sample = [sample_risk()]*n_experiments
print("Done.")
