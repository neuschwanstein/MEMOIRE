import numpy.random as rm

import synth_data
import objective as obj
from config import cfg

p,n = 3,100
cfg.p,cfg.n = p,n

# Create random distributions for features and fixed distribution for returns, with
# dependance given with Clayton copula. Then draw sample out of it.
x_mus = rm.uniform(-3,3,p)
x_vols = rm.uniform(1,6,p)
x_distrs = [synth_data.NormalDistribution(mu,vol) for mu,vol in zip(x_mus,x_vols)]
r_distr = synth_data.NormalDistribution(8,10)
cop = synth_data.ClaytonCopula(1.0)

xss,rs = synth_data.market_sample(x_distrs,r_distr,cop,n)
xss = obj.append_bias(xss)

q,val = obj.solve_objective(xss,rs,0.2)
