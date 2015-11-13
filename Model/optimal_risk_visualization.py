#This plot shows the direct effect lambda (the reg coefficient) has on the curvature of
#the objective function when taking in account a sample of n market obervations (x,r)

import config
import create_data
import objective as obj
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

cfg = config.config

mpl.rcParams['text.usetex'] = True

n,p = 10000,1
cfg.n,cfg.p = n,p
# ls = [0,1.0,1.2,1.4,1.6,1.8,2.0]
ls = np.arange(1,10.1,0.1)
ys = np.empty(ls.size)

t = create_data.create_rule()

for i,l in enumerate(ls):
    X,r = create_data.create_data(t)
    q,_ = obj.solve_objective(X,r,l)
    ys[i] = -obj.risk(X,r,q)

plt.plot(ls,ys)
plt.xlabel("$\lambda$")
plt.ylabel("Average utility")
plt.gca().set_xlim([ls[0],ls[-1]])

plt.show()
    
# plt.savefig("Figures/AverageUtility.pdf", bbox_inches='tight')
