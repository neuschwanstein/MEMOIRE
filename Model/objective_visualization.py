#This plot shows the direct effect lambda (the reg coefficient) has on the curvature of
#the objective function when taking in account a sample of n market obervations (x,r)

import config
import create_data
import objective as obj
import numpy as np
import matplotlib.pyplot as plt

cfg = config.config

n,p = 5000,1
cfg.n,cfg.p = n,p
# ls = [0,1.0,1.2,1.4,1.6,1.8,2.0]
ls = np.arange(1,5.2,0.5)

t = create_data.create_rule()

x_min,x_max=-100,100

for l in ls:
    X,r = create_data.create_data(t)
    (q1,q2),val = obj.solve_objective(X,r, l==0 and 1 or l)
    q1s = np.arange(-1,1,0.01) + q1
    q2s = np.arange(-1,1,0.01) + q2
    ys = [obj.objective(X,r,[q1,q2],l) for q2 in q2s]
    x_min = max(x_min,q2s[0])
    x_max = min(x_max,q2s[-1])
    plt.plot(q2s,ys,label="$\lambda={0}$".format(l))
    if l!=0:
        plt.plot([q2],[val],'ko')

plt.legend(loc='upper left')
plt.xlabel("$q$")
plt.ylabel("$\hat R(\hat q)$")
plt.gca().set_xlim([x_min,x_max])
plt.show()

# plt.savefig("Figures/ObjectiveVisualization.pdf", bbox_inches='tight')
