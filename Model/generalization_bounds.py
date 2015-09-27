import numpy as np
import config as cfg
import objective as obj

n,p = cfg.n,cfg.p

k = 1                           # ? See paper...
Xmax = 1                        # ? see paper...
M = 1                           # ? See paper...

def get_bound(n):
    alpha = k**2 * (cfg.rbar + cfg.Rf)*Xmax**2 / (2 * cfg.regul_q_norm * n)
    bound = 2*alpha(n) + (4*n + M)*np.sqrt(np.log(2.0/delta) / (2.0*n))
    return bound

bound = []
if (__name__ == "__main__"):
    for n in xrange(100,10000,100):
        bound_n = get_bound(n)
        bound.append(bound_n)
        # ...
        # Profit !
