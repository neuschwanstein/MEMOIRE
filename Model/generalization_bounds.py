import numpy as np
from config import *
import objective as obj
import matplotlib.pyplot as plt

k = 1                           # ? See paper...
Xmax = 1.96                     # ? see paper...
delta = 0.05

def get_bound(n):
    alpha = (r_max + Rf)**2 * Xmax**2/(2*Lambda*n)
    bound = 2*alpha + (4*n*alpha + (r_max-Rf)/(rBar+Rf)*n*alpha - Rf)*np.sqrt(np.log(2/delta)/(2*n))
    return bound

if (__name__ == "__main__"):
    bounds = []
    for n in xrange(100,10000,100):
        bound = get_bound(n)
        bounds.append(bound)

    plt.plot(range(100,10000,100),bounds)
    plt.show()
