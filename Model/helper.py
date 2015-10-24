import numpy as np

def simplex_sample(n):
    # http://cs.stackexchange.com/a/3229
    unif = np.random.uniform
    p = [0] + [unif() for _ in xrange(n-1)] + [1]
    p = np.sort(p)
    t = p[1:] - p[:-1]
    return t
