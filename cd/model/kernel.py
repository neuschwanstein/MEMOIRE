import numpy as np


def sigmoid_kernel(x1,x2):
    # Todo: add intermediate scale and loc parameters
    exp = np.exp(np.inner(x1,x2))
    return exp/(1+exp)
    
