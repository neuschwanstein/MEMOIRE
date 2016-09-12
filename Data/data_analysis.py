import matplotlib.pyplot as plt
import numpy as np

import model.utility as ut
import train_data as td


def get_CEs(train,test,u,λs):
    return [td.get_CE(train,test,λ,u) for λ in λs]


if (__name__ == '__main__'):
    # λ = 0.031622776601683791 has been not bad in the past.
    switch = True
    if switch:
        # λs = np.linspace(1e-4,0.1,15)
        λs = np.logspace(-4,-1.5,20)
        u = ut.LinearPlateauUtility(1,0.1)
        CEs = get_CEs(train,test,u,λs)
        plt.plot(λs,CEs)
        plt.xscale('log')
        plt.show()
