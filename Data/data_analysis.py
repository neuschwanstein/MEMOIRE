import matplotlib.pyplot as plt
import numpy as np

import model.utility as ut
import train_data as td


def get_CEs(train,test,u,λs):
    return [td.get_CE(train,test,λ,u) for λ in λs]


def get_optimal_λ(λs,CEs):
    in_CEs = list(zip(*CEs))[0]
    index = np.argmax(in_CEs)
    return λs[index]


def plot_cross_val(λs,CEs,u,log=True):
    plt.plot(λs,CEs)
    if log:
        plt.xscale('log')
    plt.axis(xmax=max(λs))
    plt.xlabel('$\lambda$')
    plt.legend(['In-sample CE','Out-sample CE'])
    plt.title('Cross validation of CE\n%s' % str(u))
    plt.show()


def test_price_series(X_test,r_test,q):
    r = r_test*(X_test@q)
    alg_price = np.cumprod(1+r)
    act_price = np.cumprod(1+r_test)
    plt.plot(list(zip(act_price,alg_price)))
    plt.axis(xmax=len(r))
    plt.xlabel('$t$')
    plt.ylabel('Portfolio value')
    plt.title('Test set performance')
    plt.legend(['Test performance','Algorithmic performance'])
    plt.show()

if (__name__ == '__main__'):
    # λ = 0.031622776601683791 has been not bad in the past.
    switch = False
    if switch:
        # λs = np.linspace(1e-4,0.1,15)
        λs = np.logspace(-4,-1.5,20)
        u = ut.LinearPlateauUtility(1,0.1)
        CEs = get_CEs(train,test,u,λs)
        λ0 = get_optimal_λ(λs,CEs)
        plt.plot(λs,CEs)
        plt.xscale('log')
        plt.show()
