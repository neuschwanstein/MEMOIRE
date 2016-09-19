import matplotlib.pyplot as plt
import numpy as np

import model.utility as ut
import train_data as td


def get_CEs(train,test,u,λs):
    return [td.get_CE(train,test,λ,u) for λ in λs]


def get_optimal_λ(λs,CEs):
    out_CEs = list(zip(*CEs))[1]
    index = np.argmax(out_CEs)
    return λs[index]


def plot_cross_val(λs,CEs,u,log=True):
    plt.plot(λs,CEs)
    if log:
        plt.xscale('log')
    plt.axis(xmax=max(λs),xmin=min(λs))
    plt.xlabel('$\lambda$')
    plt.legend(['In-sample CE','Out-sample CE'])
    plt.title('Cross validation of CE\n%s' % str(u))
    plt.show()


def test_price_series(test,q):
    X_test = test.X.values
    r_test = test.r.values.flatten()
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
    switch = True
    if switch:
        # λs = np.linspace(1e-4,0.1,15)
        λs = np.logspace(-10,-1.5,20)
        u = ut.LinearPlateauUtility(1,1)
        CEs = get_CEs(train,test,u,λs)
        λ0 = get_optimal_λ(λs,CEs)
        p0 = td.get_optimal_decision(train.X.values,train.r.values.flatten(),λ0,u)
        plot_cross_val(λs,CEs,u)
