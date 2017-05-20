import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import cvxpy as cvx

from cd.data import newsmarket as nm
from cd.model.kernel import sigmoid_kernel
from cd.model import utility as ut

rc('text', usetex=True)

if __name__ == '__main__':
    # u = ut.LinearUtility(0.8)
    u = ut.RiskNeutralUtility()
    data = nm.NewsMarket.load(2007,2015,features=['vol'])
    analyzer = nm.NewsMarketAnalyzer(data,u=u,
                                     shuffle=False,
                                     kernel=sigmoid_kernel)

    λs = np.logspace(-4,-2,5)
    # λs = [1e-4]
    train_ces = np.empty_like(λs)
    test_ces = np.empty_like(λs)

    for i,λ in enumerate(λs):
        print(λ)
        analyzer.solve(λ=λ)
        train_ces[i] = analyzer.train_CE()
        test_ces[i] = analyzer.test_CE()

    plt.plot(λs,train_ces,λs,test_ces)
    plt.xscale('log')
    plt.legend(['Train','Test'])
    plt.show()

    
