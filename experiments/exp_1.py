import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

import cd.datasets.market as mkt
from cd.datasets.newsmarket import NewsMarket as NM
from cd.data.newsmarketanalyzer import NewsMarketAnalyzer as NMA

import cd.model.utility as ut

plt.rcParams['font.family'] = 'serif'

if __name__ == '__main__':
    r = mkt.load(2007,2015)
    newsmarket = r[['r']]
    newsmarket = NM(newsmarket)

    analyzer = NMA(newsmarket,shuffle=False)

    # u = ut.LinearUtility(0.6)
    u = ut.LinearPlateauUtility(1,0.8*max(newsmarket.r))
    # u = ut.LinearPlateauUtility(0.8,0.8*max(newsmarket.r))
    # λs = np.logspace(-3,-1,15)
    λs = np.logspace(-5.5,-2,50)

    cvs = analyzer.cross_val(λs,u)

    plt.plot(λs,cvs)
    plt.xscale('log')
    plt.axis(xmax=max(λs),xmin=min(λs))
    plt.xlabel('$\lambda$')
    plt.ylabel('Returns (%)')
    plt.legend(['In-sample CE','Out-sample CE'])
    plt.title('Cross validation of CE\n%s' % str(u))
    plt.show()
