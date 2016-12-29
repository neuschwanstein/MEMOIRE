
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

from cd.data import newsmarket as nm
import cd.datasets.market as mkt
# import cd.datasets.newsmarket as nm
import cd.data.newsmarketanalyzer as nma

nm.

import cd.model.utility as ut


# plt.rcParams['font.family'] = 'serif'

# if __name__ == '__main__':
#     newsmarket = nm.load_all()
#     newsmarket = newsmarket.during(True)
#     vol = mkt.load_vol(2007,2015)
#     newsmarket = newsmarket.join(vol,how='inner')

#     analyzer = nma.NewsMarketAnalyzer(newsmarket,shuffle=False)

#     u = ut.LinearPlateauUtility(1,0.8*max(newsmarket.r))
#     λs = np.logspace(-5.5,-2,25)

#     cvs = analyzer.cross_val(λs,u,verbose=True)

#     plt.plot(λs,cvs)
#     plt.xscale('log')
#     plt.axis(xmax=max(λs),xmin=min(λs))
#     plt.xlabel('$\lambda$')
#     plt.ylabel('Returns (%)')
#     plt.legend(['In-sample CE','Out-sample CE'])
#     plt.title('Cross validation of CE\n%s' % str(u))
#     plt.show()
    
    # r = mkt.load(2007,2015)
    # newsmarket = r[['r']]
    # newsmarket = nm.NewsMarket(newsmarket)

    # analyzer = nma.NewsMarketAnalyzer(newsmarket,shuffle=False)

    # # u = ut.LinearUtility(0.6)
    # u = ut.LinearPlateauUtility(1,0.8*max(newsmarket.r))
    # # u = ut.LinearPlateauUtility(0.8,0.8*max(newsmarket.r))
    # # λs = np.logspace(-3,-1,15)
    # λs = np.logspace(-5.5,-2,50)

    # cvs = analyzer.cross_val(λs,u)

    # plt.plot(λs,cvs)
    # plt.xscale('log')
    # plt.axis(xmax=max(λs),xmin=min(λs))
    # plt.xlabel('$\lambda$')
    # plt.ylabel('Returns (%)')
    # plt.legend(['In-sample CE','Out-sample CE'])
    # plt.title('Cross validation of CE\n%s' % str(u))
    # plt.show()
