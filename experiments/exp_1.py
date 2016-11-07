'''The point of this experiment is to try if we can pick any signal of news occuring
during the trading session (marked as during(True)), that is if we train upon a test set
of many years we can obtain a positive expected utility on the test set.

Parameters:
 - u = LinearPlateauUtility()
'''

from cd.datasets import newsmarket as mkt
from cd.data import analysis2 as nly
import cd.model.utility as ut

if __name__ == '__main__':
    newsmarket = mkt.load_all()
    newsmarket = nly.add_bias(newsmarket,0.1)
    newsmarket = newsmarket.during(True)
    train,test = nly.split_and_normalize(newsmarket)

    λ = 1e-4
    u = ut.LinearPlateauUtility(0.1,0.08)

    q = nly.solve(train,u,λ)

    in_ce = nly.CE(train,q,u,λ)
    out_ce = nly.CE(test,q,u,λ)
