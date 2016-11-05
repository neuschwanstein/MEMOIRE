'''Implement Erick's Idea'''

import numpy as np
import matplotlib.pyplot as plt

from cd.datasets import newsmarket as mkt
from cd.data import analysis2 as nly
import cd.model.utility as ut
import cd.model.problem as pr

def plot_2dims(qs,i=0,j=1):
    plt.scatter(qs[:,0],qs[:,1])
    plt.axhline(y=0)
    plt.axvline(x=0)
    plt.show()


if __name__ == '__main__':
    newsmarket = mkt.load_all()
    newsmarket = nly.add_bias(newsmarket)
    newsmarket = newsmarket.during(True)
    train,text = nly.split_and_normalize(newsmarket)

    λ = 0
    u = ut.LinearPlateauUtility(0,0.08)
    def get_q():
        subtrain = train.sample(100,replace=True)
        p = pr.Problem(subtrain.X.values,subtrain.r.values,λ,u)
        p.solve()
        return p.q

    qs = [get_q() for _ in range(400)]
    qs = np.array(qs)

    plot_2dims(qs)
