import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cd.model.utility as ut
import cd.model.problem as pr


class Samples(object):

    def __init__(self,samples):
        # Dataset wrapper.
        self.samples = samples
        self.X_set = samples.X
        self.X = self.X_set.values
        self.r_set = samples['r'][None]
        self.r = self.r_set.values


class Data(object):

    @staticmethod
    def extract_data(data):
        X = data.X.values
        r = data.r.values.flatten()
        return X,r

    def __init__(self,samples,shuffle=False):
        self.samples = samples
        if shuffle:
            samples = samples.sample(frac=1)

        train_sz = int(0.8*len(samples))
        train,test = samples[:train_sz].copy(),samples[train_sz:].copy()

        mean = train.X.mean()
        std = train.X.std()
        train.X = (train.X-mean)/std
        test.X = (test.X-mean)/std
        bias = train.r.mean()
        train.r -= bias

        train = train.fillna(0)
        test = test.fillna(0)

        self.train = train
        self.test = test
        self.bias = float(bias)

    def get_ces(self,u,comps):
        result = np.empty((len(comps),2))
        X,r = self.extract_data(self.train)
        X_t,r_t = self.extract_data(self.test)
        for i,comp in enumerate(comps):
            p = pr.Problem(X,r,comp,u)
            p.solve()
            in_ce = p.insample_CE()
            out_ce = p.outsample_CE(X_t,r_t,self.bias)
            result[i,:] = [in_ce,out_ce]
        return result

    def get_ce(self,u,comp):
        return self.get_ces(u,[comp])[0]


if __name__ == '__main__':
    data = Data(samples)
    comps = np.logspace(-8,-2,25)
    u = ut.LinearPlateauUtility(1,0.1)
