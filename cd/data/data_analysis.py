import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import cd.model.utility as ut
import cd.model.problem as pr


def extract_data(data):
    X = data.X.values
    r = data.r.values.flatten()
    return X,r


def get_train_test(samples,shuffle=False):
    if shuffle:
        samples = samples.sample(frac=1)
    train_sz = int(0.8*len(samples))
    train,test = samples[:train_sz],samples[train_sz:]
    X_train,X_test = train.X.values,test.X.values
    X_mean = np.mean(X_train,axis=0)
    X_std = np.std(X_train,axis=0)
    X_train = (X_train - X_mean)/X_std
    X_test = (X_test - X_mean)/X_std
    train.X = X_train
    test.X = X_test
    # Either have this
    # train.r -= train.r.mean()
    # test.r -= test.r.mean()
    # Or that
    # b = 0.01
    # train[('X','bias')] = b*np.ones(len(train))
    # test[('X','bias')] = b*np.ones(len(test))
    # Or that ;) See statistical learning with sparsity, p.9 for details
    bias = np.mean(train.r.values.flatten())
    train.r -= bias
    train = train.fillna(0)
    test = test.fillna(0)
    return train,bias,test


def get_ces(train,test,u,comps,bias):
    result = np.empty((len(comps),2))
    X,r = extract_data(train)
    X_test,r_test = extract_data(test)
    for i,comp in enumerate(comps):
        p = pr.Problem(X,r,comp,u)
        p.solve()
        in_CE = p.insample_CE()
        out_CE = p.outsample_CE(X_test,r_test,bias)
        result[i,:] = [in_CE,out_CE]
    return result


def get_ce(train,test,u,comp):
    return get_ces(train,test,u,[comp])[0,1]


def get_q0(train,u,comp):
    X,r = extract_data(train)
    p = pr.Problem(X,r,comp,u)
    p.solve()
    q0 = p.q
    return q0


def get_comp0(comps,ces):
    out_ces = ces[:,1]
    index = np.argmax(out_ces)
    return comps[index]


def cross_val_comp(train,u,comps,k_fold=5,shuffle=False):
    if shuffle:
        train = train.sample(frac=1)
    folds = np.array_split(train,k_fold)
    comp0s = np.empty(k_fold)
    for i in range(k_fold):
        test = folds[i]
        train = pd.concat(folds[:i] + folds[i+1:])
        ces = get_CEs(train,test,u,comps)
        comp0 = get_comp0(comps,ces)
        comp0s[i] = comp0
    # return comp0s
    return np.mean(comp0s)


def plot_price_series(test,q,ordered=False):
    X,r = extract_data(test)
    r_alg = r*(X@q)
    pr = np.cumprod(1+r)
    pr_alg = np.cumprod(1+r_alg)
    ps = np.array([pr,pr_alg]).T
    plt.plot(ps)
    plt.legend(['Market price','Algorithmic price'])
    plt.show()


def plot_cross_val(comps,CEs,u=None,log=True):
    plt.plot(comps,CEs)
    if log:
        plt.xscale('log')
    plt.axis(xmax=max(comps),xmin=min(comps))
    plt.xlabel('$\lambda$')
    plt.legend(['In-sample CE','Out-sample CE'])
    if u is not None:
        plt.title('Cross validation of CE\n%s' % str(u))
    plt.show()


def plot_decision(test,q):
    l = np.arange(len(q))
    plt.bar(l,q)
    plt.xticks(l,test.X.columns,rotation='vertical')
    plt.show()


def test_price_series(test,q,ordered=False):
    if ordered:
        trim = test[['r']]
        trim.columns = ['r']
        trim.p = test.X@q
        alg_r = 'Algorithmic Return'
        trim[alg_r] = trim.p*trim.r
        [mp,ap] = ['Market Performance','Algorithmic Performance']
        trim[mp] = (1+trim.r).cumprod()
        trim[ap] = (1+trim[alg_r]).cumprod()
        trim[mp] /= trim[mp][0]
        trim[ap] /= trim[ap][0]
        # trim = trim.shift(1)
        # trim[[mp,ap]] = trim[[mp,ap]].fillna(value=1)
        trim[[mp,ap]].plot()
        plt.show()
        return trim
        # trim = test[['r']]
        # trim.real_price = (1 + trim.r).cumprod()
        # initial_val = trim.real_price.values[0]
        # trim.real_price /= initial_val
        # trim.alg_price = (1 + (trim.X@q)*
    else:
        X_test = test.X.values
        r_test = test.r.values.flatten()
        r = r_test*(X_test@q)
        # r = (r_test+0.5)*(X_test@q)
        alg_price = np.cumprod(1+r)
        act_price = np.cumprod(1+r_test)
        plt.plot(list(zip(act_price,alg_price)))
        plt.axis(xmax=len(r))
        plt.xlabel('$t$')
        plt.ylabel('Portfolio value')
        plt.title('Test set performance')
        plt.legend(['Test performance','Algorithmic performance'])
        plt.show()


if __name__ == '__main__':
    # comp = 0.031622776601683791 has been not bad in the past.
    # comps = np.linspace(1e-4,0.1,15)
    comps = np.logspace(-10,-1.5,20)
    u = ut.LinearPlateauUtility(1,0.1)
    CEs = get_CEs(train,test,u,comps)
    comp0 = get_optimal_comp(comps,CEs)
    p0 = td.get_optimal_decision(train.X.values,train.r.values.flatten(),comp0,u)
    plot_cross_val(comps,CEs,u)
