import logging
import datetime as dt

import pandas as pd
import numpy as np
from scipy.stats.stats import pearsonr

import cd.data.process_google_d2v as pgd
import cd.data.helper as he
import cd.data.process_quandl as qu


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def vcorrcoef(X,y):
    Xm = np.reshape(np.mean(X,axis=1),(X.shape[0],1))
    ym = np.mean(y)
    r_num = np.sum((X-Xm)*(y-ym),axis=1)
    r_den = np.sqrt(np.sum((X-Xm)**2,axis=1)*np.sum((y-ym)**2))
    r = r_num/r_den
    return r


# market = qu.get_market(dt.date(2006,10,1),dt.date(2016,2,10))

if __name__ == '__main__':
    if 'news' not in locals():
        years = range(2007,2016)
        news = [pgd.get_news(y) for y in years]
        news = pd.concat(news)
        news = news[~news.index.duplicated(keep='last')]
        news = news.xs(True,level='during')

    r = news.r.values
    X = news.filter(regex='d2v_*').values

    # Normalize data
    r = (r - r.mean())/r.std()
    X = (X - X.mean(axis=0))/X.std(axis=0)

    train_sz = 8/9 * len(X)     # Last year data
    X,X_test = X[:train_sz],X[train_sz:]
    r,r_test = r[:train_sz],r[train_sz:]

    cors = vcorrcoef(X.T,r)
    tops = np.argpartition(cors,-10)[-10:]

    cors_test = vcorrcoef(X_test.T,r_test)
    predicted_tops = cors_test[tops]
    print('Done')

    
    
