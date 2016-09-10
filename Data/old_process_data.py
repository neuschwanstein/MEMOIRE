from os.path import isfile
import datetime as dt
import itertools

from sklearn import preprocessing as pp
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import pytz
import requests
import quandl

import model.utility as ut
import model.problem as pr

quandl.ApiConfig.api_key = 'TFPsUSNkbZiK8TgJJ_qa'

_get_samples


def get_market(beg_date,end_date):
    st_format = '%Y-%m-%d'
    filename = 'sp500/sp500_%s_%s.csv' % (beg_date.strftime(st_format),
                                          end_date.strftime(st_format))
    if not isfile(filename):
        url = 'http://real-chart.finance.yahoo.com/table.csv?s=%%5EGSPC&a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&g=d&ignore=.csv'
        url = url % (beg_date.month-1,beg_date.day,beg_date.year,end_date.month-1,end_date.day,end_date.year)
        csv = requests.get(url).text
        with open(filename,'w') as csv_file:
            csv_file.write(csv)

    market = pd.read_csv(filename)
    market = market.rename(columns={'Date':'date'})

    ny_tz = pytz.timezone('America/New_York')
    op_time = dt.time(hour=9)
    market.date = market.date.apply(lambda d: ny_tz.localize(dt.datetime.combine(pd.to_datetime(d),op_time)))
    market['r'] = (market['Close']-market['Open'])/market['Open']

    return market[['date','r']]


def get_articles():
    con_st = 'postgresql://localhost/TBM'
    articles = pd.read_sql_table('articles3',columns=['id','date'],con=con_st)
    articles.date = articles.date.apply(lambda d: pytz.utc.localize(d))
    return articles


def aggregate_data(data):
    return np.mean(data,axis=0)  # TODO try different methods here!


def get_samples(articles=None,market=None,model=None):
    articles = articles or get_articles()
    market = market or get_market(beg_date=min(articles.date),end_date=max(articles.date))
    model = Doc2Vec.load('300model')

    n,p = len(market),len(model.docvecs[0])
    d2v = np.empty((n,p))

    articles = articles.sort_values(by='date',ascending=True)
    market = market.sort_values(by='date',ascending=True)

    ids = list()
    it_art = articles.itertuples()
    article = next(it_art)
    for i,record in enumerate(market.itertuples()):
        ids = list()
        while article.date < record.date:
            ids.append(str(article.id))
            try:
                article = next(it_art)
            except StopIteration:
                break
        datum = aggregate_data(model.docvecs[ids])
        d2v[i] = datum

    d2v = pd.DataFrame(d2v)
    samples = pd.concat([market,d2v],axis=1)

    d2v_is = ['d2v_'+str(i) for i in range(1,p+1)]
    d2v_cols = [('d2v',d2v_i) for d2v_i in d2v_is]
    market_cols = [(col,) for col in market.columns]
    samples_cols = pd.MultiIndex.from_tuples(market_cols + d2v_cols)

    samples.columns = samples_cols
    return samples


def preprocess_samples(train,test):
    train_d2v = train['d2v'].values
    scaler = pp.StandardScaler().fit(train_d2v)
    train_d2v = scaler.transform(train_d2v)
    train['d2v'] = train_d2v
    test['d2v'] = scaler.transform(test['d2v'].values)
    train[('d2v','bias')] = np.ones(len(train))
    test[('d2v','bias')] = np.ones(len(test))
    return train,test


def extract_data(samples):
    X = samples.d2v.values
    r = samples.r.values.flatten()
    return X,r


def get_CE(train,test,λ,u):
    train,test = preprocess_samples(train,test)
    X_train,r_train = extract_data(train)
    X_test,r_test = extract_data(test)

    problem = get_solved_problem(X_train,r_train,λ,u)
    insample_CE = problem.insample_CE()
    outsample_CE = problem.outsample_CE(X_test,r_test)
    return insample_CE,outsample_CE


def get_solved_problem(X,r,λ,u):
    problem = pr.Problem(X,r,λ,u)
    problem.solve()
    return problem


def get_train_test(samples,shuffle=False):
    if shuffle:
        samples = samples.sample(frac=1)
    train_sz = int(0.8*len(samples))
    train,test = samples[:train_sz],samples[train_sz:]
    return train,test


def cross_validate(samples):
    train,test = get_train_test(samples,shuffle=True)

    β = 0.9
    r_threshold = np.percentile(samples.r,95)
    u = ut.LinearPlateauUtility(β,r_threshold)

    results = [get_CE(train,test,λ,u) for λ in np.linspace(0,0.1,10)]
    return results


def prices_to_rates(ps):
    rates = [(clos-opn)/opn for opn,clos in zip(ps[:-1],ps[1:])]
    return rates


def rates_to_prices(rs):
    rs = [1]+list(rs)

    def rate_to_price(opn,r):
        return opn*(r+1)
    prices = list(itertools.accumulate(rs,rate_to_price))
    return prices


def exp_rates_to_prices(rs):
    rs = [1]+list(rs)
    ps = list(itertools.accumulate(rs,lambda opn,clos: opn*np.exp(clos)))
    return ps


def exp_prices_to_rates(ps):
    rates = [np.log(clos/opn) for opn,clos in zip(ps[:-1],ps[1:])]
    return rates


def get_timeseries(samples,q):
    market = samples.sort_values(by='date',ascending=True)
    rs = market.r.values.flatten()
    market.market_value = rates_to_prices(rs)[1:]

    d2v = market.d2v.values
    alg_p = d2v@q
    alg_rs = alg_p*rs
    alg_value = rates_to_prices(alg_rs)[1:]
    market.alg_value = alg_value
    return market
