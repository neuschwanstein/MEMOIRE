import datetime as dt
import itertools

from sklearn import preprocessing as pp
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import pytz
import quandl

import model.utility as ut
import model.problem as pr

quandl.ApiConfig.api_key = 'TFPsUSNkbZiK8TgJJ_qa'
day_shift = 5


def get_market(start_date,end_date):
    # Start a bit early
    start_date = start_date + dt.timedelta(days=-day_shift)

    # Get absolute prices
    sp500 = quandl.get('YAHOO/INDEX_GSPC',start_date=start_date,end_date=end_date)
    sp500.columns = [col_name.lower() for col_name in sp500.columns]
    sp500 = sp500.rename(columns={'adjusted close':'price'})
    sp500.price = sp500.price/sp500.price.values[0]
    sp500['r'] = (sp500.close - sp500.open)/sp500.open
    sp500 = sp500[['volume','price','r']]
    return sp500


def get_fnc(start_date,end_date):
    # Start again a bit early
    start_date = start_date + dt.timedelta(days=-day_shift)

    def get_dataset(key,feature_name):
        ds = quandl.get(key,start_date=start_date,end_date=end_date)
        ds.columns = [col.lower().replace(' ','_') for col in ds.columns]
        return ds

    vix = get_dataset('CBOE/VIX','vix')[['vix_close']]
    vix_of_vix = get_dataset('CBOE/VVIX','vvix')

    fnc = vix.join([vix_of_vix])
    return fnc


def get_d2v(articles,dates):

    def in_newyork(date):
        ny_tz = pytz.timezone('America/New_York')
        op_time = dt.time(hour=9)
        date_at_9am = dt.datetime.combine(date,op_time)
        date_in_newyork = ny_tz.localize(date_at_9am)
        return date_in_newyork

    dates = dates.sort_values(ascending=True)
    ny_dates = [in_newyork(date) for date in dates]
    articles = articles.sort_index(ascending=True)
    model = Doc2Vec.load('300model')

    it_art = articles.itertuples()
    article = next(it_art)
    article_date = article.Index
    for i,date in enumerate(ny_dates):
        if article_date < date:
            break

    dates = dates[i:]
    ny_dates = ny_dates[i:]
    n,p = len(dates),len(model.docvecs[0])
    d2v = np.empty((n,p))

    for i,date in enumerate(ny_dates):
        ids = list()
        while article_date < date:
            ids.append(str(article.id))
            try:
                article = next(it_art)
                article_date = article.Index
            except StopIteration:
                break
        datum = aggregate_data(model.docvecs[ids])
        d2v[i] = datum

    d2v = pd.DataFrame(d2v)
    d2v['date'] = dates
    d2v = d2v.set_index('date')
    d2v.columns = ['d2v_'+str(i) for i in range(1,p+1)]
    return d2v


def get_articles():
    con_st = 'postgresql://localhost/TBM'
    articles = pd.read_sql_table('articles3',columns=['id','date'],con=con_st)

    def apply_utc(date):
        return pytz.utc.localize(date)
    articles.date = articles.date.apply(apply_utc)
    articles = articles.set_index('date').sort_index()
    return articles


def add_timelag(ds,day_shift):

    def process_shift(ds,i):
        shift = ds.shift(i)
        shift.columns = [col+'_minus_'+str(i) for col in shift.columns]
        return shift

    shifts = [process_shift(ds,i) for i in range(1,day_shift+1)]
    return pd.concat(shifts,axis=1)


def build_dataset(market,fnc,d2v):
    market_response = market[['r','price']]
    market_response.columns = pd.MultiIndex.from_tuples([('r',None),('price',None)])

    volume = market.volume
    fnc = fnc.join(volume)
    market_features = fnc.join(d2v)
    cols_fnc = list(itertools.product(('fnc',),fnc.columns))
    cols_d2v = list(itertools.product(('d2v',),d2v.columns))
    cols = cols_fnc+cols_d2v
    market_features.columns = pd.MultiIndex.from_tuples(cols)
    ds = market_response.join(market_features)
    return ds


def aggregate_data(data):
    return np.mean(data,axis=0)  # TODO try different methods here!


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


if (__name__ == '__main__'):
    switch = False
    if switch:
        articles = get_articles()
        start_date = min(articles.index)
        end_date = max(articles.index)
        market = get_market(start_date,end_date)  # Contains price, returns and volume
        fnc = get_fnc(start_date,end_date)        # Contains various features (eg. vix)
        d2v = get_d2v(articles,market.index)  # Aggregate articles based on market dates
        
