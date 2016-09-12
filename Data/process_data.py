from math import isnan

import datetime as dt
import itertools

from sklearn import preprocessing as pp
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import pytz
import quandl


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


def get_samples(market):
    market_response = market[['r','price']]
    market_response.columns = pd.MultiIndex.from_tuples([('r',None),('price',None)])
    quandl_features = get_quandl_features(min(market.index),max(market.index))
    quandl_features = quandl_features.join(market.volume,how='outer')
    # Only retain valid dates
    quandl_features = market_response.join(quandl_features,how='left')[quandl_features.columns]
    lagged_features = add_timelag(quandl_features,3)
    cols = list(itertools.product(('X',),lagged_features.columns))
    lagged_features.columns = pd.MultiIndex.from_tuples(cols)
    result = market_response.join(lagged_features,how='left')
    # Remove rows without information
    result = result.dropna(axis=0,how='any')
    return result


def get_quandl_dataset(code,market,columns=None,buffer_days=None):
    start_date = market.index.min()
    end_date = market.index.max()
    if buffer_days is not None:
        start_date += dt.timedelta(days=-buffer_days)

    dataset = quandl.get(code,start_date=start_date,end_date=end_date)
    columns = dataset.columns if columns is None else columns
    dataset = market.join(dataset,how='outer')[columns]
    dataset = dataset.sort_index(ascending=True)
    is_new = dataset.apply(lambda row: 0 if all(isnan(c) for c in row) else 1,axis=1)
    dataset = dataset.fillna(method='pad')
    dataset['is_new_'+code[:4]] = is_new
    return dataset


def get_quandl_features(start_date,end_date):
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
        shift.columns = [col+'_m_'+str(i) for col in shift.columns]
        return shift

    shifts = [process_shift(ds,i) for i in range(1,day_shift+1)]
    return pd.concat(shifts,axis=1)


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


if (__name__ == '__main__'):
    switch = False
    if switch:
        articles = get_articles()
        start_date = min(articles.index)
        end_date = max(articles.index)
        market = get_market(start_date,end_date)  # Contains price, returns and volume
        fnc = get_quandl_features(start_date,end_date)        # Contains various features (eg. vix)
        d2v = get_d2v(articles,market.index)  # Aggregate articles based on market dates
