import re
from math import isnan

import datetime as dt
import itertools

# from sklearn import preprocessing as pp
from gensim.models.doc2vec import Doc2Vec
import numpy as np
import pandas as pd
import pytz
import quandl


quandl.ApiConfig.api_key = 'TFPsUSNkbZiK8TgJJ_qa'


def get_market(start_date,end_date):
    sp500 = quandl.get('YAHOO/INDEX_GSPC',start_date=start_date,end_date=end_date)
    sp500.columns = [col_name.lower() for col_name in sp500.columns]
    sp500 = sp500.rename(columns={'adjusted close':'price'})
    sp500.price = sp500.price/sp500.price.values[0]
    sp500['r'] = (sp500.close - sp500.open)/sp500.open
    sp500 = sp500[['volume','price','r']]
    sp500.index.name = 'time'
    return sp500


def get_samples(market):
    lag = 1
    market_response = market[['r','price']]
    market_response.columns = pd.MultiIndex.from_tuples([('r',None),('price',None)])
    quandl_features = get_quandl_features(market)
    quandl_features = quandl_features.join(market.volume,how='outer')
    # Only retain valid dates
    quandl_features = market_response.join(quandl_features,how='left')[quandl_features.columns]
    lagged_features = add_timelag(quandl_features,lag)
    cols = list(itertools.product(('X',),lagged_features.columns))
    lagged_features.columns = pd.MultiIndex.from_tuples(cols)
    market_response = market_response[lag:]
    result = market_response.join(lagged_features,how='left')
    # Remove rows without information
    # result = result.dropna(axis=0,how='any')
    return result


def get_quandl_dataset(market,code,columns=None,buffer_days=10):
    '''Queries a quandl database, with some buffer in order to introduce lag.'''
    start_date = market.index.min()
    end_date = market.index.max()
    if buffer_days is not None:
        start_date += dt.timedelta(days=-buffer_days)

    ds = quandl.get(code,start_date=start_date,end_date=end_date)
    db_name = re.search('(?<=/).*',code).group(0).lower()
    ds.columns = [col.lower().replace(' ','_') for col in ds.columns]
    columns = ds.columns if columns is None else columns
    ds = market[[]].join(ds,how='outer')[columns]
    # ds = market[[]].join(ds,how='outer')
    ds.columns = [db_name+'_'+col_name for col_name in ds.columns]
    ds = ds.sort_index(ascending=True)

    # Add new column to indicate if the feature is 'old' news or not.
    # Not necessary if updated daily.
    is_new_trivial = quandl.Dataset(code)['frequency'] == 'daily'
    if not is_new_trivial:
        is_new = ds.apply(lambda row: 0 if all(isnan(c) for c in row) else 1,axis=1)

    ds = ds.fillna(method='pad')
    if not is_new_trivial:
        ds[db_name+'_is_new'] = is_new
    return ds


def get_quandl_features(market):
    '''Get features from quandl api and join them.'''
    datasets = [
        ('CBOE/VIX',['vix_close']),
        ('CBOE/VVIX',),
        ('AAII/AAII_SENTIMENT',['bullish','neutral','bearish']),
        # U.S. Weekly Leading Index
        ('ECRI/USLEADING',),
        # Housing Starts: Total: New Privately Owned Housing Units Started
        ('FRED/HOUST',),
        # 10-Year Treasury Constant Maturity Rate
        ('FRED/DGS10',),
        # Crude Oil Futures, Continuous Contract #1
        ('CHRIS/CME_CL1',),
        # Gold price
        ('BUNDESBANK/BBK01_WT5511',),
        # Non-Manufacturing Backlog of Orders Index
        ('ISM/MAN_BACKLOG',['%_greater','%_same','%_less','net']),
        # Euro Emerging Markets Corporate Bond Index (Yield)
        ('ML/EEMCBI',),
        # US Corporate Bonds Total Return Index
        ('ML/TRI',),
        # Stock Market Confidence Indices
        ('YALE/US_CONF_INDEX_VAL_INDIV',['index_value'])
    ]
    queried_datasets = [get_quandl_dataset(market,*ds) for ds in datasets]
    features = pd.concat(queried_datasets,axis=1,join='outer')
    return features


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


if __name__ == '__main__':
    start_date = dt.date(2005,1,1)
    end_date = dt.date(2015,1,1)
    market = get_market(start_date,end_date)
    samples = get_samples(market)
    print('Obtained %d samples.' % len(samples))
