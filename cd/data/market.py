import datetime as dt
import os

import quandl
import pandas as pd

filename = os.path.join(os.path.dirname(__file__),'market/market%d-%d.csv')


def make(start_year,end_year):
    start_date = dt.date(start_year-1,11,1)
    end_date = dt.date(end_year+1,2,1)
    market = quandl.get('YAHOO/INDEX_GSPC',start_date=start_date,end_date=end_date)
    market.columns = [col_name.lower() for col_name in market.columns]
    market = market.rename(columns={'adjusted close':'price'})
    market.price = market.price/market.price.values[0]
    market['r'] = (market.close - market.open)/market.open
    market = market[['volume','price','r']]
    market.index.name = 'time'
    return market


def save(market,start_year,end_year):
    market.to_csv(filename % (start_year,end_year),encoding='utf-8')


def load(start_year,end_year):
    try:
        market = pd.read_csv(filename % (start_year,end_year),parse_dates=['time'])
        market = market.set_index('time')
    except (FileNotFoundError,OSError):
        market = make(start_year,end_year)
        save(market,start_year,end_year)
    return market


def make_vol(r):
    index = r.index.sort_values()
    start_date = index.min()
    end_date = index.max()
    vol = quandl.get('CBOE/VIX',start_date=start_date,end_date=end_date)
    vol = vol[['VIX High']]
    vol.columns = ['f_vix']
    vol = r[[]].join(vol,how='left')
    return vol


def save_vol(vol,start_year,end_year):
    filename = os.path.join(os.path.dirname(__file__),'market/vol%d-%d.csv')
    vol.to_csv(filename % (start_year,end_year),encoding='utf-8')


def load_vol(start_year,end_year):
    filename = os.path.join(os.path.dirname(__file__),'market/vol%d-%d.csv')
    try:
        vol = pd.read_csv(filename % (start_year,end_year),encoding='utf-8')
        vol = vol.set_index('time')
    except (FileNotFoundError,OSError):
        r = load(start_year,end_year)
        vol = make_vol(r)
        save_vol(vol,start_year,end_year)
    return vol
