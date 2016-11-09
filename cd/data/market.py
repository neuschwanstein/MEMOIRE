import datetime as dt
import os

import quandl
import pandas as pd

filename = {
    'r': os.path.join(os.path.dirname(__file__),'market/market%d.csv'),
    'vol': os.path.join(os.path.dirname(__file__),'market/vol%d.csv')
}


def make_r(year):
    start_date = dt.date(year-1,11,1)
    end_date = dt.date(year+1,2,1)

    market = quandl.get('YAHOO/INDEX_GSPC',start_date=start_date,end_date=end_date)
    market.columns = [col_name.lower() for col_name in market.columns]
    market = market.rename(columns={'adjusted close':'price'})
    market.price = market.price/market.price.values[0]
    market['r'] = (market.close - market.open)/market.open
    market = market[['r']]
    market.index.name = 'time'

    return market


def make_vol(year):
    r = make_r(year)
    index = r.index.sort_values()
    start_date = index.min()
    end_date = index.max()
    vol = quandl.get('CBOE/VIX',start_date=start_date,end_date=end_date)
    vol = vol[['VIX High']]
    vol.columns = ['f_vix']
    vol = r[[]].join(vol,how='left')

    return vol


# There must be another way...
def make(year,what):
    if what is 'r':
        return make_r(year)
    elif what is 'vol':
        return make_vol(year)


def save(value,year,what):
    value.to_csv(filename[what] % year,encoding='utf-8')


def load(*years,what):
    try:
        years = range(years[0],years[1]+1)
    except IndexError:
        years = [years[0]]

    def load(year):
        try:
            value = pd.read_csv(filename[what] % year,parse_dates=['time'])
            value = value.set_index('time')
        except OSError:
            value = make(year,what)
            save(value,year,what)
        return value

    values = [load(y) for y in years]
    values = pd.concat(values,axis=0)
    values = values.drop_duplicates()

    return values
