from collections import namedtuple
from os.path import isfile
import datetime as dt

import requests
from csv import reader as csv_reader
from dateutil.parser import parse as date_parse
import numpy as np
import pytz

from helper.collections import Namedtuples

ny_tz = pytz.timezone('America/New_York')
beg_trading_time = dt.time(hour=16,microsecond=1,tzinfo=ny_tz) # Opening and closing time of NY stock exchange.
end_trading_time = dt.time(hour=16,tzinfo=ny_tz)


def get_sp500_records(beg_date='2007-01-01',end_date='2015-12-31'):
    filename = 'sp500/sp500_%s_%s.csv' % (beg_date,end_date)
    if not isfile(filename):
        url = 'http://real-chart.finance.yahoo.com/table.csv?s=%%5EGSPC&a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&g=d&ignore=.csv'
        try:
            url = url % (beg_date.month-1,beg_date.day,beg_date.year,end_date.month-1,end_date.day,end_date.year)
        except AttributeError:
            beg_date = date_parse(beg_date).date()
            end_date = date_parse(end_date).date()
            url = url % (beg_date.month-1,beg_date.day,beg_date.year,end_date.month-1,end_date.day,end_date.year)
        csv = requests.get(url).text
        with open(filename,'w') as csv_file:
            csv_file.write(csv)

    with open(filename,'r') as csv_file:
        rows = csv_reader(csv_file)
        headers = next(rows)
        headers = [header.lower().replace(' ','_') for header in headers]
        SP500 = namedtuple('SP500',headers)
        SP500s = [SP500(*row) for row in rows]

    SP500s = sorted(SP500s,key=lambda el: date_parse(el.date).date())

    Record = namedtuple('Record',['beg_date','end_date','logreturn'])
    ny_tz = pytz.timezone('America/New_York')
    beg_time = dt.time(hour=16,microsecond=1,tzinfo=ny_tz) # Opening and closing time of NY stock exchange.
    end_time = dt.time(hour=16,tzinfo=ny_tz)
    def create_record(beg,end):
        beg_date = dt.datetime.combine(date_parse(beg.date),beg_time)
        end_date = dt.datetime.combine(date_parse(end.date),end_time)
        logreturn = np.log(float(end.adj_close)/float(beg.adj_close))
        return Record(beg_date,end_date,logreturn)
    records = [create_record(beg,end) for beg,end in zip(SP500s[:-1],SP500s[1:])]
    return records
