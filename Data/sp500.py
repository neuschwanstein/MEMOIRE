from collections import namedtuple
from os.path import isfile

import requests
from csv import reader as csv_reader
from dateutil.parser import parse as date_parse

from helper.collections import Namedtuples

def get_csv(d_beg='2007-01-01',d_end='2012-12-31'):
    d_beg = date_parse(d_beg)
    d_end = date_parse(d_end)
    filename = 'sp500/sp500_%s_%s.csv' % (d_beg.date(),d_end.date())

    if not isfile(filename):
        url = 'http://real-chart.finance.yahoo.com/table.csv?s=%%5EGSPC&a=%d&b=%d&c=%d&d=%d&e=%d&f=%d&g=d&ignore=.csv'
        url = url % (d_beg.month-1,d_beg.day,d_beg.year,d_end.month-1,d_end.day,d_end.year)
        csv = requests.get(url).text
        with open(filename,'w') as csv_file:
            csv_file.write(csv)

    with open(filename,'r') as csv_file:
        rows = csv_reader(csv_file)
        headers = next(rows)
        headers = [header.lower().replace(' ','_') for header in headers]
        SP500 = namedtuple('SP500',headers)

        SP500s = Namedtuples([SP500(*row) for row in rows])
        SP500s.dates = [date_parse(r.date).date() for r in SP500s]
        SP500s.adj_closes = [float(r.adj_close) for r in SP500s]
        SP500s.highs = [float(r.high) for r in SP500s]
        SP500s.opens = [float(r.open) for r in SP500s]
        SP500s.lows = [float(r.low) for r in SP500s]
        SP500s.closes = [float(r.close) for r in SP500s]
        SP500s.volumes = [float(r.volume) for r in SP500s]
        return SP500s
        # for record in SP500s:
        #     date = date_parse(record.date).date()
        #     # open = float(record.open)
        #     high = float(record.high)
        #     low = float(record.low)
        #     close = float(record.close)
        #     volume = float(record.volume)
        #     adj_close = float(record.adj_close)
        #     record._replace(date=date,open=open,high=high,low=low,close=close,volume=volume,adj_close=adj_close)
        # return SP500s
        # # SP500s.dates = [date_parse(d).date() for d in SP500s.dates]
        # # return SP500s

