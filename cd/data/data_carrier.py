import datetime as dt
import requests as rq
from collections import namedtuple
from multiprocessing import Pool
from functools import partial

from bs4 import BeautifulSoup as bs
import pandas as pd


class Daterange(object):

    def __init__(self,start,end):
        self.date = start
        self.end = end

    def __iter__(self):
        return self

    def __next__(self):
        if self.date <= self.end:
            rez = self.date
            self.date += dt.timedelta(days=1)
            return rez
        else:
            raise StopIteration()


# News = namedtuple('News','time content href')
News = namedtuple('News','time content')


def get_url(date):
    return 'http://www.reuters.com/resources/archive/us/%s.html' % \
        date.strftime('%Y%m%d')


def parse_website(date,**kwargs):
    if 'verbose' in kwargs and kwargs['verbose']:
        print(date)
    url = get_url(date)
    r = rq.get(url)
    if r.status_code is not 200:
        # Do something
        pass
    else:
        soup = bs(r.text,'html.parser')
        news_divs = soup.find_all('div',{'class':'headlineMed'})
        results = []
        for n in news_divs:
            try:
                text = n.a.text
                # href = n.a['href']
                time = n.text.replace(n.a.text,'').replace('\xa0','')
                time = time[:-4]  # Remove timezone (all set in NYC)
                time,am_or_pm = time[:-2],time[-2:]
                h,m = time[:2],time[3:]
                h,m = int(h),int(m)
                if am_or_pm == 'PM' and h is not 12:
                    h += 12

                d = dt.datetime(year=date.year,month=date.month,day=date.day,
                                hour=h,minute=m)
                # news = News(time=d,content=text,href=href)
                news = News(time=d,content=text)
                results.append(news)
            except:
                continue
        results = pd.DataFrame(results)
        results = results.set_index('time')
        return results


def get_news(start_date,end_date,**kwargs):
    if 'parallel' in kwargs and kwargs['parallel']:
        p = Pool(10)
        results = p.map(partial(parse_website,**kwargs),
                        Daterange(start_date,end_date))
    else:
        results = [parse_website(date,**kwargs)
                   for date in Daterange(start_date,end_date)]
    results = pd.concat(results)
    return results


if __name__ == '__main__':
    for y in range(2007,2016):
        start_date = dt.date(y,1,1)
        end_date = dt.date(y,12,31)
        results = get_news(start_date,end_date,parallel=True,verbose=True)
        filename = 'news_%s_%s.csv' % (start_date,end_date)
        filename = 'dataset/' + filename
        results.to_csv(filename,encoding='utf-8')
