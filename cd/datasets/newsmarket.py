import datetime as dt
import os
import re

import numpy as np
import pandas as pd
import gensim.models.word2vec as w2v

from . import market as mkt
from . import news as ns

pattern_process = re.compile('[^a-zA-Z]+')
vec_length = 300
if 'gmodel' not in locals():
    gmodel = None

filename = os.path.join(os.path.dirname(__file__),'newsmarket/newsmarket%d.csv')


class NewsMarket(pd.DataFrame):

    @property
    def _constructor(self):
        return NewsMarket

    @property
    def X(self):
        return self.filter(regex='^d2v_')

    def during(self,bool):
        return self.xs(bool,level='during')

    def __getitem__(self,key):
        if key is 'X':
            return self.X
        else:
            return super().__getitem__(key)

    def __setitem__(self,att,val):
        if att == 'X':
            cols = self.columns[self.columns.str.contains('d2v_')]
            try:
                self.loc[:,cols] = val.values
            except:
                self.loc[:,cols] = val
        else:
            super().__setitem__(att,val)


def init_gmodel():
    global gmodel
    if gmodel is None:
        filename = os.path.join(os.path.dirname(__file__),
                                'w2v/GoogleNews-vectors-negative300.bin')
        gmodel = w2v.Word2Vec.load_word2vec_format(filename,binary=True)


def to_list_of_words(s):
    ss = [w.lower() for w in pattern_process.sub(' ',s).split() if len(w) > 1]
    return ss


def mean(lst):
    global gmodel

    def vector(s):
        try:
            return gmodel[s]
        except KeyError:
            return np.zeros(300)
    return np.mean([vector(s) for s in lst],axis=0)


def remove_duplicates(ds):
    dups = ds.duplicated('content')
    return ds[~dups]


def process_vectors(newsmarket):
    # First clean up
    vectors = newsmarket['content']
    vectors = vectors[~vectors.isnull()]
    vectors = vectors.apply(to_list_of_words)
    empty_vectors = vectors.apply(len) == 0
    vectors = vectors[~empty_vectors]
    vectors = vectors[~vectors.isnull()]

    # Mean of the words
    vectors = vectors.apply(mean)

    # Then convert it to proper newsmarket
    index = vectors.index
    vectors = np.vstack(vectors)
    vectors = pd.DataFrame(vectors)
    cols = ['d2v_%d' % i for i in range(1,vec_length+1)]
    vectors.columns = cols
    vectors.index = index
    return vectors


def collapse_time(reference,newsmarket):
    morning = dt.time(9,30)
    afternoon = dt.time(16,0)
    morning_of = lambda d: dt.datetime.combine(d,morning)
    afternoon_of = lambda d: dt.datetime.combine(d,afternoon)

    if reference.min() > morning_of(newsmarket['time'].min()) or \
       reference.max() < afternoon_of(newsmarket['time'].max()):
        raise Exception('The reference series need to overlap the newsmarket time column')

    reference = reference.sort_values()
    reference = reference.map(pd.Timestamp.date)
    newsmarket = newsmarket.sort_values('time')

    reference = iter(reference)
    events = iter(newsmarket['time'])

    collapsed_times = []
    during = []

    # First seek when the actual reference date starts
    event_time = next(events)
    while True:
        ref_date = next(reference)
        if event_time < morning_of(ref_date):
            collapsed_times.append(ref_date)
            during.append(False)
            break

    # Then map to every event a corresponding reference date Rules:
    # 1. If it happens during the trading session (9:30am to 4pm) then
    # map it to the noon of the trading date.
    # 2. If it happens after trading session, map it to the morning of the
    # affected trading session (ie. the next morning)
    for event_time in events:
        if event_time < morning_of(ref_date):
            collapsed_times.append(ref_date)
            during.append(False)
        elif event_time >= morning_of(ref_date) and event_time < afternoon_of(ref_date):
            collapsed_times.append(ref_date)
            during.append(True)
        else:
            while True:
                next_ref_date = next(reference)
                if event_time < morning_of(next_ref_date):
                    collapsed_times.append(next_ref_date)
                    during.append(False)
                    break
            ref_date = next_ref_date

    # Update time column of the newsmarket with results
    newsmarket['time'] = pd.to_datetime(collapsed_times)
    newsmarket['during'] = during
    newsmarket = newsmarket.set_index('time')
    return newsmarket


def make(year):
    init_gmodel()

    # Load newsmarket and translate their content to vectorial representation
    newsmarket = ns.load(year)
    newsmarket = remove_duplicates(newsmarket)
    vectors = process_vectors(newsmarket)
    newsmarket = newsmarket.join(vectors,how='right')
    newsmarket = newsmarket.drop('content',axis=1)

    # Collapse dates to either have them during the trading session or
    # in the after hours.
    market = mkt.load(year,year)
    reference = market.index
    newsmarket = collapse_time(reference,newsmarket)

    # Aggregate (Mean) each trading and after hours sessions newsmarket vector representation
    newsmarket = newsmarket.groupby([newsmarket.index,newsmarket.during]).aggregate(np.mean)

    # Then add a return column
    newsmarket = newsmarket.join(market['r'],how='left')
    return newsmarket


def save(newsmarket,year):
    newsmarket.to_csv(filename % year,encoding='utf-8')


def load(year):
    newsmarket = pd.read_csv(filename % year,parse_dates=['time'])
    newsmarket = newsmarket.set_index(['time','during'])
    return NewsMarket(newsmarket)


def load_all():
    years = range(2007,2016)
    newsmarket = [load(y) for y in years]
    newsmarket = pd.concat(newsmarket,axis=0)
    newsmarket = newsmarket.drop_duplicates(keep='last')
    return newsmarket


def make_all():
    init_gmodel()
    for year in range(2007,2016):
        pnews = make(year)
        save(pnews,year)


if __name__ == '__main__':
    make_all()
