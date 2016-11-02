import datetime as dt
import os
import re

import numpy as np
import pandas as pd
import gensim.models.word2vec as w2v

import news as ns
import market as mkt

pattern_process = re.compile('[^a-zA-Z]+')
vec_length = 300
if 'gmodel' not in locals():
    gmodel = None

filename = os.path.join(os.path.dirname(__file__),'parsed_news/pnews%d.csv')


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


def process_vectors(news):
    # First clean up
    vectors = news['content']
    vectors = vectors[~vectors.isnull()]
    vectors = vectors.apply(to_list_of_words)
    empty_vectors = vectors.apply(len) == 0
    vectors = vectors[~empty_vectors]
    vectors = vectors[~vectors.isnull()]

    # Mean of the words
    vectors = vectors.apply(mean)

    # Then convert it to proper dataset
    index = vectors.index
    vectors = np.vstack(vectors)
    vectors = pd.DataFrame(vectors)
    cols = ['d2v_%d' % i for i in range(1,vec_length+1)]
    vectors.columns = cols
    vectors.index = index
    return vectors


def collapse_time(reference,news):
    morning = dt.time(9,30)
    afternoon = dt.time(16,0)
    morning_of = lambda d: dt.datetime.combine(d,morning)
    afternoon_of = lambda d: dt.datetime.combine(d,afternoon)

    if reference.min() > morning_of(news['time'].min()) or \
       reference.max() < afternoon_of(news['time'].max()):
        raise Exception('The reference series need to overlap the news time column')

    reference = reference.sort_values()
    reference = reference.map(pd.Timestamp.date)
    news = news.sort_values('time')

    reference = iter(reference)
    events = iter(news['time'])

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

    # Update time column of the news with results
    news['time'] = pd.to_datetime(collapsed_times)
    news['during'] = during
    news = news.set_index('time')
    return news


def make(year):
    init_gmodel()

    # Load news and translate their content to vectorial representation
    news = ns.load(year)
    news = remove_duplicates(news)
    vectors = process_vectors(news)
    news = news.join(vectors,how='right')
    news = news.drop('content',axis=1)

    # Collapse dates to either have them during the trading session or
    # in the after hours.
    market = mkt.load(year,year)
    reference = market.index
    news = collapse_time(reference,news)

    # Aggregate (Mean) each trading and after hours sessions news vector representation
    news = news.groupby([news.index,news.during]).aggregate(np.mean)

    # Then add a return column
    news = news.join(market['r'],how='left')
    return news


def save(news,year):
    news.to_csv(filename % year,encoding='utf-8')


def load(year):
    news = pd.read_csv(filename % year,parse_dates=['time'])
    news = news.set_index(['time','during'])
    return news


def load_all():
    years = range(2007,2016)
    dataset = [load(y) for y in years]
    dataset = pd.concat(dataset,axis=0)
    dataset = dataset.drop_duplicates(keep='last')
    return dataset


def make_all():
    init_gmodel()
    for year in range(2007,2016):
        pnews = make(year)
        save(pnews,year)


if __name__ == '__main__':
    make_all()
