import datetime as dt
import os
import re

import numpy as np
import pandas as pd
import gensim.models.word2vec as w2v

from . import market as mkt
from . import raw_news as raw

pattern_process = re.compile('[^a-zA-Z]+')
vec_length = 300
if 'gmodel' not in locals():
    gmodel = None

news_filename = os.path.join(os.path.dirname(__file__),'news/news%d.csv')
# raw_news_filename = os.path.join(os.path.dirname(__file__),'raw_news/raw_news%d.csv')


def init_gmodel():
    global gmodel
    if gmodel is None:
        w2v_filename = os.path.join(os.path.dirname(__file__),
                                    'w2v/GoogleNews-vectors-negative300.bin')
        gmodel = w2v.Word2Vec.load_word2vec_format(w2v_filename,binary=True)


def delete_gmodel():
    global gmodel
    if gmodel is not None:
        del gmodel


def mean(lst):
    global gmodel

    def vector(s):
        try:
            return gmodel[s]
        except KeyError:
            return np.zeros(300)
    return np.mean([vector(s) for s in lst],axis=0)


# def process_vectors(raw_news):
#     # First clean up
#     vectors = raw_news['content']
#     vectors = vectors[~vectors.isnull()]
#     vectors = vectors.apply(to_list_of_words)
#     empty_vectors = vectors.apply(len) == 0
#     vectors = vectors[~empty_vectors]
#     vectors = vectors[~vectors.isnull()]

#     # Mean of the words
#     vectors = vectors.apply(mean)

#     # Then convert it to proper news
#     index = vectors.index
#     vectors = np.vstack(vectors)
#     vectors = pd.DataFrame(vectors)
#     cols = ['f_d2v_%d' % i for i in range(1,vec_length+1)]
#     vectors.columns = cols
#     vectors.index = index
#     return vectors


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


def vector(s):
    try:
        return gmodel[s]
    except KeyError:
        return np.zeros(300)


def process_raw_news(s):
    hello = s
    # First convert it to list of words
    try:
        s = [w.lower() for w in pattern_process.sub(' ',s).split() if len(w) > 1]
    except Exception as e:
        print(s)
        raise e

    # Then each word is represented by its vector
    s = [vector(w) for w in s]

    # Finally return the mean of all vectors
    if s == []:
        return np.nan
    s = np.mean(s,axis=0)
    return s


def preprocess_raw_news(raw_news):
    raw_news = raw_news[~raw_news['content'].duplicated()]
    raw_news = raw_news[~raw_news['content'].isnull()]
    empty_content = raw_news['content'].apply(len) == 0
    raw_news = raw_news.loc[~empty_content,:]
    return raw_news


def postprocess_news(news):
    # Remove trivial results
    news = news[~news['content'].isnull()]
    news = news[~(news['content'].apply(len) == 0)]

    # Then convert it to a full fledged matrix
    w2v = np.vstack(news['content'])
    w2v = pd.DataFrame(w2v)
    cols = ['f_d2v_%d' % i for i in range(1,len(w2v.columns)+1)]
    w2v.columns = cols
    w2v.index = news.index

    news = news.drop('content',axis=1)
    news = pd.concat([news,w2v],axis=1)
    return news


def make(year):
    init_gmodel()

    # Load raw news and translate their content to vectorial representation
    raw_news = raw.load(year)
    news = preprocess_raw_news(raw_news)
    news['content'] = news['content'].apply(process_raw_news)
    news = postprocess_news(news)

    # Collapse dates to either have them during the trading session or
    # in the after hours.
    market = mkt.load(year,year)
    reference = market.index
    news = collapse_time(reference,news)

    # Aggregate (Mean) each trading and after hours sessions newsmarket vector representation
    news = news.groupby([news.index,news.during]).aggregate(np.mean)

    # Then add a return column
    # newsmarket = newsmarket.join(market['r'],how='left')
    return news


def save(newsmarket,year):
    newsmarket.to_csv(news_filename % year,encoding='utf-8')


def load(year):
    news = pd.read_csv(news_filename % year,parse_dates=['time'])
    news = news.set_index(['time','during'])
    return news


def load_all():
    years = range(2007,2016)
    newsmarket = [load(y) for y in years]
    newsmarket = pd.concat(newsmarket,axis=0)
    newsmarket = newsmarket.drop_duplicates(keep='last')
    return newsmarket


def make_all():
    init_gmodel()
    for year in range(2007,2016):
        print(year)
        pnews = make(year)
        save(pnews,year)


if __name__ == '__main__':
    make_all()
