import os
import re

import numpy as np
import pandas as pd
import gensim.models.word2vec as w2v

from datasets import news as ns

# Put this line in the toplevel
pattern_process = re.compile('[^a-zA-Z]+')
vec_length = 300
if 'gmodel' not in locals():
    gmodel = None


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


def make(year):
    init_gmodel()
    news = ns.load(year)
    news = remove_duplicates(news)
    vectors = process_vectors(news)
    news = news.join(vectors,how='right')
    news = news.drop('content',axis=1)
    return news


def save(news,year):
    filename = os.path.join(os.path.dirname(__file__),
                            'parsed_news/pnews%d.csv' % year)
    news.to_csv(filename,encoding='utf-8')


def load(year):
    filename = os.path.join(os.path.dirname(__file__),
                            'parsed_news/pnews%d.csv' % year)
    news = pd.read_csv(filename,parse_dates=['time'])
    news = news.set_index(['time','during'])
    return news


def make_all():
    init_gmodel()
    for year in range(2007,2016):
        pnews = make(year)
        save(pnews,year)

if __name__ == '__main__':
    make_all()
