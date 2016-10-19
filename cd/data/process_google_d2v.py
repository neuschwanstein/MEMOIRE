import re

import numpy as np
import pandas as pd
import gensim.models.word2vec as w2v

# Put this line in the toplevel
pattern_process = re.compile('[^a-zA-Z]+')
vec_length = 300
if 'gmodel' not in locals():
    gmodel = None


def init_gmodel():
    global gmodel
    if gmodel is None:
        gmodel = w2v.Word2Vec.load_word2vec_format(
            'dataset/GoogleNews-vectors-negative300.bin',binary=True)


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


def to_csv(news,year):
    csv_file = 'dataset/parsednews%d.csv' % year
    news.to_csv(csv_file)


def get_news(year,try_cache=True):
    if try_cache:
        # BAD!!! the two files don't even return the same object.
        try:
            csv_file = 'dataset/parsednews%d.csv' % year
            news = pd.read_csv(csv_file,parse_dates=['time'])
            news = news.set_index(['time','during'])
            return news
        except FileNotFoundError:
            pass

    init_gmodel()
    csv_file = 'dataset/news%d.csv' % year
    news = pd.read_csv(csv_file,parse_dates=['time'])
    news = remove_duplicates(news)
    vectors = process_vectors(news)
    news = news.join(vectors,how='right')
    news = news.drop('content',axis=1)
    return news
