import logging
import re

import numpy as np
import pandas as pd
import gensim.models.word2vec as w2v

# Put this line in the'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pattern_process = re.compile('[^a-zA-Z]+')
vec_length = 300
gmodel = None


def init_gmodel():
    global gmodel
    if gmodel is None:
        gmodel = w2v.Word2Vec.load_word2vec_format(
            'dataset/GoogleNews-vectors-negative300.bin',binary=True)


def pre_process(s):
    ss = [w.lower() for w in pattern_process.sub(' ',s).split() if len(w) > 1]
    return ss


def mean_vector(lst):
    def vector(s):
        try:
            return gmodel[s]
        except KeyError:
            return np.zeros(300)
    return np.mean([vector(s) for s in lst],axis=0)


def remove_duplicates(ds):
    dups = ds.duplicated('content')
    return ds[~dups]


def process_vectors(vectors):
    vectors = vectors[~vectors.isnull()]
    index = vectors.index
    vectors = np.vstack(vectors)
    vectors = pd.DataFrame(vectors)
    cols = ['d2v_%d' % i for i in range(1,vec_length+1)]
    vectors.columns = cols
    vectors.index = index
    return vectors


def get_news(year):
    init_gmodel()
    csv_file = 'dataset/news%d.csv' % year
    news = pd.read_csv(csv_file,parse_dates=['time'])
    news = remove_duplicates(news)
    vectors = news['content'].apply(pre_process).apply(mean_vector)
    vectors = process_vectors(vectors)
    news = news.join(vectors,how='right')
    news = news.drop('content',axis=1)
    return news
