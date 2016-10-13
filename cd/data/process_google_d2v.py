import logging
import re
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
import gensim.models.word2vec as w2v

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pattern_process = re.compile('[^a-zA-Z]+')


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

if __name__ == '__main__':
    switch = False
    if switch:
        gmodel = w2v.Word2Vec.load_word2vec_format('dataset/GoogleNews-vectors-negative300.bin',binary=True)
    news2007 = pd.read_csv('dataset/news_2007.csv')
    news2007['content'] = news2007['content'].apply(pre_process)
    news2007['mean_vector'] = news2007['content'].apply(mean_vector)
    print('Done')
