import logging
import re
from multiprocessing import cpu_count

import pandas as pd
import gensim.models.doc2vec as d2v

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pattern_process = re.compile('[^a-zA-Z]+')


class Documents(object):

    def __init__(self,csv_files,word2vec=False,doc2vec=False):
        self.word2vec = word2vec
        self.doc2vec = doc2vec
        self.csv_files = csv_files
        self.iter_csv_files = iter(csv_files)
        self.news = iter([])

    def _to_TaggedDocument(self,news):
        words = pre_process(news.content)
        tags = [news.Index]
        if self.word2vec:
            return words
        else:
            return d2v.TaggedDocument(words=words,tags=tags)

    def init_next_file(self):
        df = pd.read_csv(next(self.iter_csv_files))
        self.news = df.itertuples()
        return self._to_TaggedDocument(next(self.news))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._to_TaggedDocument(next(self.news))
        except StopIteration:
            try:
                return self.init_next_file()
            except StopIteration as e:
                self.__init__(self.csv_files)
                raise e


def pre_process(s):
    ss = [w.lower() for w in pattern_process.sub(' ',s).split() if len(w) > 1]
    return ss


def trim_rule(word,count,min_count):
    print((word,count,min_count))

if __name__ == '__main__':
    # years = range(2007,2011)
    years = [2007]
    files = ['dataset/news_%d.csv' % y for y in years]
    documents = Documents(files,doc2vec=True)
    # model = d2v.Word2Vec(documents,workers=1)
    model = d2v.Doc2Vec(documents,workers=12)
    # model = d2v.Doc2Vec(workers=cpu_count(),trim_rule=trim_rule,iter=5)
    # model.build_vocab(documents)
