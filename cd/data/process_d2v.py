import logging
import re
from multiprocessing import cpu_count

import pandas as pd
import gensim.models.doc2vec as d2v

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
pattern_process = re.compile('[^a-zA-Z]+')


class Documents(object):

    def __init__(self,csv_files):
        self.csv_files = csv_files
        self.iter_csv_files = iter(csv_files)
        self.news = iter([])

    @staticmethod
    def _to_TaggedDocument(news):
        words = news.content
        tags = [news.Index]
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


empty_chars = '( ) - \' \" .  : ! 0 1 2 3 4 5 6 7 8 9 %'.split()

empty_chars = {c:' ' for c in empty_chars}
translator = str.maketrans(empty_chars)


def pre_process(s):
    ss = [w.lower() for w in pattern_process.sub(' ',s).split() if len(w) > 1]
    return ss


def trim_rule(word,count,min_count):
    print((word,count,min_count))

if __name__ == '__main__':
    # years = range(2007,2011)
    years = [2013]
    files = ['dataset/news_%d.csv' % y for y in years]
    documents = Documents(files)
    model = d2v.Doc2Vec(workers=cpu_count(),trim_rule=trim_rule,iter=5)
    model.build_vocab(documents)
