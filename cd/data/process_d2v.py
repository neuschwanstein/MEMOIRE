import pandas as pd
import gensim.models.doc2vec as d2v


class Documents(object):

    def __init__(self,csv_files):
        self.csv_files = iter(csv_files)
        self.news = iter([])

    @staticmethod
    def _to_TaggedDocument(news):
        words = news.content
        tags = news.Index
        return d2v.TaggedDocument(words=words,tags=tags)

    def init_next_file(self):
        df = pd.read_csv(next(self.csv_files))
        self.news = df.itertuples()
        return self._to_TaggedDocument(next(self.news))

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._to_TaggedDocument(next(self.news))
        except StopIteration:
            return self.init_next_file()
