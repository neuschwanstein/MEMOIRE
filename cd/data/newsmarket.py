import pandas as pd

from . import market,news as ns


class NewsMarket(pd.DataFrame):

    @property
    def _constructor(self):
        return NewsMarket

    @property
    def X(self):
        return self.filter(regex='^f_')

    def during(self,bool):
        return self.xs(bool,level='during')

    def __getitem__(self,key):
        if key is 'X':
            return self.X
        else:
            return super().__getitem__(key)

    def __setitem__(self,key,val):
        if key is 'X':
            cols = self.columns[self.columns.str.contains('f_')]
            try:
                self.loc[:,cols] = val.values
            except:
                self.loc[:,cols] = val
        else:
            super().__setitem__(key,val)

    @classmethod
    def load(cls,*years,features=[]):
        newsmarket = market.load(*years,what='r')
        newsmarket = newsmarket[['r']]

        if 'vol' in features:
            vol = market.load(*years,what='vol')
            newsmarket = newsmarket.join(vol,how='inner')

        if 'news' in features:
            news = ns.load(*years)
            news = news.reset_index(level=1)
            # newsmarket = newsmarket.join(news,how='inner')
            newsmarket = newsmarket.merge(news,left_index=True,right_index=True)
            newsmarket = newsmarket.set_index('during',append=True)

        return NewsMarket(newsmarket)
