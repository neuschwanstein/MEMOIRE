import pandas as pd

from . import market,news


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
    def load(cls,years=[2007,2015],features=None):
        try:
            start_year = min(years)
            end_year = max(years)
        except:
            start_year = years
            end_year = years
        years = (start_year,end_year)

        newsmarket = market.load(*years)
        newsmarket = newsmarket[['r']]

        if 'vol' is in features:
            vol = market.load_vol(*years)
            newsmarket = newsmarket.join(vol,how='inner')

        if 'news' is in features:
            news = news.load(

        return NewsMarket(newsmarket)
