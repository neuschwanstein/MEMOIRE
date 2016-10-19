import logging
import datetime as dt

import pandas as pd

import cd.data.process_google_d2v as pgd
import cd.data.helper as he


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    years = range(2007,2016)
    news = [pgd.get_news(y) for y in years]
    news = pd.concat(news)
    news = news[~news.index.duplicated(keep='last')]
