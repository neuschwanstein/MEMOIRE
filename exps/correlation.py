import logging
import datetime as dt

import pandas as pd

from cd.data.process_quandl import get_market
from cd.data.process_google_d2v import get_news
import cd.data.helper as he


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if __name__ == '__main__':
    power_switch = False

    if 'market' not in locals() or power_switch:
        start = dt.date(2006,10,1)
        end = dt.date(2016,2,10)
        logging.info('Fetching market data from %s to %s' % (start,end))
        market = get_market(start,end)

    if 'news' not in locals() or power_switch:
        years = list(range(2007,2016))
        logging.info('Fetching market news from %s to %s' % (years[0],years[-1]))
        news = pd.DataFrame()
        for y in years:
            logging.info('Fetching year %s' % y)
            current = get_news(y)
            news = news.append(current,ignore_index=True)
        del current

    news = he.normalize_time(market.index,news)
    news = he.aggregate_returns(news)
    total = he.merge_news_with_returns(market,news)
    
