'''The point of this experiment is to try if we can pick any signal of news occuring
during the trading session (marked as during(True)), that is if we train upon a test set
of many years we can obtain a positive expected utility on the test set.

'''

from cd.datasets import newsmarket as mkt
from 

if __name__ == '__main__':
    newsmarket = mkt.load_all()
    newsmarket = newsmarket.during(True)

    train_sz = int(0.8*len(newsmarket))
    train,test = newsmarket[:train_sz],newsmarket[train_sz:]
    
