import datetime as dt

import numpy as np
import pandas as pd


def normalize_time(reference,dataset):
    morning = dt.time(9,30)
    afternoon = dt.time(16,0)
    morning_of = lambda d: dt.datetime.combine(d,morning)
    afternoon_of = lambda d: dt.datetime.combine(d,afternoon)

    if reference.min() > morning_of(dataset['time'].min()) or \
       reference.max() < afternoon_of(dataset['time'].max()):
        raise Exception('The reference series need to overlap the dataset time column')

    reference = reference.sort_values()
    reference = reference.map(pd.Timestamp.date)
    dataset = dataset.sort_values('time')

    reference = iter(reference)
    events = iter(dataset['time'])

    collapsed_times = []
    during = []

    # First seek when the actual reference date starts
    event_time = next(events)
    while True:
        ref_date = next(reference)
        if event_time < morning_of(ref_date):
            collapsed_times.append(ref_date)
            during.append(False)
            break

    # Then map to every event a corresponding reference date Rules:
    # 1. If it happens during the trading session (9:30am to 4pm) then
    # map it to the noon of the trading date.
    # 2. If it happens after trading session, map it to the morning of the
    # affected trading session (ie. the next morning)
    for event_time in events:
        if event_time < morning_of(ref_date):
            collapsed_times.append(ref_date)
            during.append(False)
        elif event_time >= morning_of(ref_date) and event_time < afternoon_of(ref_date):
            collapsed_times.append(ref_date)
            during.append(True)
        else:
            while True:
                next_ref_date = next(reference)
                if event_time < morning_of(next_ref_date):
                    collapsed_times.append(next_ref_date)
                    during.append(False)
                    break
            ref_date = next_ref_date

    # Update time column of the dataset with results
    dataset['time'] = pd.to_datetime(collapsed_times)
    dataset['during'] = during
    dataset = dataset.set_index('time')
    return dataset


def merge_news_with_returns(market,news):
    # Protip: the dataset must have been normalized using above method
    # news['temp'] = news.index.map(lambda d: d.date())
    # news['temp'] = pd.to_datetime(news['temp'])
    # news = pd.merge(left=news,right=market[['r']],how='left',left_on='temp',right_index=True)
    # news = pd.merge(left=news,right=market[['r']],how='left',left_index=True,right_index=True)
    # news = news.drop('temp',axis=1)
    news = news.join(market['r'],how='left')
    return news


def aggregate_returns(news):
    # Perhaps this method needs more careful design.
    # news = news.groupby('time').aggregate(np.mean)
    news = news.groupby([news.index,news.during]).aggregate(np.mean)
    return news
