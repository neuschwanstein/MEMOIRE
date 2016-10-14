from collections import deque
import datetime as dt

import pandas as pd


def normalize_time(reference,dataset,half_days=True):
    morning = dt.time(9,30)
    noon = dt.time(12,0)
    afternoon = dt.time(16,0)
    morning_of = lambda d: dt.datetime.combine(d,morning)
    noon_of = lambda d: dt.datetime.combine(d,noon)
    afternoon_of = lambda d: dt.datetime.combine(d,afternoon)

    if reference.min() > morning_of(dataset['time'].min()) or \
       reference.max() < afternoon_of(dataset['time'].max()):
        raise Exception('The reference series need to overlap the dataset time column')

    reference = reference.sort_values()
    reference = reference.map(pd.Timestamp.date)
    dataset = dataset.sort_values('time')

    reference = iter(reference)
    events = iter(dataset['time'])

    collapsed_dates = []

    # First seek when the actual reference date starts
    event_time = next(events)
    while True:
        ref_date = next(reference)
        if event_time < morning_of(ref_date):
            collapsed_dates.append(ref_date)
            break

    # Then map to every event a corresponding reference date
    # If it happens during trading
    for event_time in events:
        if event_time < morning_of(ref_date):
            collapsed_dates.append(morning_of(ref_date))
        elif event_time >= morning_of(ref_date) and event_time < afternoon_of(ref_date):
            collapsed_dates.append(noon_of(ref_date))
        else:
            while True:
                next_ref_date = next(reference)
                if event_time < morning_of(next_ref_date):
                    collapsed_dates.append(morning_of(next_ref_date))
                    break
            ref_date = next_ref_date

    # Update time column of the dataset with results
    dataset['time'] = collapsed_dates
    return dataset
