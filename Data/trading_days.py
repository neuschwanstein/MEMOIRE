import datetime as dt

from dateutil.rrule import rrule,rruleset,YEARLY,DAILY,SA,SU,FR,MO,TH,WEEKLY


# Inspired by: https://gist.github.com/jckantor/d100a028027c5a6b8340
def next_trading_day(date):
    date = date + dt.timedelta(days=1)

    stop_days = rruleset()
    stop_days.rrule(rrule(YEARLY,dtstart=date,byweekday=(SA,SU)))
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=12,bymonthday=31,byweekday=FR))  # New Years Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=1,bymonthday=1))                 # New Years Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=1,bymonthday=2,byweekday=MO))    # New Years Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=1,byweekday=MO(3)))              # Martin Luther King Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=2,byweekday=MO(3)))              # Washington's Birthday
    stop_days.rrule(rrule(YEARLY,dtstart=date,byeaster=-2))                            # Good Friday
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=5,byweekday=MO(-1)))             # Memorial Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=7,bymonthday=3,byweekday=FR))    # Independence Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=7,bymonthday=4))                 # Independence Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=7,bymonthday=5,byweekday=MO))    # Independence Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=9,byweekday=MO(1)))              # Labor Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=11,byweekday=TH(4)))             # Thanksgiving Day
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=12,bymonthday=24,byweekday=FR))  # Christmas
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=12,bymonthday=25))               # Christmas
    stop_days.rrule(rrule(YEARLY,dtstart=date,bymonth=12,bymonthday=26,byweekday=MO))  # Christmas
    stop_days.exrule(rrule(WEEKLY,dtstart=date,byweekday=(SA,SU)))

    set = rruleset()
    set.rrule(rrule(DAILY,count=10,dtstart=date))
    set.exrule(rrule(WEEKLY,dtstart=date,byweekday=(SA,SU)))
    set.exrule(stop_days)

    return next(iter(set))
