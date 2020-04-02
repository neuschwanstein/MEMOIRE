import numpy as np
import matplotlib.pyplot as plt
import quandl as qd
import pandas as pd

# Part I
# Get data
qd.ApiConfig.api_key = 'TFPsUSNkbZiK8TgJJ_qa'

vix = qd.get('CBOE/VIX')[['VIX Close']]
vix = vix.rename(columns={'VIX Close':'vix'})

nas = qd.get('NASDAQOMX/COMP')[['Index Value']]
nas = nas.rename(columns={'Index Value':'nas'})

amz = qd.get_table('WIKI/PRICES', ticker = 'AMZN')[['date','close']].set_index('date')
amz = amz.rename(columns={'close':'amz'})


# Part II
# Create dataset
t = nas.join([vix,amz],how='inner')

t['r'] = t['amz'].shift(-1)

t = t/t.shift(1) - 1

t = (t - t.mean())/t.std()

t['mnas'] = t.nas.rolling(5).mean()
t['mvix'] = t.vix.rolling(5).mean()
t['mamz'] = t.amz.rolling(5).mean()

t['vnas'] = t.nas.rolling(5).std()
t['vvix'] = t.vix.rolling(5).std()
t['vamz'] = t.amz.rolling(5).std()



# Part III
