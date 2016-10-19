# This script is intended to be run once all news have been processed
# by a word2vec process and stored in a csv file in the dataset
# folder. We then just loop and apply aggreation rules found in the
# helper file
#
# I'm actually helpless on how I should organize my code...

# TODO...


# ys = range(2007,2016)
# for y in ys:
#   news = get_csv(y,try_cache=False)
#   news = he.normalize_time(market.index,news)
#   news = he.aggregate_returns(news)
#   news = he.merge_news_with_returns(market,news)
#   to_csv(news,y)
