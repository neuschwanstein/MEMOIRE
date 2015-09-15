# For each news feed or news feed, we want a iterator with a certain common api
# (say derving from class Feed) such that each feed has a title, a date, an author,
# and a corpus.
# Then, doing NLP on our sources should not be too hard and eventually we should end up
# having a vector for each of our document coming out of these iterated feed.

import requests
import re
import psycopg2
from urlparse import urlparse
from datetime import datetime

from robustDictionary import RobustDictionary

FEEDLY_URL = "http://cloud.feedly.com/v3/streams/contents"
REUTERS_URL = "feed/http://feeds.reuters.com/reuters/globalmarketsNews"


QUERY_STRING = """
INSERT INTO reuters_globalMarketNews
(feedly_id,href,published,title,engagement,engagement_rate,content)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""
count = 10000
continuation = ''

con = psycopg2.connect(dbname='feed')
db = con.cursor()

while continuation is not None:
    r = requests.get(FEEDLY_URL, \
                    params = { "streamId": REUTERS_URL, \
                               "count": count, \
                               "continuation": continuation })

    if not r.ok:
        raise Exception("Error code: " + str(r.status_code))

    items = RobustDictionary(r.json())
    continuation = items['continuation']

    for item in items['items']:
        item = RobustDictionary(item)

        feedly_id = item['id']
        published = datetime.fromtimestamp(int(item['published'])/1000)
        title = item['title']
        engagement = int(item['engagement']) if item['engagement'] is not None else 0
        engagement_rate = float(item['engagement_rate']) if item['engagement_rate'] is not None else 0
        
        href = item['alternate'][0]['href']

        if item['summary'] is None:
            content = None
        else:
            content = item['summary']['content']

        db.execute(QUERY_STRING, \
                   (feedly_id,href,published,title,engagement,engagement_rate,content))

    con.commit()
    print "DONE"

con.close()
