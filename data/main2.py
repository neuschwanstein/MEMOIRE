import requests
import re
import psycopg2
from urlparse import urlparse
from datetime import datetime
    
from robustDictionary import RobustDictionary

FEED_URL = "feed/http://finance.yahoo.com/rss/headline?s="
FEEDLY_URL = "http://cloud.feedly.com/v3/streams/contents"

QUERY_STRING = """
INSERT INTO feed (ticker,href,published,title,engagement,engagement_rate,provider,content)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
"""

ticker = "GOOG"
count = 10000
continuation = ''

while continuation is not None:
    r = requests.get(FEEDLY_URL, \
                     params = { "streamId": FEED_URL + ticker, \
                                "count": count, \
                                "continuation": continuation })

    if not r.ok:
        raise Exception("Error code: " + str(r.status_code))

    items = RobustDictionary(r.json())
    continuation = items['continuation']

    # Database operations
    con = psycopg2.connect(dbname='feed', user='')
    cur = con.cursor()

    for item in items['items']:
        item = RobustDictionary(item)

        if (item['origin']['title'] == "Yahoo! Finance: RSS feed not found"):
            continue

        published = datetime.fromtimestamp(int(item['published'])/1000) #published field expressed in ms.
        title = item['title']
        engagement = int(item['engagement']) if item['engagement'] is not None else 0
        engagement_rate = float(item['engagementRate']) if item['engagementRate'] is not None else 0

        total_href = item['alternate'][0]['href']
        if (total_href != "http://finance.yahoo.com/q/h?s=yhoo"):
            regex_href = re.match("(.+)\*(.+)", total_href)
            href = regex_href.group(2)
            provider = urlparse(href).netloc
        else:
            href = None
            provider = None

        if (item['summary'] is None):
            content = None
        else:
            m = re.match("(at )?(.+)\] - (.+)", item['summary']['content'])
            content = m.group(3)

        cur.execute(QUERY_STRING, \
                    (ticker, href, published, title, engagement, engagement_rate, provider, content))

    con.commit()
    print "DONE"
