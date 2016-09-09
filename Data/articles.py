import psycopg2 as db
from collections import namedtuple

conn = db.connect('dbname=TBM')
cur = conn.cursor()

'''Adaptor of the articles3 table.'''
class Articles:
    def __init__(self,fields=['id','date','title','content','link'],n='ALL'):
        self.fields = fields
        self.article_record = namedtuple('Article',fields)
        select_query = 'SELECT %s FROM articles3 LIMIT %s' % (','.join(fields),n)
        self.batch_size = 1000
        cur.execute(select_query)
        self.records = []
        self.length = None

    def __iter__(self):
        return self

    def __next__(self):
        if not self.records:
            self.records = cur.fetchmany(self.batch_size)
            if not self.records:
                raise StopIteration()
        next_article = self.records.pop(0)
        return self.article_record(*next_article)
