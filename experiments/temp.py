from gensim.utils import lemmatize
from nltk.corpus import stopwords

stops = set(stopwords.words('english'))

def process(s):
    s = [w for w in s if w not in stops]
    s = bigram[s]
    # s = [w.split('/')[0] for w in lemmatize(' '.join(s),allowed_tags=re.compile('(NN)'),min_length=3)]
    return s
