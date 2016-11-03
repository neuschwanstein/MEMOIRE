def split_and_normalize(newsmarket,sz=None):
    if sz is None:
        sz = int(0.8*len(newsmarket))
    train,test = newsmarket[:sz],newsmarket[sz:]
    mean = train.X.mean(axis=0)
    std = train.X.std(axis=0)
    train.X = train.
