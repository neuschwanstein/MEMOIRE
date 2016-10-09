from collections import UserList


class Namedtuples(UserList):
    def __init__(self,lst):
        super().__init__(lst)
        try:
            self.fields = lst[0]._fields
        except:
            pass

    def __getattr__(self,key):
        return [getattr(s,key) for s in self.data]

    def __setattr__(self,key,values):
        if key is not 'fields' and key[-1] == 's' and key[:-1] in self.fields:
            key = key[:-1]
            for i,val in enumerate(values):
                self.data[i] = self.data[i]._replace(**{key:val})
        else:
            super().__setattr__(key,values)

    def __getitem__(self,i):
        # http://stackoverflow.com/a/27552501/152244
        res = self.data[i]
        return type(self)(res) if isinstance(i,slice) else res


class _namedtuples(UserList):
    def __init__(self,lst):
        super().__init__(lst)
        if lst != []:
            self._fields = lst[0]._fields
            self._create_accessors()

    def _create_accessors(self,field=None):
        return 1


if (__name__ == '__main__'):
    C = _namedtuples([])
