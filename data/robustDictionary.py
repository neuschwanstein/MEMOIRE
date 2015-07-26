class RobustDictionary(dict):
    def __getitem__(self,key):
        try:
            result = dict.__getitem__(self,key)
            if (type(result) is dict):
                return RobustDictionary(result)
            elif (type(result) is list):
                return RobustList(result)
            else:
                return result
        except Exception:
            return None

class RobustList(list):
    def __getitem__(self,index):
        result = list.__getitem__(self,index)
        if (type(result) is not dict):
            return result
        else:
            return RobustDictionary(result)

# from requests import Response

# class RobustResponse(Response):
#     def __init__(self, response):
#         self = response

#     def __getitem__(self,key):
#         try:
#             # return dict.__getitem__(self,key)
#             return self[key]
#         except Exception:
#             return None
