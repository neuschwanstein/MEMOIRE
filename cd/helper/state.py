import os
from datetime import datetime as dt
import pickle as pcl

import numpy as np

def saver(datas,params,title=None,details=None):
    """Serializes numeric data and parameter that describes the experiment in a subfolder.

    :param datas: LIST of numpy elements. Each should be the output you want to save.
    :param params: DICT of parameters that lead to the output. If loops have been applied, then you should have arrays as vals
    :param title: Name of the experiment. Will also be the name of the subfolder. If absent, current date will be applied.
    :param details: Textual comments.
    :returns: None
    :rtype: None

    """
    subfolder_name = title or str(dt.now()).replace(' ','_')
    directory = './data/' + subfolder_name + '/'
    os.makedirs(directory,exist_ok=True)
    
    for var_name,data in datas.items():
        filename = directory + var_name
        np.save(filename,data)

    # DELETEME!
    # # Change key for utf-8 characters in case keys have greek letters (λ,β,etc.)
    # for key in params.keys():
    #     new_key = key.encode('utf-8')
    #     params[new_key] = params.pop(key)

    with open(directory + 'params.pcl','wb') as file:
        pcl.dump(params,file)

def loader(title):
    """Loads datas and params from specified directory

    :param title: subdirectory name.
    :returns: datas,params both as dictionaries
    :rtype: tuple

    """
    directory = './data/' + title + '/'

    datas = {}
    for file in os.listdir(directory):
        if not file.endswith('.npy'):
            continue
        name = file.strip('.npy')
        data = np.load(directory + file)
        datas[name] = data

    with open(directory + 'params.pcl','rb') as file:
        params = pcl.load(file)

    return datas,params
