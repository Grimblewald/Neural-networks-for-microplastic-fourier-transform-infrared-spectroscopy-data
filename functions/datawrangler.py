# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:47:07 2023

@author: fritz
"""
import pandas as pd
from functions.extrafunctions import get_file_extension

def read(data_info):
    
    datasetdict = {}
    
    for key in data_info:
        name = data_info[key]['name']
        path = data_info[key]['path']
        datatype = data_info[key]['type']
        labelcolname = data_info[key]['label_column']
        
        dataformat = get_file_extension(path)
        
        data = None
        
        match datatype:
            case 'file':
                match dataformat:
                    case 'csv':
                        data = pd.read_csv(path)
                        print("read {} succesfully".format(name))
                    case 'excel':
                        data = pd.read_excel(path)
                        print("read {} succesfully".format(name))
            case 'folder':
                print('folder')
                data = "a"
            case _:
                raise TypeError("No recognised type set for dataset {}, check for typos/capitals?".format(name))
        
        datasetdict[name] = {'path':path,
                             'labelcol':labelcolname,
                             'data':data}

    return datasetdict
    
def process(config, data):
    data = data.keys()
    return data
