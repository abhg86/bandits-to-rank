#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""


from os import listdir
from os.path import isfile, join
import numpy as np
import json
import gzip

def retrieve_data_from_zip(file_name, my_assert=True, accept_empty_file=True):
    if isfile(file_name):
        with gzip.GzipFile(file_name, 'r') as fin:
            json_bytes = fin.read()

        json_str = json_bytes.decode('utf-8')  # 2. string (i.e. JSON)
        data = json.loads(json_str)
    else:
        data = None
    return data


def time_sec_to_HMS(sec):
    heure=sec//3600
    rest_h=sec%3600
    minute=rest_h//60
    rest_m=rest_h%60
    return heure, minute, rest_m


if __name__ == "__main__":
    mypath = "C:/Users/gauthieca/Desktop/Code_These/bandits-to-rank/Automatisation/results/timings"
    for f in [f for f in listdir(mypath) if isfile(join(mypath, f))] : # and f.find("Yandex_CM_0")>=0]:
        data = retrieve_data_from_zip(join(mypath, f))
        time_mean = np.mean(np.transpose(data['time_to_play'])[-1])
        print(f, data['time_recorded'][-1][-1], data['time_to_play'][-1][-1])
        print(time_sec_to_HMS(time_mean), time_mean/ data['time_recorded'][-1][-1])


    #print(time_sec_to_HMS(3600*4+60*11+0)*100/1)
