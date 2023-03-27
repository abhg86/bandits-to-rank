#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""""""


from os import listdir
from os.path import isfile, join
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
    mypath = "/Users/rgaudel/tmp/igrida/timings"
    for f in [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.find("Yandex_0")>=0 and f.find("PBM-TS")>=0]:
        data = retrieve_data_from_zip(join(mypath, f))
        print(f, data['time_recorded'][-1][-1], data['time_to_play'][-1][-1])
        print(time_sec_to_HMS(data['time_to_play'][-1][-1]), data['time_to_play'][-1][-1] / data['time_recorded'][-1][-1])


    #print(time_sec_to_HMS(3600*4+60*11+0)*100/1)