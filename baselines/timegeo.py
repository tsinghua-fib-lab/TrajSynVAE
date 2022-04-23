# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Sep 15th 18:05:53 2020

Baseline 2: TimeGeo
"""


import timegeo_master.only_map
import timegeo_master.eval_map
from time import strftime, localtime
import os 
import numpy as np
import pandas as pd
from data_prepare import MYDATA


def ISPToTimegeo():
    #Load
    with open('./data/ISP/Trace.txt', 'r') as f:
        raw = []
        for line in f.readlines():
            raw.append(list(map(int, line.split())))
        raw = np.array(raw)
    D = pd.DataFrame(raw, columns=['usr', 'tim', 'loc'])
    # Time
    def abstime(tim):
        time = [tim // 1000000, (tim % 1000000) // 10000, (tim % 10000) // 100, tim % 100]
        return pd.Timestamp('2016-04-19') + pd.Timedelta(time[0] * 24 * 60 * 60 + time[1] * 60 * 60 + time[2] * 60 + time[3], unit='sec')
    D['AbsTime'] = D['tim'].apply(abstime)

    # Location
    with open('./data/ISP/POIdis.txt', 'r') as f:
        poi = []
        for line in f.readlines():
            poi.append(list(map(int, line.split())))
    poi = np.array(poi)
    poi = poi[:, 1:]

    gps = []
    with open('./data/ISP/LocList.txt', 'r') as f:
        lines = f.readlines()
    for l in lines:
        gps.append([float(x) for x in l.split()])
    gps = np.array(gps)

    D['latitude'] = D['loc'].apply(lambda x:gps[x, 1])
    D['longitude'] = D['loc'].apply(lambda x:gps[x, 0])

    lines = []
    for user in np.sort(D['usr'].unique()):
        line = str(user) + '\t'
        for _, row in D[D['usr'] == user].iterrows():
            line += '%f,%f,%s;' % (row['longitude'], row['latitude'], row['AbsTime'])
        print(user)
        lines.append(line)

    with open('./data/ISP/TimeGeo.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def FourSquareToTimeGeo(CITY):
    
    # Load
    D = pd.read_csv('./data/FourSquare/' + CITY + '/dataset_TSMC2014_' + CITY + '.csv')
        
    # Time
    MONTH = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
    def current_time(x):
        offset, utctime = x['timezoneOffset'], x['utcTimestamp']
        day, month, year, time = utctime[8: 10], utctime[4: 7], utctime[26: ], utctime[10: 19]
        return pd.Timestamp(year + '-' + MONTH[month] + '-' + day + time) + pd.Timedelta(minutes = offset) 
    D['localTime'] = D.apply(current_time, axis = 1)
    D = D.sort_values(by=['userId', 'localTime'])
    D = D.drop_duplicates().reset_index(drop=True)

    lines = []
    for user in np.sort(D['userId'].unique()):
        line = str(user) + '\t'
        for _, row in D[D['userId'] == user].iterrows():
            line += '%f,%f,%s;' % (row['longitude'], row['latitude'], row['localTime'])
        print(user)
        lines.append(line)

    with open('./data/FourSquare/' + CITY + '/TimeGeo.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def GeoLifeToTimeGeo():
    D = pd.read_csv('./data/GeoLife/GeoLife.csv')

    lines = []
    for user in np.sort(D['usr'].unique()):
        line = str(user) + '\t'
        for _, row in D[D['usr'] == user].iterrows():
            line += '%f,%f,%s %s;' % (row['lon'], row['lat'], row['date'], row['time'])
        print(user)
        lines.append(line)

    with open('./data/GeoLife/TimeGeo.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')


def timegeo(data):

    # Get the input for TimeGeo ready
    if os.path.exists(data.PATH + 'timegeo.npy'):
        infos = np.load(data.PATH + 'timegeo.npy', allow_pickle=True)
    else:
        infos = timegeo_master.only_map.main(filter_mode='user', fp=data.PATH + 'TimeGeo.txt')

    infos = [user for user in infos if int(user['user_id']) in data.USERLIST]

    # Generate sequences
    rhythm = eval(open('timegeo_master/rhythm').readline())
    deltar = eval(open('timegeo_master/deltar').readline())
    simulate = timegeo_master.eval_map.main(rhythm_global=rhythm, deltar_global=deltar, map_mode='simulate', infos=infos)
    geos = timegeo_master.eval_map.main(rhythm_global=rhythm, deltar_global=deltar, map_mode='predict', infos=simulate)

    # np.save('./data_TimeGeo/' + data.PATH[7:] + 'geos_' + data.LOCATION_MODE + '.npy', geos)
    
    # Change the data format for the output trajectory
    TimeGeo = {}
    for geo in geos:
        stays = geo['predict_trace']
        user = int(geo['user_id'])
        print(user)
        TimeGeo[user] = {}
        loc = []
        tim = []
        for stay in stays:
            if stay == 0:
                continue
            location = np.argmin(np.abs(data.GPS[:,0] - stay[1]) + np.abs(data.GPS[:,1] - stay[2]))
            if len(loc) != 0 and location == loc[-1]:
                continue
            def abstime(tim):
                time = [tim // 1000000, (tim % 1000000) // 10000, (tim % 10000) // 100, tim % 100]
                return time[0] * 24 * 60 + time[1] * 60 + time[2] + time[3] / 60
            time = abstime(int(strftime("%d%H%M%S", localtime(stay[3]))))
            loc.append(location)
            tim.append(time)
        traj = pd.DataFrame({'loc': loc, 'tim': tim})
        traj['sta'] = traj['tim'].shift(-1) - traj['tim']
        traj = traj.iloc[:-1]
        traj['day'] = traj['tim'] // 1440
        def cut(x):
            TimeGeo[user][len(TimeGeo[user])] = x.to_dict(orient = 'list')
            return x
        _ = traj.groupby('day').apply(cut)
        
    np.save('./data_TimeGeo/' + data.PATH[7:] + 'timegeo_' + data.LOCATION_MODE + '.npy', TimeGeo)

    return TimeGeo

if __name__ == '__main__':
    dataI0 = MYDATA('ISP', 0)
    dataI1 = MYDATA('ISP', 1)
    dataI2 = MYDATA('ISP', 2)
    dataG0 = MYDATA('GeoLife', 0)
    dataG1 = MYDATA('GeoLife', 1)
    dataG2 = MYDATA('GeoLife', 2)
    dataFN0 = MYDATA('FourSquare_NYC', 0)
    dataFN1 = MYDATA('FourSquare_NYC', 1)
    dataFN2 = MYDATA('FourSquare_NYC', 2)
    dataFT0 = MYDATA('FourSquare_TKY', 0)
    dataFT1 = MYDATA('FourSquare_TKY', 1)
    dataFT2 = MYDATA('FourSquare_TKY', 2)
    timegeo(dataI0)
    timegeo(dataI1)
    timegeo(dataI2)
    timegeo(dataG0)
    timegeo(dataG1)
    timegeo(dataG2)
    timegeo(dataFN0)
    timegeo(dataFN1)
    timegeo(dataFN2)
    timegeo(dataFT0)
    timegeo(dataFT1)
    timegeo(dataFT2)