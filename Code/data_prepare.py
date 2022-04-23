# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Sep 15th 18:05:53 2020

Prepare Data for the Model
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

class MYDATA(Dataset):

    def __init__(self, NAME, LOCATION_MODE) -> None:
        super().__init__()

        self.LOCATION_MODE = str(LOCATION_MODE)
        DIVIDE = {'0':0.003, '1':0.01, '2':0.03}
        self.DIVIDE_LEVEL = DIVIDE[self.LOCATION_MODE]

        if len(NAME) > 10:
            self.CITY = NAME[-3:]
            self.NAME = NAME[:10]
            self.PATH = './data/' + self.NAME + '/' + self.CITY + '/'  
        if NAME == 'ISP':
            self.NAME = NAME
            self.PATH = './data/' + self.NAME + '/' 
        if NAME == 'GeoLife':
            self.NAME = NAME
            self.PATH = './data/' + self.NAME + '/' 

        self.MIN_LEN = 5
        self.TIME_MIN = 10
        self.TIME_MAX = 10080 if self.NAME == 'FourSquare' else 1440

        self.EXIST = self.loaddata()
        if not self.EXIST:
            if self.NAME == 'ISP':
                Data = self.dfprepare_ISP()
            if self.NAME == 'GeoLife':
                Data = self.dfprepare_GL()
            if self.NAME == 'FourSquare':
                Data = self.dfprepare_FS()
            self.DATA = self.preprocess(Data)
        self.attrprepare()


    def dfprepare_FS(self):
        # Load
        D = pd.read_csv(self.PATH + 'dataset_TSMC2014_' + self.CITY + '.csv')
        
        # Time
        MONTH = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06', 'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}
        def current_time(x):
            offset, utctime = x['timezoneOffset'], x['utcTimestamp']
            day, month, year, time = utctime[8: 10], utctime[4: 7], utctime[26: ], utctime[10: 19]
            return pd.Timestamp(year + '-' + MONTH[month] + '-' + day + time) + pd.Timedelta(minutes = offset) 
        D['localTime'] = D.apply(current_time, axis = 1)
        D = D.sort_values(by=['userId', 'localTime'])
        D = D.drop_duplicates().reset_index(drop=True)
        def timedelta(x):
            x['timeDelta'] = x['localTime'].shift(-1) - x['localTime']
            return x
        D = D.groupby("userId").apply(timedelta).reset_index(drop=True)
        D['timeDelta'] = D['timeDelta'].dt.total_seconds().fillna(0.0) / 60
        D['absTime'] = (D['localTime'] - D['localTime'].min()).dt.total_seconds() / 60
       
        # Location
        MIN_LAT = D['latitude'].min()
        MIN_LON = D['longitude'].min()
        MAX_LAT = D['latitude'].max()
        MAX_LON = D['longitude'].max()
        def location2Id(x):
            return int(((x['latitude'] - MIN_LAT) // self.DIVIDE_LEVEL) * (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL) + (x['longitude'] - MIN_LON) // self.DIVIDE_LEVEL)
        self.VALIDID = np.sort(D.apply(location2Id, axis = 1).unique())
        def locId(x):
            return np.where(self.VALIDID == location2Id(x))[0][0]
        D['locId'] = D.apply(locId, axis = 1)
        self.D = D

        # POI & GPS
        POI_NAME = D['venueCategory'].unique()
        self.POI = np.zeros((len(D['locId'].unique()), len(POI_NAME)), dtype=int)
        for _, row in D.iterrows():
            self.POI[row['locId'], np.where(POI_NAME == row['venueCategory'])] += 1
        def locId2location(locId):
            x = self.VALIDID[locId]
            lat = (x // (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL)) * self.DIVIDE_LEVEL + MIN_LAT + 0.5 * self.DIVIDE_LEVEL
            lon = (x % (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL)) * self.DIVIDE_LEVEL + MIN_LON + 0.5 * self.DIVIDE_LEVEL
            return (lat, lon)
        self.GPS = np.array([locId2location(id) for id in range(len(self.VALIDID))])
    
        # Output
        Data = D[['userId', 'locId', 'absTime', 'timeDelta']]
        Data = Data.rename(columns={'userId': 'usr', 'locId': 'loc', 'absTime': 'tim', 'timeDelta': 'sta'})
        return Data

    def dfprepare_GL(self):
        # Load
        D = pd.read_csv(self.PATH + 'GeoLife.csv')
        def tim(x):
            return pd.Timestamp(x['date'] + ' ' + x['time'])
        D['tim'] = D.apply(tim, axis = 1)
        D = D[['usr', 'lat', 'lon', 'tim']]
        D = D.astype({'lon': float, 'lat': float, 'usr': int})

        # Time
        D = D.sort_values(by=['usr', 'tim'])
        def timedelta(x):
            x['sta'] = x['tim'].shift(-1) - x['tim']
            return x
        D = D.groupby("usr").apply(timedelta).reset_index(drop=True)
        D['sta'] = D['sta'].dt.total_seconds().fillna(0.0) / 60
        D['absTime'] = (D['tim'] - D['tim'].min()).dt.total_seconds() / 60

        # Location
        MIN_LAT = D['lat'].min()
        MIN_LON = D['lon'].min()
        MAX_LAT = D['lat'].max()
        MAX_LON = D['lon'].max()
        def location2Id(x):
            return int(((x['lat'] - MIN_LAT) // self.DIVIDE_LEVEL) * (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL) + (x['lon'] - MIN_LON) // self.DIVIDE_LEVEL)
        self.VALIDID = np.sort(D.apply(location2Id, axis = 1).unique())
        def locId(x):
            return np.where(self.VALIDID == location2Id(x))[0][0]
        D['loc'] = D.apply(locId, axis = 1)
        self.D = D

        # POI & GPS
        self.POI = None
        def locId2location(locId):
            x = self.VALIDID[locId]
            lat = (x // (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL)) * self.DIVIDE_LEVEL + MIN_LAT + 0.5 * self.DIVIDE_LEVEL
            lon = (x % (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL)) * self.DIVIDE_LEVEL + MIN_LON + 0.5 * self.DIVIDE_LEVEL
            return (lat, lon)
        self.GPS = np.array([locId2location(id) for id in range(len(self.VALIDID))])

        # Output
        Data = D[['usr', 'loc', 'absTime', 'sta']]
        Data = Data.rename(columns={'absTime': 'tim'})
        return Data

    def dfprepare_ISP(self):

        #Load
        with open(self.PATH + 'Trace.txt', 'r') as f:
            raw = []
            for line in f.readlines():
                raw.append(list(map(int, line.split())))
            raw = np.array(raw)
        D = pd.DataFrame(raw, columns=['usr', 'tim', 'loc'])

        # Time
        def abstime(tim):
            time = [tim // 1000000, (tim % 1000000) // 10000, (tim % 10000) // 100, tim % 100]
            return time[0] * 24 * 60 + time[1] * 60 + time[2] + time[3] / 60
        D['tim'] = pd.DataFrame(D['tim'].apply(abstime))
        D = D.sort_values(by=['usr', 'tim'])
        def timedelta(x):
            x['sta'] = (x['tim'].shift(-1) - x['tim']).fillna(0.0)
            return x
        D = D.groupby("usr").apply(timedelta).reset_index(drop=True)

        # Location
        with open(self.PATH + 'POIdis.txt', 'r') as f:
            poi = []
            for line in f.readlines():
                poi.append(list(map(int, line.split())))
        poi = np.array(poi)
        poi = poi[:, 1:]

        gps = []
        with open(self.PATH + 'LocList.txt', 'r') as f:
            lines = f.readlines()
        for l in lines:
            gps.append([float(x) for x in l.split()])
        gps = np.array(gps)

        D['lat'] = D['loc'].apply(lambda x:gps[x, 1])
        D['lon'] = D['loc'].apply(lambda x:gps[x, 0])

        if self.LOCATION_MODE == 0:
            self.POI = poi
            self.GPS = gps
            self.D = D

        else:
            MIN_LAT = D['lat'].min()
            MIN_LON = D['lon'].min()
            MAX_LAT = D['lat'].max()
            MAX_LON = D['lon'].max()
            def location2Id(x):
                return int(((x['lat'] - MIN_LAT) // self.DIVIDE_LEVEL) * (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL) + (x['lon'] - MIN_LON) // self.DIVIDE_LEVEL)
            self.VALIDID = np.sort(D.apply(location2Id, axis = 1).unique())
            def locId(x):
                return np.where(self.VALIDID == location2Id(x))[0][0]
            D['loc'] = D.apply(locId, axis = 1)
            self.D = D

            def locId2location(locId):
                x = self.VALIDID[locId]
                lat = (x // (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL)) * self.DIVIDE_LEVEL + MIN_LAT + 0.5 * self.DIVIDE_LEVEL
                lon = (x % (1 + (MAX_LON - MIN_LON) // self.DIVIDE_LEVEL)) * self.DIVIDE_LEVEL + MIN_LON + 0.5 * self.DIVIDE_LEVEL
                return (lat, lon)
            self.GPS = np.array([locId2location(id) for id in range(len(self.VALIDID))])
            self.POI = np.zeros((self.VALIDID.shape[0], poi.shape[1]))
            for idx, coor in enumerate(gps):
                ID = location2Id(pd.DataFrame({'lon': [coor[0]], 'lat': [coor[1]]}))
                if ID in self.VALIDID:
                    self.POI[np.where(self.VALIDID == ID)[0][0]] += poi[idx]
        
        # Output
        Data = D[['usr', 'loc', 'tim', 'sta']]
        return Data

    def preprocess_duplicate(self, data):
        d = pd.DataFrame.from_dict(data)
        dup_start = (d['loc'] != d['loc'].shift(1)) & (d['loc'] == d['loc'].shift(-1))
        dup_end = (d['loc'] == d['loc'].shift(1)) & (d['loc'] != d['loc'].shift(-1))
        dup = (dup_start * dup_start.cumsum()) + (dup_end * dup_end.cumsum()).to_numpy()
        if sum(dup) == 0:
            return data
        for x in range(1, max(dup) + 1):
            slice = np.where(dup == x)[0]
            dup[slice[0]: slice[1] + 1] = x
        d['dup'] = dup
        def duplicate(x):
            if (x['dup'] == 0).sum():
                return x
            a = x.iloc[0:1]
            a['sta'] = x['sta'].sum()
            return pd.DataFrame(a)
        m = d.groupby('dup').apply(duplicate)
        m = m.reset_index(drop = True).sort_values(by=['tim'])
        return {'loc': np.array(m['loc']), 'tim': np.array(m['tim']), 'sta': np.array(m['sta'])}

    def preprocess_aggregate(self, data):
        d = pd.DataFrame.from_dict(data)
        if d.shape[0] == 0:
            return 
        agg = (d['sta'] < self.TIME_MIN)
        agg_start = (agg != agg.shift(1)) & (agg == agg.shift(-1)) & agg
        agg_end = (agg == agg.shift(1)) & (agg != agg.shift(-1)) & agg
        agg = (agg_start * agg_start.cumsum()) + (agg_end * agg_end.cumsum()).to_numpy()
        for x in range(1, max(agg) + 1):
            slice = np.where(agg == x)[0]
            agg[slice[0]: slice[1] + 1] = x
        d['agg'] = agg
        def aggregate(x):
            if (x['agg'] == 0).sum():
                return x
            a = x.groupby('loc').sum()
            a = a.reset_index()
            b = a[a['sta'] == a['sta'].max()].iloc[0:1]
            b['tim'] = x['tim'].min()
            b['sta'] = x['sta'].sum()
            return b
        m = d.groupby('agg').apply(aggregate).reset_index(drop = True)
        m = m[m['sta'] >= self.TIME_MIN]
        if m.shape[0] == 0:
            return
        m = m.sort_values(by='tim')
        n = m['sta'].iloc[-1]
        m['sta'] = (m['tim'].shift(-1) - m['tim']).fillna(n)
        return {'loc': np.array(m['loc']), 'tim': np.array(m['tim']), 'sta': np.array(m['sta'])}

    def preprocess_sparse(self, data, method='day'):
        d = pd.DataFrame.from_dict(data)
        d['cut'] = (d['sta'] >= self.TIME_MAX)
        d['cut'] = d['cut'].cumsum().shift(1).fillna(0)
        d['day'] = d['tim'] // 1440
        cut = {}
        def sparse(x):
            if x.shape[0] < self.MIN_LEN:
                return 
            tim = x['tim'].min()
            if x['sta'].iloc[-1] < self.TIME_MAX :
                cut[tim] = {'loc': np.array(x['loc']), 'tim': np.array(x['tim']), 'sta': np.array(x['sta'])}
            else:
                cut[tim] = {'loc': np.array(x['loc'])[:-1], 'tim': np.array(x['tim'])[:-1], 'sta': np.array(x['sta'])[:-1]}
        _ = d.groupby(method).apply(sparse)
        return cut

    def preprocess(self, D):
        D = D[D['sta'] != 0]
        
        data1 = {}
        def divide(x):
            user = x['usr'].min()
            time = x['tim'].min()
            data1[user] = {time: {'loc': np.array(x['loc'])[:-1], 'tim': np.array(x['tim'])[:-1], 'sta': np.array(x['sta'])[:-1]}}
        _ = D.groupby('usr').apply(divide)

        data2 = {}
        data = {}
        for usr in data1:
            data2[usr] = {}
            data[usr] = {}
            for tim in data1[usr]:
                data2[usr][tim] = self.preprocess_duplicate(data1[usr][tim])
                data2[usr][tim] = self.preprocess_aggregate(data2[usr][tim])
                if data2[usr][tim] == None:
                    continue
                data2[usr][tim] = self.preprocess_duplicate(data2[usr][tim])
                method = 'day' if self.NAME == 'ISP' else 'cut'
                data[usr].update(self.preprocess_sparse(data2[usr][tim], method))
        for usr in data:
            data[usr] = {idx: x for idx, x in enumerate(list(data[usr].values()))} 
        return data

    def attrprepare(self):

        self.USERLIST = np.array([usr for usr in self.DATA if len(self.DATA[usr]) > 0], dtype=int)

        if not self.EXIST:
            self.FILTEREDID = np.array([])
            for usr in self.USERLIST:
                for idx in self.DATA[usr]:
                    self.FILTEREDID = np.append(self.FILTEREDID, self.DATA[usr][idx]['loc'])
            self.FILTEREDID = np.unique(np.sort(self.FILTEREDID)).astype(int)

            self.GPS = self.GPS[np.ix_(self.FILTEREDID)]
            if self.NAME != 'GeoLife':
                self.POI = self.POI[np.ix_(self.FILTEREDID)]

            self.DATA = {usr: self.DATA[usr] for usr in self.USERLIST}
            for usr in self.USERLIST:
                for idx in self.DATA[usr]:
                    self.DATA[usr][idx]['loc'] = np.array([np.where(self.FILTEREDID == id)[0][0] for id in self.DATA[usr][idx]['loc']])
            
            
            np.save(self.PATH + 'DATA_' + self.LOCATION_MODE + '.npy', self.DATA)
            np.save(self.PATH + 'GPS_' + self.LOCATION_MODE + '.npy', self.GPS)
            if self.NAME != 'GeoLife':
                np.save(self.PATH + 'POI_' + self.LOCATION_MODE + '.npy', self.POI)

        self.IDX = np.cumsum([len(self.DATA[user]) for user in self.DATA])
        self.REFORM = {}
        self.GENDATA = []

        self.poi_size = self.POI.shape[1] if self.NAME != 'GeoLife' else 0
        self.usr_size = len(self.USERLIST)
        self.tim_size = 10080 if self.NAME == 'FourSquare' else 1440
        self.loc_size = self.GPS.shape[0]

        self.infer_maxlast = 10080
        self.infer_maxinternal = 1440
        self.infer_divide = 1440

    def loaddata(self):
        if os.path.exists(self.PATH + 'DATA_' + self.LOCATION_MODE + '.npy'):

            self.DATA = np.load(self.PATH + 'DATA_' + self.LOCATION_MODE + '.npy', allow_pickle=True).item()
            self.GPS = np.load(self.PATH + 'GPS_' + self.LOCATION_MODE + '.npy', allow_pickle=True)
            if self.NAME != 'GeoLife':
                self.POI = np.load(self.PATH + 'POI_' + self.LOCATION_MODE + '.npy', allow_pickle=True)
            else:
                self.POI = None
            
            return True
        return False

    def __getitem__(self, index):
        user = np.where(self.IDX > index)[0][0]
        traj = index - self.IDX[user - 1]  if user > 0 else index 
        userID = self.USERLIST[user]
        output = self.DATA[userID][traj]
        output['usr'] = userID * np.ones(output['sta'].shape[0], dtype=int)
        return output

    def __len__(self):
        return self.IDX[-1]

def getsets(data, train_prop, valid_prop):
    train_size = int(train_prop * len(data))
    valid_size = int(valid_prop * len(data))
    test_size = len(data) - train_size - valid_size
    return torch.utils.data.random_split(data, [train_size, valid_size, test_size])

def mycollatefunc(batch):
    output = {}
    for feature in batch[0]:
        output[feature] = []
        for data in batch:
            output[feature].append(torch.tensor(data[feature]))
        pad_val = -1 if feature == 'usr' else 0
        output[feature] = pad_sequence(output[feature], batch_first=True, padding_value=pad_val)
    return output

def reform(testset, name):
    test_data = {}
    for x in testset.indices:
        traj = testset.dataset[x]
        user = traj['usr'][0]
        if user not in test_data:
            test_data[user] = {0: {'loc': traj['loc'], 'tim': traj['tim'], 'sta': traj['sta']}}
        else:
            key = len(test_data[user])
            test_data[user][key] = {'loc': traj['loc'], 'tim': traj['tim'], 'sta': traj['sta']}
    testset.dataset.REFORM[name] = test_data

def timefixed(data):
    D = pd.DataFrame.from_dict(data)
    D['zon'] = D['tim'].apply(lambda x:int(np.round(x / 10, 0)))
    output = pd.DataFrame({'tim': np.arange(D['zon'].iloc[0] // 3, (D['tim'].iloc[D['tim'].shape[0] - 1] + D['sta'].iloc[D['sta'].shape[0] - 1]) // 30)})
    def corresponding_location(x):
        target = D[(D['zon'] >= 3 * x) & (D['zon'] < 3 * (x + 1))]
        last = D[D['zon'] < 3 * x]
        if target.shape[0] == 0:
            last = last.iloc[last.shape[0] - 1]
            l = last['loc']
        elif target.shape[0] == 1: 
            if last.shape[0] == 0:
                l = target['loc'].iloc[0]
            else:
                last = last.iloc[(last.shape[0] - 1): last.shape[0]]
                tim = target['zon'].iloc[0]
                locid = np.argmax([tim - 3 * x, 3 * (x + 1) - tim])
                target = pd.concat([last, target])
                l = target['loc'].iloc[locid]
        elif target.shape[0] == 2: 
            if target['zon'].iloc[1] < 3 * x + 2:
                l = target['loc'].iloc[1]
            else:
                l = target['loc'].iloc[0]
        else:
            l = target['loc'].iloc[1]
        return l
    output['loc'] = output['tim'].apply(corresponding_location)
    start = int(output['tim'].iloc[0] % 48)
    return {'internal': (start, start + output.shape[0] - 1),'loc': output['loc'].to_numpy()}

def ToTimeFixed(data, PATH, MODE):
    fixed = {}
    for user in data:
        fixed[user] = {}
        for traj in data[user]:
            fixed[user][traj] = timefixed(data[user][traj])
    np.save('./data/' + PATH + '/fixed_' + MODE + '.npy', fixed)
    return fixed


if __name__ == '__main__':
    
    ToTimeFixed(MYDATA('ISP', 0).DATA, 'ISP', '0')
    ToTimeFixed(MYDATA('ISP', 1).DATA, 'ISP', '1')
    ToTimeFixed(MYDATA('ISP', 2).DATA, 'ISP', '2')
    '''
    for i in range(3):
        data = MYDATA('ISP', i)
        def data_test(data, user, traj):
            D = pd.DataFrame.from_dict(data)
            if ((D['loc'] == D['loc'].shift(1)) | (D['loc'] == D['loc'].shift(-1))).sum():
                print(user, traj)
                print(D[(D['loc'] == D['loc'].shift(1)) | (D['loc'] == D['loc'].shift(-1))])
            if ((D['tim'].shift(-1) - D['tim'] - D['sta']).abs() > 1e-4).sum():
                print(user, traj)
                print(D[(D['tim'].shift(-1) - D['tim']) != D['sta']])
            if ((D['sta'] < 10) | (D['sta'] > 10080)).sum():
                print(user, traj)
                print(D[(D['sta'] < 10) | (D['sta'] > 10080)])
        for user in data.DATA:
            for traj in data.DATA[user]:
                data_test(data.DATA[user][traj], user, traj)
    '''