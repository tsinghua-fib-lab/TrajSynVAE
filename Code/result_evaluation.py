# -*- coding: utf-8 -*-
# !/usr/bin/python
"""
Created on Sep 15th 18:05:53 2020

Functions for result evaluation
"""
import numpy as np
import pandas as pd
import random

import scipy.stats
from sklearn.metrics import mean_squared_error
from collections import Counter

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import MultipleLocator, FormatStrFormatter



# Calculate the distance between two place given their longitude and latitude
def distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


# Calculate the travel distance of one trajectory
def travel_distance(traj, param):
    D = 0
    for i in range(len(traj['loc'])-1):
        loc1 = int(traj['loc'][i])
        loc2 = int(traj['loc'][i+1])
        D += distance(param.GPS[loc1][0], param.GPS[loc1][1], param.GPS[loc2][0], param.GPS[loc2][1])
    return D / np.sum(traj['sta'])


# Calculate the action radius of one trajectory
def radius(traj, param):
    if len(traj['loc']) == 0:
        return 0
    geo = []
    for i in traj['loc']:
        geo.append(param.GPS[int(i)])
    geo = np.array(geo)
    center = (np.mean(geo[:, 0]), np.mean(geo[:, 1]))
    D = 0
    for i in range(len(traj['loc'])):
        loc = int(traj['loc'][i])
        D = max(D, distance(param.GPS[loc][0], param.GPS[loc][1], center[0], center[1]))
    return D / (np.sum(traj['sta'])) # ** 2)


# Calculate locations visited by the user in one trajectory
def locations(traj, param):
    a = Counter(np.sort(traj['loc']))

    count = np.zeros(param.loc_size)
    for x in a:
        count[int(x)] += a[int(x)]
    '''
    count_time = np.zeros(param.loc_size)
    loctim =  pd.DataFrame.from_dict(traj).groupby('loc').sum().reset_index()
    for _, row in loctim.iterrows():
        count_time[int(row['loc'])] = row['sta']
    '''
    return len(list(a.keys())) / np.sum(traj['sta']), count



# Calculate the time intervals between events
def duration(traj):
    tim = np.array(traj['sta'])
    return tim

# Calculate the percentage of people that are shifting their position during the time slot
def transport_count(method, slot=60):
    count = np.zeros(1440 // slot, dtype=int)
    count_num = np.zeros(1440 // slot, dtype=int)
    for user in method:
        count_user = np.zeros(1440 // slot, dtype=int)
        for trajs in method[user]:
            traj = (method[user][trajs]['tim'] % 1440) // int(slot)
            internal = (method[user][trajs]['sta'])
            for j, i in enumerate(traj):
                count_user[int(i)] += internal[j]
                count_num[int(i)] += 1
        count = count + count_user
    count = np.array([(count[i] / count_num[i]) / 12 if count_num[i] != 0 else 0 for i in range(1440 // slot)])
    return count, count_num


# Calculate the JSD of two probability distributions
def JSD(p,q):
    M = (p+q)/2
    return 0.5 * scipy.stats.entropy(p, M) + 0.5 * scipy.stats.entropy(q, M)


# Calculate PDF, CDF, CCDF for the random variable given its probability distribution
def probability(count, internal=None, smooth=True):

    if internal != None:
        if smooth:
            length = (internal[1] - internal[0]) / 50
            a = Counter(np.sort((count - internal[0]) // length))
            x = np.arange(internal[0] + length, internal[1] + length, length)
            if len(x) > 50:
                x = x[:50]
            P = np.array([a[id] if id in a else 0 for id in range(50)])
        else:
            x = np.arange(internal[0], internal[1])
            a = Counter(np.sort(count))
            P = np.array([a[id] if id in a else 0 for id in x])

    else:
        unzip = np.array(list(Counter(np.sort(count)).item()))
        x = unzip[:, 0]
        P = unzip[:, 1]
    p = P / sum(P) if sum(P) > 0 else np.ones_like(P) / len(P)
    c = np.cumsum(p)
    cc = (1 - c)[:-1]   
    return x, p, c, cc


# Plot with a particular standard
def prob_plot(x, frequency, x_axis, y_axis, name1, name2, Log=False):
    plt.figure()
    if len(x) <= 50:
        ln1, = plt.plot(x, frequency['original'], color='red', linewidth=1, linestyle='-', marker="^", markersize=3)
        ln2, = plt.plot(x, frequency['method_1'], color='blue', linewidth=1, linestyle='-', marker="^", markersize=3)
        ln3, = plt.plot(x, frequency['method_2'], color='green', linewidth=1, linestyle='-', marker="^", markersize=3)
    else:
        ln1, = plt.plot(x, frequency['original'], color='red', linewidth=2, linestyle='-')
        ln2, = plt.plot(x, frequency['method_1'], color='blue', linewidth=2, linestyle='-')
        ln3, = plt.plot(x, frequency['method_2'], color='green', linewidth=2, linestyle='-')

    plt.xlabel(x_axis, size=20)
    plt.ylabel(y_axis, size=20)
    # tick = np.arange(min(x), max(x) + (max(x) - min(x)) / 3, (max(x) - min(x)) / 3)
    # if len(tick) > 4:
        # tick = tick[:4]
    # plt.xticks(tick, fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    if Log:
        ax.set_yscale("log")
    # else:
        # y = frequency['original']
        # tick = np.arange(min(y), max(y) + (max(y) - min(y)) / 3, (max(y) - min(y)) / 3)
        # if len(tick) > 4:
            # tick = tick[:4]
        # plt.yticks(tick)

    plt.legend(handles=[ln1, ln2, ln3], labels=['Original', name1, name2], fontsize=18, frameon=False)
    plt.tight_layout()


# Plot for several evaluation criteria
def evaluation_plot(param, type, method_1, method_2, original, name1, name2, mode = 'jsd'):

    Data = {'method_1': method_1, 'method_2': method_2, 'original': original}

    count = {}
    for method in ['method_1', 'method_2', 'original']:
        count[method] = np.array([])
        for user in Data[method]:
            for Traj in Data[method][user]:
                if type == 'Distance':
                    D = travel_distance(Data[method][user][Traj], param)
                    name = 'Distance (km/h)'
                if type == 'Radius':
                    D = radius(Data[method][user][Traj], param)
                    name = 'Radius (km/h)'
                if type == 'LocVisit':
                    D, _ = locations(Data[method][user][Traj], param)
                    name = 'Locations (N/h)'
                if type == 'Duration':
                    D = duration(Data[method][user][Traj])
                    name = 'Duration (min)'
                count[method] = np.append(count[method], [D])
        count[method] = 60 * count[method][count[method] > 0] if type != 'Duration' else count[method][count[method] > 0]

    unit = (np.percentile(count['original'], 95) - min(count['original'])) / 200
    internal = (min(count['original']) - unit, np.percentile(count['original'], 95) + unit)

    frequency = {'PDF': {}, 'CDF': {}, 'CCDF': {}}
    X = {}
    for method in count:
        x, p, c, cc = probability(count[method], internal)
        X[method] = x
        frequency['PDF'][method] = p
        frequency['CDF'][method] = c
        frequency['CCDF'][method] = cc

    jsd = [JSD(frequency['PDF']['method_1'], frequency['PDF']['original']), JSD(frequency['PDF']['method_2'], frequency['PDF']['original'])]

    if mode == 'plot':
        for mode in ['PDF', 'CDF', 'CCDF']:
            Log = True if mode == 'CCDF' else False
            xx = x[:-1] if mode == 'CCDF' else x
            prob_plot(xx, frequency[mode], name, mode, name1, name2, Log)
            plt.savefig(param.save_path + '/plots' +  '/' + type + '_' + mode+ '.png')
            plt.savefig(param.save_path + '/plots' +  '/' + type + '_' + mode+ '.eps')

    return jsd


# Plot the evaluation of location rank
def location_rank_plot(param, method_1, method_2, original, name1, name2, mode='jsd', user=True):
    Data = {'method_1': method_1, 'method_2': method_2, 'original': original}

    count = {}
    count_user = {user:{} for user in Data['original']}
    for method in ['method_1', 'method_2', 'original']:
        count[method] = np.zeros(param.loc_size)
        for usr in Data['original']:
            count_user[usr][method] = np.zeros(param.loc_size)
            if usr in Data[method]:
                for traj in Data[method][usr]:
                    _, D = locations(Data[method][usr][traj], param)
                    count[method] += D
                    count_user[usr][method] += D
    internal = (0, param.loc_size)
    x = np.arange(internal[0], internal[1])
    def toprob(x):
        if sum(x) == 0:
            return np.ones_like(x) / len(x)
        return x / sum(x)
    PDF = {}
    PDF = {method: toprob(count[method]) for method in count}
    jsd_G = [JSD(PDF['method_1'], PDF['original']), JSD(PDF['method_2'], PDF['original'])]


    PDF_user = {user:{} for user in Data['original']}
    jsd_user = {user:[] for user in Data['original']}
    for user in count_user:
        PDF_user[user] = {method: toprob(count_user[user][method]) for method in count_user[user]}
        jsd_user[user] = [JSD(PDF_user[user]['method_1'], PDF_user[user]['original']), JSD(PDF_user[user]['method_2'], PDF_user[user]['original'])]
    jsd_I = [np.mean([jsd_user[user][0] for user in jsd_user]), np.mean([jsd_user[user][1] for user in jsd_user])]

    if mode == 'plot':
        frequency = {}

        unzip = np.array(sorted(np.array([x, PDF['original']]).T, key=lambda x: x[1], reverse=True))
        Len = min(50, np.argwhere(unzip[:, 1] == 0)[0][0])
        X = unzip[:Len, 0]
        x = np.arange(Len)
        frequency['original'] = unzip[:Len, 1]

        for method in ['method_1', 'method_2']:
            frequency[method] = np.array([PDF[method][int(id)] for id in X])

        plot_type = 'G-rank' 
        prob_plot(x, frequency, plot_type, 'P', name1, name2)
        plt.savefig(param.save_path + '/plots' + '/' + plot_type + '.png')
        plt.savefig(param.save_path + '/plots' + '/' + plot_type + '.eps')
    return jsd_G, jsd_I


# Plot the average waiting time of records having different beginnings
def transport_count_plot(param, method_1, method_2, original, slot, name1, name2, mode='jsd'):
    count_1, num1 = transport_count(method_1, slot)
    count_2, num2 = transport_count(method_2, slot)
    count_3, num3 = transport_count(original, slot)
    def toprob(x):
        if sum(x) == 0:
            return np.ones_like(x) / len(x)
        return x / sum(x)
    count_1 = toprob(count_1)
    count_2 = toprob(count_2)
    count_3 = toprob(count_3)
    num1 = toprob(num1)
    num2 = toprob(num2)
    num3 = toprob(num3)

    if mode == 'plot':
        plt.figure()
        x = np.arange(0, 24, slot / 60)
        ln1, = plt.plot(x, count_3, color='red', linewidth=1, linestyle='-', marker='^', markersize=3)
        ln2, = plt.plot(x, count_1, color='blue', linewidth=1, linestyle='-', marker='^', markersize=3)
        ln3, = plt.plot(x, count_2, color='green', linewidth=1, linestyle='-', marker='^', markersize=3)

        plt.xlabel('Time(hour)', size=20)
        plt.ylabel('Average Waiting Time(Normalized)', size=15)
        plt.xticks(np.arange(0, 30, 6), fontsize=15)
        plt.yticks(np.arange(0, 0.1, 0.02), fontsize=15)

        plt.legend(handles=[ln1, ln2, ln3], labels=['Original', name1, name2], fontsize=18, frameon=False)
        plt.tight_layout()

        plt.savefig(param.save_path + '/plots' + '/stay_' + str(slot) + '.png')
        plt.savefig(param.save_path + '/plots' + '/stay_' + str(slot) + '.eps')

        plt.figure()
        ln1, = plt.plot(x, num3, color='red', linewidth=1, linestyle='-', marker='^', markersize=3)
        ln2, = plt.plot(x, num1, color='blue', linewidth=1, linestyle='-', marker='^', markersize=3)
        ln3, = plt.plot(x, num2, color='green', linewidth=1, linestyle='-', marker='^', markersize=3)

        plt.xlabel('Time(hour)', size=20)
        plt.ylabel('P', size=15)
        plt.xticks(np.arange(0, 30, 6), fontsize=15)
        plt.yticks(np.arange(0, 0.15, 0.05), fontsize=15)

        plt.legend(handles=[ln1, ln2, ln3], labels=['Original', name1, name2], fontsize=18, frameon=False)
        plt.tight_layout()

        plt.savefig(param.save_path + '/plots' + '/move_' + str(slot) + '.png')
        plt.savefig(param.save_path + '/plots' + '/move_' + str(slot) + '.eps')

    jsd = [JSD(num1, num3), JSD(num2, num3)]
    mse = [100 * mean_squared_error(count_1, count_3), 100 * mean_squared_error(count_2, count_3)]
    return jsd, mse


# For a easier call
def EVALUATION(param, method_1, method_2, original, name1, name2, mode = 'jsd'):
    jsd = {}
    jsd['move'], jsd['stay'] = transport_count_plot(param, method_1, method_2, original, 60, name1, name2, mode=mode)
    jsd['travel_distance'] = evaluation_plot(param, 'Distance', method_1, method_2, original, name1, name2, mode)
    jsd['radius'] = evaluation_plot(param, 'Radius', method_1, method_2, original, name1, name2, mode)
    jsd['duration'] = evaluation_plot(param, 'Duration', method_1, method_2, original, name1, name2, mode)
    jsd['G_rank'], jsd['I_rank'] = location_rank_plot(param, method_1, method_2, original, name1, name2, mode)
    jsd = pd.DataFrame(jsd, index=[name1, name2])
    if mode == 'plot':
        print(jsd)
    return jsd


''' SSPD
def Distance_Point(x, y, X, Y):
    if len(X) == 0:
        return 0
    if len(X) == 1:
        return ((X - x)**2 + (Y - y)**2) ** 0.5
    x_1, y_1, x_0, y_0 = X[1:], Y[1:], X[:-1], Y[:-1]
    dominator = ((y_1 - y_0)**2 + (x_1 - x_0)**2)**0.5
    x_1, y_1, x_0, y_0 = x_1[np.ix_(dominator > 0)], y_1[np.ix_(dominator > 0)], x_0[np.ix_(dominator > 0)], y_0[np.ix_(dominator > 0)]
    if len(x_1) == 0:
        print(X, Y)
    # print(x_1, y_1, x_0, y_0)
    dominator = dominator[np.ix_(dominator > 0)]
    judge = (((y_1 - y_0) * (y - y_1) + (x_1 - x_0) * (x - x_1)) * ((y_1 - y_0) * (y - y_0) + (x_1 - x_0) * (x - x_0))) <= 0
    distance = np.append(((x_0 - x)**2 + (y_0 - y)**2) ** 0.5, ((x_1[-1] - x)**2 + (y_1[-1] - y)**2) ** 0.5)
    distance = np.array([min(distance[i], distance[i + 1]) for i in range(len(distance) - 1)])
    Distance = np.abs((y_1 - y_0) * (x - x_0) - (x_1 - x_0) * (y - y_0)) / dominator
    return Distance * judge + distance * (1 - judge)


def SSPD(traj1, traj2, GPS):
    xymin = np.array([np.min(GPS[:, 0]), np.min(GPS[:, 1])])
    P1, P2 = 100 * (GPS[np.ix_(traj1['loc'].astype(int))] - xymin), 100 * (GPS[np.ix_(traj2['loc'].astype(int))] - xymin)

    SPD1 = np.mean(np.array([Distance_Point(P1[i][0], P1[i][1], P2[:, 0], P2[:, 1]) for i in range(P1.shape[0])]))
    SPD2 = np.mean(np.array([Distance_Point(P2[i][0], P2[i][1], P1[:, 0], P1[:, 1]) for i in range(P2.shape[0])]))
    return (SPD1 + SPD2) / 2
    

def overlapping_ratio(modeldata, original, param, name, plot=True):
    minSSPD = []
    avgSSPD = []
    maxSSPD = []
    for user in original:
        if user not in modeldata:
            continue
        count = np.array([min([SSPD(traj1, traj2, param.GPS) for traj2 in original[user].values()]) for traj1 in modeldata[user].values()])
        count = count[count <= 30]
        if len(count) > 0:
            minSSPD.append(np.min(count))
            avgSSPD.append(np.mean(count))
            maxSSPD.append(np.max(count))
        else:
            minSSPD.append(30)
            avgSSPD.append(30)
            maxSSPD.append(30)

    if plot == False:
        return minSSPD, avgSSPD, maxSSPD

    x_1, p_1, c_1, cc_1 = probability(np.array(minSSPD), internal=(0, 30))
    x_2, p_2, c_2, cc_2 = probability(np.array(avgSSPD), internal=(0, 30))
    x_3, p_3, c_3, cc_3 = probability(np.array(maxSSPD), internal=(0, 30))

    plt.figure()
    ln1, = plt.plot(x_1, c_1, color='red', linewidth=1, linestyle='-', marker="^", markersize=3)
    ln2, = plt.plot(x_1, c_2, color='blue', linewidth=1, linestyle='-', marker="^", markersize=3)
    ln3, = plt.plot(x_1, c_3, color='green', linewidth=1, linestyle='-', marker="^", markersize=3)

    plt.xlabel('SSPD', size=20)
    plt.ylabel('CDF', size=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))

    plt.legend(handles=[ln1, ln2, ln3], labels=['MinSSPD', 'AvgSSPD', 'MaxSSPD'], fontsize=18, frameon=False)
    plt.tight_layout()
    plt.savefig('./SSPD_' + name + '.eps', dpi=300)
    plt.savefig('./SSPD_' + name + '.png')
    return np.mean(minSSPD), np.mean(avgSSPD), np.mean(maxSSPD)
'''


'''Individual Analysis
def Individual_Traj_plot(param, output_sequence):
    for user in output_sequence:
        for trajnum in output_sequence[user]:
            if len(output_sequence[user][trajnum]['loc']) < 6:
                print(user, trajnum)
                traj = output_sequence[user][trajnum]

                mpl.rcParams['legend.fontsize'] = 10
                fig = plt.figure()
                ax = fig.gca(projection='3d')

                x = [param.GPS[traj['loc'][-1]][0], param.GPS[traj['loc'][-1]][0]]
                y = [param.GPS[traj['loc'][-1]][1], param.GPS[traj['loc'][-1]][1]]
                z = [0]
                for i, loc in enumerate(traj['loc']):
                    x.append(param.GPS[loc][0])
                    y.append(param.GPS[loc][1])
                    x.append(param.GPS[loc][0])
                    y.append(param.GPS[loc][1])

                for i in range(len(traj['tim'])):
                    z.append(traj['tim'][i])
                    z.append(traj['tim'][i])
                z.append(1440)

                ax.plot(x, y, np.array(z) / 100, color='red')
                ax.scatter(x, y, np.array(z) / 100, color='red', s=10)
                ax.xaxis.set_major_locator(MultipleLocator(max((max(x) - min(x)) / 4, 0.001)))
                ax.yaxis.set_major_locator(MultipleLocator(max((max(y) - min(y)) / 4, 0.01)))
                ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
                ax.set_zlim(0, 24)
                ax.zaxis.set_major_locator(MultipleLocator(6))
                plt.grid(color='whitesmoke')
                plt.show()
'''


''' Overlapping Ratio & Population Analysis ***Copied from Version 1.0, revision needed***
def to_fixed(data, location = None):
    d = pd.DataFrame.from_dict(data)
    d['tim'] = d['tim'] % 1440

    if location == None:
        d.loc[0, 'sta'] += d.iloc[0, 1]
        d.loc[0, 'tim'] = 0
    else: 
        d = pd.concat([pd.DataFrame({'loc': [location], 'tim': [0], 'sta': [d.iloc[0, 1]]}), d])
    d.iloc[-1, 2] = 1440 - d.iloc[-1, 1]

    fix = []
    for i in np.arange(0, 1440, 30):
        dslot = pd.DataFrame(d[(d['tim'] < (i + 30)) & ((d['tim'] + d['sta']) > i)])
        
        if dslot.shape[0] == 1:
            fix.append(dslot.iloc[0, 0])
        else:
            dslot.loc[0, 'sta'] = dslot.iloc[0, 1] + dslot.iloc[0, 2] - i
            dslot.loc[-1, 'sta'] = i + 30 - dslot.iloc[-1, 1]
            fix.append(dslot[dslot['sta'] == dslot['sta'].max()].iloc[0, 0])
    return np.array(fix, dtype=int)


# Calculate the time interval given two timestamp
def time_interval(tim1, tim2):
    Tim1 = (tim1 // 10000) * 1440 + ((tim1 % 10000) // 100) * 60 + (tim1 % 100)
    Tim2 = (tim2 // 10000) * 1440 + ((tim2 % 10000) // 100) * 60 + (tim2 % 100)
    return Tim2 - Tim1


# Process trajectories to ones that have a fixed time slot for every record point
def Overlapping_ratio_count_prepare(location, traj, param):

    loc = np.concatenate(([location],traj['loc']))
    tim = np.concatenate(([0], (traj['tim'] % 1000000) // 100))
    if param.data_type == 'ISP':
        Tim = tim_detrans(np.arange(0, 49) * 6)//100
    else:
        Tim = tim_detrans(np.arange(0, 49) * 30)//100

    count = {}
    A = 48
    b = 30
    for i in range(A):
        count[i] = []

    K = [min([i for i, Time in enumerate(Tim) if time_interval(time, Time) > 0]) for time in tim]
    for i, key in enumerate(K):

        if key == K[-1]:
            for J in range(key, A):
                count[J].append((loc[i], b))
            count[key-1].append((loc[i], time_interval(tim[i], Tim[key])))

        else:
            Key = K[i+1]
            if Key > key:

                for J in range(key, Key-1):
                    count[J].append((loc[i], b))
                a = np.array(count[key-1])
                a = a[:, 0] if len(a) != 0 else np.array([])

                if loc[i] in a and len(a) != 0:
                    I = [x for x in range(len(a)) if a[x] == loc[i]]
                    I = I[0]
                    count[key-1][I] = (loc[i], count[key-1][I][1] + time_interval(tim[i], Tim[key]))
                else:
                    count[key-1].append((loc[i], time_interval(tim[i], Tim[key])))
                count[Key-1].append((loc[i], time_interval(Tim[Key-1], tim[i+1])))

            else:

                a = np.array(count[key-1])
                a = a[:, 0] if len(a) != 0 else np.array([])

                if loc[i] in a and len(a) != 0:
                    I = [x for x in range(len(a)) if a[x] == loc[i]]
                    I = I[0]
                    count[key-1][I] = (loc[i], count[key-1][I][1] + time_interval(tim[i], tim[i+1]))
                else:
                    count[key-1].append((loc[i], time_interval(tim[i], tim[i+1])))

    split = []
    locs = {}
    c = 24
    for i in range(c):
        locs[i] = np.array([])

    for i in range(A):

        count[i] = sorted(count[i], key=lambda x: x[1], reverse=True)
        count[i] = np.array(count[i])
        split.append(count[i][0][0])

        Locs = [x[0] for x in count[i] if x[1] != 0]
        d = i//2
        locs[d] = np.append(locs[d], Locs)
        locs[d] = np.array(list(set(locs[d]))).astype(np.int)

    return split, locs


# Calculate the overlapping ratio of trajectories generated by our model and the original ones
def Overlapping_ratio_count(location, target, trajs):
    Target, _ = Overlapping_ratio_count_prepare(location, target)
    Overlapping_ratio = []
    for traj in trajs:
        loc = trajs[traj]['loc'][0] if traj == 0 else trajs[traj-1]['loc'][-1]
        Traj = to_fixed(trajs[traj], loc)
        A = 48
        Overlapping_ratio.append(sum([1 for x in range(A) if Target[x] == Traj[x]]))
    Overlapping_ratio = sorted(Overlapping_ratio, reverse=True)
    return Overlapping_ratio


# Plot the overlapping ratio between generated trajectories and real ones
def Overlapping_ratio_plot(method, original, param):

    top = {1: [], 2: [], 3: []}
    for user in method:
        k = 0
        for traj in method[user]:
            if k > 10:
                continue
            target = method[user][traj]
            location = method[user][traj]['loc'][0]
            trajs = original[user]
            ratio = Overlapping_ratio_count(location, target, trajs)
            if len(ratio) < 3:
                zeros = np.zeros(3 - len(ratio))
                ratio = np.concatenate((ratio, zeros))
            top[1].append(ratio[0])
            top[2].append(ratio[1])
            top[3].append(ratio[2])
            k += 1

    P1 = probability(top, 1, 'CDF', smooth=True)
    P2 = probability(top, 2, 'CDF', smooth=True)
    P3 = probability(top, 3, 'CDF', smooth=True)

    x = np.arange(51) / 50
    zeros_1 = np.ones(len(x) - len(P1))
    P1 = np.concatenate((P1, zeros_1))
    zeros_2 = np.ones(len(x) - len(P2))
    P2 = np.concatenate((P2, zeros_2))
    zeros_3 = np.ones(len(x) - len(P3))
    P3 = np.concatenate((P3, zeros_3))

    plt.figure()
    x_max = 1
    max_index = 0
    for i, j in enumerate(x):
        if P1[i] > 0.99:
            x_max = j
            max_index = i
            break
    ln1, = plt.plot(x[: max_index + 1], P1[: max_index + 1], color='red', linewidth=1, linestyle='-', marker="^",
                    markersize=4)
    ln2, = plt.plot(x[: max_index + 1], P2[: max_index + 1], color='blue', linewidth=1, linestyle='-', marker="^",
                    markersize=4)
    ln3, = plt.plot(x[: max_index + 1], P3[: max_index + 1], color='green', linewidth=1, linestyle='-', marker="^",
                    markersize=4)

    plt.xlabel('Overlapping Ratio', size=18)
    plt.ylabel('CDF', size=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(0.25))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    ax.set_ylim(0, 1.01)
    ax.set_xlim(-0.01, x_max)

    plt.legend(handles=[ln1, ln2, ln3], labels=['Top1', 'Top2', 'Top3'], fontsize=20, frameon=False)

    plt.tight_layout()
    plt.savefig(param.save_path + '/plots' + '/Overlapping_ratio.png')


# Calculate the change of population of one particular place
def population_count(Data, param):
    c = 24
    population = {}
    for location in range(param.loc_size):
        population[location] = np.zeros(c)
    for user in Data:
        population_user = {}
        for location in range(param.loc_size):
            population_user[location] = np.zeros(c)
        for traj in Data[user]:
            _, count = Overlapping_ratio_count_prepare(Data[user][traj]['loc'][0], Data[user][traj])
            for i in range(c):
                for location in range(param.loc_size):
                    if location in count[i]:
                        population_user[location][i] += 1
        for location in range(param.loc_size):
            population[location] += population_user[location] / len(Data[user])
    return population


# Plot the change of population of several places for further analysis
def population_plot(method, input_original):

    ours = population_count(method)
    original = population_count(input_original)
    for location in ours:
        if JSD(ours[location] / max(ours[location]), original[location] / max(original[location])) < 0.01:
            location_plot(ours[location] / max(ours[location]), original[location])

'''