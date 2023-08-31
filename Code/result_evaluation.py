"""
Created on Sep 15th 18:05:53 2020

Author: Qizhong Zhang

Functions for evaluation
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, FormatStrFormatter

'''
PART 1 - USABILITY EVALUATION: Distance, Radius, G-Rank, Duration, Move, Stay
'''

# Calculate the distance between two place given their longitude and latitude
def distance(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371
    return c * r


# Calculate the travel distance of one trajectory
def travel_distance(traj, param):
    if len(traj['loc']) == 0:
        return 0
    geo = param.GPS[np.ix_(np.array(traj['loc']).astype(int))]
    return np.sum(distance(geo[:-1, 0], geo[:-1, 1], geo[1:, 0], geo[1:, 1]))


# Calculate the action radius of one trajectory
def radius(traj, param):
    if len(traj['loc']) == 0:
        return 0
    geo = param.GPS[np.ix_(np.array(traj['loc']).astype(int))]
    center = np.mean(geo, axis = 0)
    return np.sqrt(np.mean(distance(geo[:, 0], geo[:, 1], center[0], center[1])))


# Calculate locations visited by the user in one trajectory
def locations(traj, param):
    count = np.bincount(traj['loc'].astype(int))
    return np.append(count, np.zeros(param.loc_size - count.shape[0]))


# Calculate the time intervals between events
def duration(traj):
    return np.array(traj['sta'])


# Calculate the percentage of people that are shifting their position during the time slot as well as the time people stay at new places
def transport_count(method, slot=60):
    count = [[] for x in range(1440 // slot)]
    count_num = np.zeros(1440 // slot, dtype=int)
    for user in method:
        for trajs in method[user]:
            traj = (method[user][trajs]['tim'] % 1440) // int(slot)
            interval = (method[user][trajs]['sta'])
            for j, i in enumerate(traj):
                count[int(i)].append(interval[j])
                count_num[int(i)] += 1
    return count, count_num, [np.mean(c) for c in count] 


# Calculate the JSD of two probability distributions
def JSD(P_A, P_B):
    epsilon = 1e-8
    P_A = P_A + epsilon
    P_B = P_B + epsilon
    P_merged = 0.5 * (P_A + P_B)
    
    kl_PA_PM = np.sum(P_A * np.log(P_A / P_merged))
    kl_PB_PM = np.sum(P_B * np.log(P_B / P_merged))
    
    jsd = 0.5 * (kl_PA_PM + kl_PB_PM)
    return jsd


# Calculate PDF, CDF, CCDF for the random variable given its probability distribution
def probability(count, interval=None):
    p, x = np.histogram(count, bins = 50, range = interval)
    x = (x[:-1] + x[1:]) / 2
    p = p / sum(p) if sum(p) > 0 else np.ones_like(p) / len(p)
    c = np.cumsum(p)
    cc = np.append(1, (1 - c)[:-1])   
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
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    if Log:
        ax.set_yscale("log")

    plt.legend(handles=[ln1, ln2, ln3], labels=['Original', name1, name2], fontsize=18, frameon=False)
    plt.tight_layout()


# Elementary evaluation criteria plotting
def evaluation_plot(param, type, method_1, method_2, original, name1, name2, mode = 'jsd'):

    Data, count = {'method_1': method_1, 'method_2': method_2, 'original': original}, {}
    for method in ['method_1', 'method_2', 'original']:
        count[method] = np.array([])
        for user in Data['original']:
            if user in Data[method]:
                for Traj in Data[method][user]:
                    if type == 'Distance':
                        D = travel_distance(Data[method][user][Traj], param)
                        name = 'Distance (km)'
                    if type == 'Radius':
                        D = radius(Data[method][user][Traj], param)
                        name = 'Radius (km)'
                    if type == 'Duration':
                        D = duration(Data[method][user][Traj])
                        name = 'Duration (min)'
                    count[method] = np.append(count[method], [D])
        count[method] = count[method][count[method] > 0]

    unit = (np.percentile(count['original'], 95) - min(count['original'])) / 200
    interval = (min(count['original']) - unit, np.percentile(count['original'], 95) + unit)

    frequency, X = {'PDF': {}, 'CDF': {}, 'CCDF': {}}, {}
    for method in count:
        x, p, c, cc = probability(count[method], interval)
        X[method] = x
        frequency['PDF'][method] = p
        frequency['CDF'][method] = c
        frequency['CCDF'][method] = cc

    jsd = [JSD(frequency['PDF']['method_1'], frequency['PDF']['original']), JSD(frequency['PDF']['method_2'], frequency['PDF']['original'])]

    if mode == 'plot':
        for mode in ['PDF', 'CDF', 'CCDF']:
            Log = True if mode == 'CCDF' else False
            prob_plot(x, frequency[mode], name, mode, name1, name2, Log)
            plt.savefig(param.save_path + '/plots' +  '/' + type + '_' + mode+ '.png')
            plt.savefig(param.save_path + '/plots' +  '/' + type + '_' + mode+ '.eps')
    return jsd


# Global location rank plotting
def location_rank_plot(param, method_1, method_2, original, name1, name2, mode='jsd', user=True):
    Data, count = {'method_1': method_1, 'method_2': method_2, 'original': original}, {}
    for method in ['method_1', 'method_2', 'original']:
        count[method] = np.zeros(param.loc_size)
        for usr in Data['original']:
            if usr in Data[method]:
                for traj in Data[method][usr]:
                    D = locations(Data[method][usr][traj], param)
                    count[method] += D
    toprob = lambda x: np.ones_like(x) / len(x) if sum(x) == 0 else x / sum(x)
    PDF = {method: toprob(count[method]) for method in count}
    jsd_G = [JSD(PDF['method_1'], PDF['original']), JSD(PDF['method_2'], PDF['original'])]

    if mode == 'plot':
        unzip = np.array(sorted(np.array([np.arange(param.loc_size), PDF['original']]).T, key=lambda x: x[1], reverse=True))
        Len = min(50, np.argwhere(unzip[:, 1] == 0)[0][0])
        frequency = {'original': unzip[:Len, 1]}

        for method in ['method_1', 'method_2']:
            frequency[method] = np.array([PDF[method][int(id)] for id in unzip[:Len, 0]])

        prob_plot(np.arange(Len), frequency, 'G-rank', 'P', name1, name2)
        plt.savefig(param.save_path + '/plots/G-rank.png')
        plt.savefig(param.save_path + '/plots/G-rank.eps')

    return jsd_G


# Average waiting time and movement count plotting
def transport_count_plot(param, method_1, method_2, original, slot, name1, name2, mode='jsd'):

    user_filter = lambda x:{usr: x[usr] for usr in original if usr in x}
    count1, num1, count_1 = transport_count(user_filter(method_1), slot)
    count2, num2, count_2 = transport_count(user_filter(method_2), slot)
    count3, num3, count_3 = transport_count(original, slot)

    toprob = lambda x: np.ones_like(x) / np.cumprod(x.shape)[-1] if np.sum(x) == 0 else x / np.sum(x)
    count_1, count_2, count_3 = toprob(count_1), toprob(count_2), toprob(count_3)
    num1, num2, num3= toprob(num1), toprob(num2), toprob(num3)    

    count = np.concatenate(count3)
    unit = (np.percentile(count, 95) - min(count)) / 200
    interval = (min(count) - unit, np.percentile(count, 95) + unit)
    P = toprob(np.array([np.histogram(c, bins = 50, range = interval)[0] for c in count3]).astype(float))
    Q1 = toprob(np.array([np.histogram(c, bins = 50, range = interval)[0] for c in count1]).astype(float))
    Q2 = toprob(np.array([np.histogram(c, bins = 50, range = interval)[0] for c in count2]).astype(float))

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
    return [JSD(num1, num3), JSD(num2, num3)], [JSD(P, Q1), JSD(P, Q2)]


# CALL THIS FUNCTION TO EVALUATE GENERATED SEQUENCES
def EVALUATION(param, method_1, method_2, original, name1, name2, mode = 'jsd'):
    jsd = {}
    jsd['travel_distance'] = evaluation_plot(param, 'Distance', method_1, method_2, original, name1, name2, mode)
    jsd['radius'] = evaluation_plot(param, 'Radius', method_1, method_2, original, name1, name2, mode)
    jsd['G_rank']= location_rank_plot(param, method_1, method_2, original, name1, name2, mode)
    jsd['duration'] = evaluation_plot(param, 'Duration', method_1, method_2, original, name1, name2, mode)
    jsd['move'], jsd['stay'] = transport_count_plot(param, method_1, method_2, original, 60, name1, name2, mode)
    jsd = pd.DataFrame(jsd, index=[name1, name2])
    print(jsd)
    return jsd

'''
PART 2 - UNIQUENESS EVALUATION: SSPD, Overlapping Ratio, Data Quality
'''

# Convert latitude-longitude coordinates to cartesian cooridinates
def latlon_to_easting_northing(latitude, longitude, origin_latitude, origin_longitude):
    # Convert latitude and longitude from degrees to radians
    latitude_rad = np.radians(latitude)
    longitude_rad = np.radians(longitude)
    origin_latitude_rad = np.radians(origin_latitude)
    origin_longitude_rad = np.radians(origin_longitude)

    # Calculate the differences in latitude and longitude
    delta_latitude = latitude_rad - origin_latitude_rad
    delta_longitude = longitude_rad - origin_longitude_rad

    # Calculate the easting and northing
    radius_earth = 6371  # Radius of Earth in kilometers
    easting = radius_earth * delta_longitude * np.cos(origin_latitude_rad)
    northing = radius_earth * delta_latitude

    return easting, northing

# Distance between trajectories
def Distance_Point(x, y, X, Y):
    if len(X) == 0:
        return 0
    if len(X) == 1:
        xx, yy = latlon_to_easting_northing(X, Y, x, y)
        return np.sqrt(xx**2 + yy**2)
    xx, yy = latlon_to_easting_northing(X, Y, x, y)
    x_1, y_1, x_0, y_0 = xx[1:], yy[1:], xx[:-1], yy[:-1]
    dominator = ((y_1 - y_0)**2 + (x_1 - x_0)**2)**0.5
    Distance = np.abs(x_1 * y_0 - y_1 * x_0) / dominator
    judge = (((y_1 - y_0) * y_1 + (x_1 - x_0) * x_1) * ((y_1 - y_0) * y_0 + (x_1 - x_0) * x_0)) <= 0
    pjt_dist = np.min(Distance[judge]) if len(Distance[judge]) > 0 else np.inf
    ptw_dist = np.min(np.sqrt(xx**2 + yy**2))
    return min(pjt_dist, ptw_dist)


# SSPD Calculation
def SSPD(traj1, traj2, GPS):
    P1, P2 = np.array(GPS[np.ix_(traj1['loc'].astype(int))]), np.array(GPS[np.ix_(traj2['loc'].astype(int))])
    SPD1 = np.mean(np.array([Distance_Point(P1[i][0], P1[i][1], P2[:, 0], P2[:, 1]) for i in range(P1.shape[0])]))
    SPD2 = np.mean(np.array([Distance_Point(P2[i][0], P2[i][1], P1[:, 0], P1[:, 1]) for i in range(P2.shape[0])]))
    return (SPD1 + SPD2) / 2


# Plot the CDF of SSPD counts
def SSPD_plot(modeldata, original, param, name, plot=True):
    minSSPD, avgSSPD, maxSSPD = [], [], []
    for user in original:
        if user not in modeldata:
            continue
        count = np.array([min([SSPD(traj1, traj2, param.GPS) for traj2 in modeldata[user].values()]) for traj1 in original[user].values()])
        minSSPD.append(np.min(count))
        avgSSPD.append(np.mean(count))
        maxSSPD.append(np.max(count))

    if plot == False:
        return minSSPD, avgSSPD, maxSSPD
    M = np.ceil(np.percentile(np.concatenate([minSSPD, avgSSPD, maxSSPD]), 99))
    x_1, _, c_1, _ = probability(np.array(minSSPD), interval=(0, M))
    _, _, c_2, _ = probability(np.array(avgSSPD), interval=(0, M))
    _, _, c_3, _ = probability(np.array(maxSSPD), interval=(0, M))

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
    return minSSPD, avgSSPD, maxSSPD


# Discretize trajctories to 1 records / min, 1440 mins(1 day) in total
def padding(traj, tim_size):
    def intcount(seq):
        a, b = np.array(seq[:-1]), np.array(seq[1:])
        return (a == a.astype(int)) + np.ceil(b) - np.floor(a) - 1
    locs = np.concatenate(([-1], traj['loc'], [-1]))
    tims = np.concatenate(([0], traj['tim'] % tim_size, [tim_size]))
    tims[-2] = tims[-1] if (tims[-2] < tims[-3]) else tims[-2]
    return np.concatenate([[locs[id]] * int(n) for id, n in enumerate(intcount(tims))]).astype(int)


# Discretize trajctories to (1/slot) records / min
def fixed(pad_traj, slot = 30):
    return np.array([np.argmax(np.bincount((pad_traj + 1)[(slot*i):(slot*i+slot)])) - 1 for i in range(int(len(pad_traj)/slot))])


# Calculate the overlapping ratio given two sets of trajectories
def overlap_count(compa, contr):
    def overlap(a, b):
        return np.sort(np.sum(a == b, axis=1) / b.shape[0])[::-1]
    return np.array([overlap(compa, i) for i in contr])


# Plot the overlapping ratio between generated trajectories and real ones
def Overlapping_ratio_plot(method, original, tim_size, name = None):

    top = {1: [], 2: [], 3: []}
    for user in original:
        if user not in method:
            continue
        compa = np.array([padding(method[user][traj], tim_size) for traj in method[user]])
        contr = np.array([padding(original[user][traj], tim_size) for traj in original[user]])
        
        result = overlap_count(compa, contr)
        for i in range(3):
            top[i + 1].append(np.mean(result[:, i]))
    x, _, c1, _ = probability(top[1], interval=[0, 1])
    x, _, c2, _ = probability(top[2], interval=[0, 1])
    x, _, c3, _ = probability(top[3], interval=[0, 1])

    plt.figure()
    ln1, = plt.plot(x, c1, color='red', linewidth=1, linestyle='-', marker="^", markersize=3)
    ln2, = plt.plot(x, c2, color='blue', linewidth=1, linestyle='-', marker="^", markersize=3)
    ln3, = plt.plot(x, c3, color='green', linewidth=1, linestyle='-', marker="^", markersize=3)

    plt.xlabel('Overlapping Ratio', size=20)
    plt.ylabel('CDF', size=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))

    plt.legend(handles=[ln1, ln2, ln3], labels=['Top1', 'Top2', 'Top3'], fontsize=18, frameon=False)
    plt.tight_layout()
    plt.savefig('./Overlapping_Ratio_' + name + '.eps', dpi=300)
    plt.savefig('./Overlapping_Ratio_' + name + '.png')
    return top


# Plot the performances of model under different data qualities
def data_quality_plot(result, type):
    xname = ['Spatial Resolution', 'Ratio']
    name = [['default', '1km', '3km'], [0.3, 0.5, 0.7, 0.9]]
    criteria = ['Distance', 'Radius', 'Duration', 'G-rank', 'Move', 'Stay']
    plt.figure(figsize=(12, 6)) 
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.plot(range(result.shape[0]), result[:, i], linestyle='-', color='b', linewidth=2)
        plt.xlim(0, result.shape[0] - 1)
        plt.xticks(range(result.shape[0]), name[type], fontsize=15)
        plt.yticks(fontsize=15)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(FormatStrFormatter('%1.3f'))
        plt.xlabel(xname[type], fontsize=18)
        plt.ylabel(criteria[i], fontsize=18)
        plt.grid(True)
        plt.tight_layout()
    plt.savefig(xname[type] + '.eps', dpi=300)
    plt.savefig(xname[type] + '.png')

'''
PART 3 - CASE STUDY: locations and individual trajectories
'''

# Calculate the change of population of one particular place
def population_count(Data, param):
    population = np.zeros([param.loc_size, 48])
    for user in Data:
        popu = np.zeros([param.loc_size, 48])
        for traj in Data[user]:
            output = fixed(padding(Data[user][traj], 1440))
            for id, loc in enumerate(output):
                if loc == -1:
                    continue
                popu[int(loc), int(id%48)] += 1
        population += popu / len(Data[user])
    return population


 # Plot the change of population of one particular place
def location_plot(p, q):
    x = np.arange(48)
    p, q = p / p.max(), q / q.max()
    plt.bar(x/2, p, color='red', width=0.15,label='Simulation')
    plt.bar(x/2 + 0.25, q, color='blue', width=0.15, label='Real Data')
    ax=plt.gca()
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.1f'))
    ax.set_ylim(0,1.18)
    plt.ylabel('Population(Normalized)',fontsize=18)
    plt.xlabel('Time, hour', fontsize=18)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.grid(color='lightgray')
    plt.legend(loc=9, ncol=2, fontsize=17.9, borderaxespad=0 )
    ax.spines['bottom'].set_color('lightgray')
    ax.spines['top'].set_color('lightgray')
    ax.spines['left'].set_color('lightgray')
    ax.spines['right'].set_color('lightgray')
    plt.savefig('location_plot.png')
    plt.savefig('location_plot.eps')


# Individual trajectory Analysis
def Individual_Traj_plot(param, traj):
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

    ax.plot(x, y, np.array(z) / 60, color='red')
    ax.scatter(x, y, np.array(z) / 60, color='red', s=10)
    ax.xaxis.set_major_locator(MultipleLocator(max((max(x) - min(x)) / 4, 0.001)))
    ax.yaxis.set_major_locator(MultipleLocator(max((max(y) - min(y)) / 4, 0.01)))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    ax.set_zlim(0, 24)
    ax.zaxis.set_major_locator(MultipleLocator(6))
    plt.grid(color='whitesmoke')
    plt.savefig('Individual_Traj.eps', dpi=300)
    plt.savefig('Individual_Traj.png')
