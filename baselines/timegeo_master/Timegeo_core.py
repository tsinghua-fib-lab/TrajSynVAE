#!/usr/bin/env python
import time
import math
import random
import numpy as np

# global parameter setting
train_ratio = 1
active_user_stay = 0
active_home_stay = 5
active_work_stay = 5
time_slot = 10  # 10 minutes
spatial_threshold = 0.005  # approximate to 500m
home_start = 19
home_end = 8
home_threshold = 0.3
work_start = 8
work_end = 19
work_spatial_threshold = 0.005
work_frequency_threshold = 3

max_stay_duration = 12
max_explore_range = 200  # multiply spatial_resolution
spatial_resolution = 0.005
eta = 0.035
rho = 0.8
gamma = 0.21
beta1 = 70
beta2 = 700

performance_spatial_threshold = 0.005

end_date = '2020-01-08 23:59:59' 
decide_interval = 10 


def date2stamp(time_date):
    time_array = time.strptime(time_date, "%Y-%m-%d %H:%M:%S")
    time_stamp = int(time.mktime(time_array))
    return time_stamp


def stamp2date(time_stamp):
    time_array = time.localtime(time_stamp)
    time_date = time.strftime("%Y-%m-%d %H:%M:%S", time_array)
    return time_date


def stamp2array(time_stamp):
    return time.localtime(float(time_stamp))


# Distance function
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def stay_center(points):
    center = [0, 0]
    for p in points:
        center[0] += p[0]
        center[1] += p[1]
    center[0] /= len(points)
    center[1] /= len(points)
    return center


def cut_point(point_str):
    tmp = point_str.split('.')
    return float(tmp[0] + '.' + tmp[1][0:3])


def round_point(lat_lon):
    lat_lon_3 = []
    for p in lat_lon:
        lat_lon_3.append(int(p * 1000) / 1000.0)
    return lat_lon_3


def smooth_traces(user_stay, place):
    for i, p in enumerate(user_stay):
        if distance(p[0:2], place) < spatial_threshold:
            user_stay[i][0:2] = place
    return user_stay


# Pre_processing trace data
def preprocessing(line):
    user_id, traces = line.split('\t')
    user_trace = traces.split(';')

    user_traces = []
    for i, point in enumerate(user_trace):
        # point: lat,lon,time_date->time_stamp
        pp = point.strip('\n').split(',')
        if len(pp) != 3:
            continue
        time_stamp = date2stamp(pp[2])
        user_traces.append([cut_point(pp[0]), cut_point(pp[1]), time_stamp])

    user_pass = merge_same(user_traces)  # merge same point
    user_stay_tmp = merge_near(user_pass)  # merge near point

    # filter pass point by duration threshold
    temporal_threshold = 60 * time_slot  # 10 mins
    user_stay = []
    for point in user_stay_tmp:
        if (point[3] - point[2]) < temporal_threshold:
            pass
        else:
            point[0:2] = round_point(point[0:2])
            user_stay.append(point)
    if len(user_stay) <= active_user_stay:
        return {'user_stay': len(user_stay)}
    # identify location types
    home = identify_home(user_stay)
    if len(home) > 0:
        user_stay = smooth_traces(user_stay, home)
        work = identify_work(user_stay, home)
    else:
        work = []
    if len(work) > 0:
        user_stay = smooth_traces(user_stay, work)

    stay_count = len(user_stay)
    start_day = stamp2array(user_stay[0][2]).tm_yday
    end_day = stamp2array(user_stay[-1][3]).tm_yday

    test_start_day = start_day + int((end_day - start_day) * train_ratio)
    # regularize trace format (home:0, work:1, other:2)
    stays = []
    ground_truth = []
    home_stay_count = 0
    work_stay_count = 0
    for point in user_stay:
        if len(home) > 0 and distance(point[0:2], home) == 0:
            location_type = 0
            home_stay_count += 1
        elif len(work) > 0 and distance(point[0:2], work) == 0:
            location_type = 1
            work_stay_count += 1
        else:
            location_type = 2
        # p:[location,[lat,lon,in_time,out_time]]
        p = [location_type, point]
        stays.append(p)
        '''
        if stamp2array(point[2]).tm_yday < test_start_day:
            stays.append(p)
        else:
            ground_truth.append(p)
        '''

    info = {}
    info['user_id'] = user_id
    info['user_stay'] = stay_count
    info['home'] = home
    info['home_stay'] = home_stay_count
    info['work'] = work
    info['work_stay'] = work_stay_count
    info['start_day'] = start_day
    info['end_day'] = test_start_day - 1
    info['stays'] = stays
    info['ground_truth'] = ground_truth

    return info


def identify_home(user_stay):
    candidate = {}

    for p in user_stay:
        pid = str(p[0]) + ',' + str(p[1])
        duration = float(p[3] - p[2])
        start_time = stamp2array(p[2])
        end_time = stamp2array(p[3])
        home_duration = float(24 - home_start + home_end)
        r = 0
        if start_time.tm_wday in [0, 1, 2, 3, 4]:
            if start_time.tm_hour > home_start:
                if end_time.tm_hour < home_end:
                    r = min(1, duration / home_duration / 3600)
                else:
                    r = min(1, (24 - start_time.tm_hour + home_end) / home_duration)
            else:
                if end_time.tm_hour < home_end:
                    r = min(1, (24 - home_start + end_time.tm_hour) / home_duration)
                else:
                    r = 1

        if pid in candidate:
            candidate[pid] += r
        else:
            candidate[pid] = r
    res = sorted(candidate.items(), key=lambda e: e[1], reverse=True)

    # merge places which may be the same places 2017.3.24 20:52
    res_float = []
    for p in res:
        tmp = [float(x) for x in p[0].split(',')]
        pp = [tmp[0], tmp[1], p[1]]
        if len(res_float) == 0:
            res_float.append(pp)
        else:
            flag = 0
            for i, rp in enumerate(res_float):
                if distance(rp[0:2], pp[0:2]) < spatial_threshold:
                    res_float[i][2] += pp[2]
                    flag = 1
                    break
            if flag == 0:
                res_float.append(pp)

    if len(res_float) > 0 and sum([x[2] for x in res_float]) > 0:
        r = res_float[0][2] / float(sum([x[2] for x in res_float]))
    else:
        r = 0

    if r > home_threshold:
        return res_float[0][0:2]
    else:
        return []


def identify_work(user_stay, home):
    candidate = {}

    for p in user_stay:
        id = str(p[0]) + ',' + str(p[1])
        start_time = stamp2array(p[2])
        end_time = stamp2array(p[3])
        if start_time.tm_hour > work_start and end_time.tm_hour < work_end:
            if id in candidate:
                candidate[id] += 1
            else:
                candidate[id] = 1
    if len(candidate) == 0:
        return []
    for p in candidate:
        d = distance(home, [float(x) for x in p.split(',')])
        n = candidate[p]
        candidate[p] = [d, n]
    res = sorted(candidate.items(), key=lambda e: e[1][0] * e[1][1], reverse=True)
    if res[0][1][0] > work_spatial_threshold and res[0][1][1] >= work_frequency_threshold:
        return [float(x) for x in res[0][0].split(',')]
    else:
        return []


def merge_same(user_trace):
    last_place = []
    user_pass = []
    for point in user_trace:
        if last_place and distance(last_place, point[0:2]) < spatial_threshold * 0.1:
            user_pass[-1][3] = point[2]
        else:
            # stay: lat,lon,in_time,out_time
            stay = [point[0], point[1], point[2], point[2]]
            user_pass.append(stay)
            last_place = point[0:2]
    return user_pass


def merge_near(user_pass):
    merge_tmp = []
    user_stay = []
    for i, point in enumerate(user_pass):
        if len(merge_tmp) == 0:
            merge_tmp.append([point[0], point[1], i])
        else:
            flag = 0
            for p in merge_tmp:
                if distance(p[0:2], point[0:2]) < spatial_threshold:
                    merge_tmp.append([point[0], point[1], i])
                    flag = 1
                    break
            if flag:
                continue
            else:
                center = stay_center(merge_tmp)
                id1 = merge_tmp[0][2]
                id2 = merge_tmp[-1][2]
                ele = [center[0], center[1], user_pass[id1][2], user_pass[id2][3]]
                user_stay.append(ele)
                merge_tmp = [[point[0], point[1], i]]
    if len(merge_tmp) > 0:
        center = stay_center(merge_tmp)
        id1 = merge_tmp[0][2]
        id2 = merge_tmp[-1][2]
        ele = [center[0], center[1], user_pass[id1][2], user_pass[id2][3]]
        user_stay.append(ele)
    return user_stay


# Calculate parameters of individual: n_w
def individual_nw(info):
    user_trace = info['stays']

    nw = 0
    for i, p in enumerate(user_trace):
        if i == len(user_trace) - 1:
            break
        else:
            trip_origin = user_trace[i][0]
            trip_end = user_trace[i + 1][0]
            if trip_origin == 0 and trip_end == 2:
                nw += 1
    # we assume that the duration of the records is shorter than 1 years
    nw = nw * 7.0 / max(1, (info['end_day'] - info['start_day']))
    if nw < 6:
        nw = 6
    return nw


# Calculate parameters of global: P_t
def global_rhythm(info):
    rhythm = [0] * 7 * 24 * int(60 / time_slot)
    for i, p in enumerate(info['stays'][1:-1]):
        # point:[type,[lat,lon,start_time,end_time]]
        np = p[1]
        try:
            np_start = stamp2array(np[2])
            np_end = stamp2array(np[3])
            start_id = int(np_start.tm_wday * 24 * 6 + np_start.tm_hour * 6 + np_start.tm_min / time_slot)
            end_id = int(np_end.tm_wday * 24 * 6 + np_end.tm_hour * 6 + np_end.tm_min / time_slot)
            if info['stays'][i - 1][0] != 1 and p[0] != 1:
                rhythm[start_id] += 1
            if info['stays'][i + 1][0] != 1 and p[0] != 1:
                rhythm[end_id] += 1
        except:
            pass
    return 'rhythm', rhythm


# global explore delta_r distribution
def global_displacement(info):
    delta_r = [0] * max_explore_range
    last_location = info['stays'][0]
    for p in info['stays'][1:]:
        r = distance(p[1][0:2], last_location[1][0:2])
        rid = int(r / spatial_resolution)
        delta_r[min(len(delta_r) - 1, rid)] += 1
    return 'delta_r', delta_r


# For Spark: reduce function for global parameter
def global_reduce(pairs):
    c = []
    for i, a in enumerate(list(pairs[1])):
        if i == 0:
            c = a
        else:
            for j, d in enumerate(a):
                c[j] += a[j]
    total = float(sum(c))
    c2 = [x / total for x in c]
    return c2


# predict the time parameter of next place for the individual based on its current location type.
def predict_next_place_time(n_w, P_t, beta1, beta2, current_location_type):
    p1 = 1 - n_w * P_t
    p2 = 1 - beta1 * n_w * P_t
    p3 = beta1 * n_w * P_t * beta2 * n_w * P_t
    location_is_change = 0
    new_location_type = 'undefined'
    if current_location_type == 'home':
        if random.uniform(0, 1) <= p1:
            new_location_type = 'home'
            location_is_change = 0
        else:
            new_location_type = 'other'
            location_is_change = 1
    elif current_location_type == 'other':
        p = random.uniform(0, 1)
        if p <= p2:
            new_location_type = 'other'
            location_is_change = 0
        elif p <= p2 + p3:
            new_location_type = 'other'
            location_is_change = 1
        else:
            new_location_type = 'home'
            location_is_change = 1
    if new_location_type == 'home':
        return 0, location_is_change
    else:
        return 2, location_is_change


# simulate traces, run_mode:spark or streaming
def simulate_traces(info, rhythm_global, run_mode):
    n_w = individual_nw(info)
    '''
    if run_mode == 'spark':
        P_t = rhythm_global.value[0]
    elif run_mode == 'streaming':
        P_t = rhythm_global
    day_duration = float(info['end_day'] - info['start_day'])
    start_timestamp = info['stays'][0][1][2]

    # history data
    stay_duration_history = []
    trip_count_history = len(info['stays'])
    for p in info['stays']:
        duration = p[1][3] - p[1][2]
        stay_duration_history.append(duration)

    # simulate data
    test_pool = []
    for x1 in range(1, 20):
        for x2 in range(1, 101, 5):
            location_type_now = 'home'
            location_duration_now = 0
            trip_count_simulate = 0
            stay_duration_simulate = []

            for timer_shift in range(0, 60 * 60 * 24 * int(day_duration), time_slot * 60):
                timer = start_timestamp + timer_shift
                tm = stamp2array(timer)
                time_id = int(tm.tm_wday * 24 * 6 + tm.tm_hour * 6 + tm.tm_min / time_slot)
                location_type, location_change = predict_next_place_time(n_w, P_t[time_id], x1, x2, location_type_now)
                location_type_now = location_type
                location_duration_now += time_slot * 60
                if location_change:
                    trip_count_simulate += 1
                    stay_duration_simulate.append(location_duration_now)
                    location_duration_now = 0
            if location_duration_now > 0:
                stay_duration_simulate.append(location_duration_now)

            # calculate error
            test = pdf_minus(stay_duration_simulate, stay_duration_history) + eta * abs(
                trip_count_history - trip_count_simulate) / day_duration
            test_pool.append(test)
    beta_id = test_pool.index(min(test_pool))
    beta1 = beta_id / 20 + 1
    beta2 = (beta_id - beta1 * 20) * 5 + 1
    '''

    info['beta'] = [beta1, beta2]
    info['n_w'] = n_w
    return info


def pdf_minus(stay_duration_simulate, stay_duration_history):
    delta = time_slot * 60  # 10 minutes
    length = int(max_stay_duration * 60.0 / time_slot)
    res1 = [0] * length  # stay duration max length: max_stay_duration hours
    res2 = [0] * length
    for p in stay_duration_history:
        pid = int(p / delta)
        res1[min(length - 1, pid)] += 1
    for p in stay_duration_simulate:
        pid = int(p / delta)
        res2[min(length - 1, pid)] += 1
    sum1 = float(sum(res1))
    sum2 = float(sum(res2))
    res1 = [x / sum1 for x in res1]
    res2 = [x / sum2 for x in res2]

    pdf_err = 0
    for i in range(0, length):
        pdf_err += abs(res1[i] - res2[i])
    pdf_err *= delta
    return pdf_err


# predict the spatial parameter of next place for the individual based on its current location type.
def predict_next_place_location_simplify(P_new, delta_r, location_history, current_location):
    rp = random.uniform(0, 1)
    prob_accum = 0
    next_location = [2, 0, 0]
    if random.uniform(0, 1) < P_new:
        # explore
        for i, r in enumerate(delta_r):
            prob_accum += r
            if rp < prob_accum:
                radius = (i + 1) * spatial_resolution
                direction = random.uniform(0, 1) * 360
                next_lat = current_location[0] * radius * math.sin(direction)
                next_lon = current_location[1] * radius * math.cos(direction)
                next_location = [2, [next_lat, next_lon]]
                break
    else:
        # return
        return_selection = sorted(location_history.items(), key=lambda x: x[1], reverse=True)
        for lo in return_selection:
            prob_accum += lo[1][1]
            if rp < prob_accum:
                # type,[lat,lon]
                next_location = [lo[1][0], [float(x) for x in lo[0].split(',')]]
                break
    return next_location


def predict_next_place_location(info, delta_r, current_location):
    location_history = {}
    for p in info['stays']:
        pid = str(p[1][0]) + ',' + str(p[1][1])
        if pid in location_history:
            location_history[pid][1] += 1
        else:
            location_history[pid] = [p[0], 1]
    s = len(location_history)
    for lo in location_history:
        location_history[lo][1] /= len(info['stays'])

    P_new = rho * s ** (-gamma)
    return predict_next_place_location_simplify(P_new, delta_r, location_history, current_location)


# prepare time id for predict_evaluate
def time_id(time_stamp):
    ti = stamp2array(time_stamp)
    return int(ti.tm_wday * 24 * 60 / time_slot + ti.tm_hour * 60 / time_slot + ti.tm_min / time_slot)


# predict the next location during the whole next day and evaluate the predict performance
# run_mode:spark or streaming
def predict_evaluate(info, rhythm_global, deltar_global, run_mode):
    key = np.random.choice(len(info['stays']))
    start_location = info['stays'][key]
    start_time = start_location[1][2]
    time_date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    time_date = '2020-01-01' + time_date[10:]
    start_time = int(time.mktime(time.strptime(time_date, "%Y-%m-%d %H:%M:%S")))
    end_time = int(time.mktime(time.strptime(end_date, "%Y-%m-%d %H:%M:%S")))
    ground_duration = (end_time - start_time) / 60 / decide_interval


    n_w = info['n_w']
    beta1, beta2 = info['beta']
    if run_mode == 'spark':
        deltar = deltar_global.value[0]
        rhythm = rhythm_global.value[0]
    elif run_mode == 'streaming':
        deltar = deltar_global
        rhythm = rhythm_global
    location_type = ['home', 'work', 'other']

    location_history = {}
    for p in info['stays']:
        pid = str(p[1][0])[0:7] + ',' + str(p[1][1])[0:6]
        if pid in location_history.keys():
            location_history[pid][1] += 1
        else:
            location_history[pid] = [p[0], 1]
    s = len(location_history)
    for lo in location_history:
        location_history[lo][1] /= float(len(info['stays']))
    P_new = rho * s ** (-gamma)

    '''
    ground_trace = [0] * int(ground_duration)
    for p in info['ground_truth']:
        p_in = p[1][2]
        p_out = p[1][3]
        bins = (p_out - p_in) / 60 / time_slot
        time_id_now = (p_in - start_time) / 60 / time_slot
        for t in range(int(time_id_now), int(time_id_now + bins)):
            ground_trace[t] = [p[0], p[1][0], p[1][1], p_in + t * 60 * time_slot]
    '''

    predict_trace = [0] * int(ground_duration)
    current_location = [start_location[0], start_location[1][0:2]]
    for tid in range(70, int(ground_duration)):
        t = start_time + tid * 60 * time_slot
        ta = stamp2array(t)


        '''
        if ta.tm_hour in [2, 3, 4]:
            tmp = ground_trace[tid]
            predict_trace[tid] = tmp
            if tmp != 0:
                current_location = [tmp[0], [tmp[1], tmp[2]]]
        '''
        if 1 != 1:
            pass
        else:
            P_t = rhythm[time_id(t)]
            now_type, location_change = predict_next_place_time(n_w, P_t, beta1, beta2,
                                                                location_type[current_location[0]])
            p1 = 1 - n_w * P_t
            p2 = 1 - beta1 * n_w * P_t
            p3 = beta1 * n_w * P_t * beta2 * n_w * P_t
            if location_change:
                next_location = predict_next_place_location_simplify(P_new, deltar, location_history,
                                                                     current_location[1])
            else:
                next_location = [now_type, current_location[1]]
            current_location = next_location
            predict_trace[tid] = [next_location[0], next_location[1][0], next_location[1][1], t]

    '''
    predict_correct = 0
    for n in range(0, len(predict_trace)):
        if ground_trace[n] == 0 or predict_trace[n] == 0:
            continue
        elif distance(predict_trace[n][1:3], ground_trace[n][1:3]) < performance_spatial_threshold and stamp2array(
                t).tm_hour not in [2, 3, 4]:
            predict_correct += 1
    

    predict = 0
    ground = 0
    for i, j in zip(predict_trace, ground_trace):
        if i != 0:
            predict += 1
        if j != 0:
            ground += 1
    '''

    info['location_history'] = location_history
    info['predict_trace'] = predict_trace
    # info['ground_trace'] = ground_trace
    # info['performance'] = [predict_correct, predict, ground]
    return info
