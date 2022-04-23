# encoding: utf-8
import string
import numpy as np


data = ''

gps = []
data_path = ''
with open(data_path+'/LocList.txt') as f:
#with open('/data/geolife/GPS') as f:
    lines = f.readlines()
for l in lines:
    gps.append([float(x) for x in l.split()])


def read_data_from_file(fp):
    """Read a bunch of trajectory data from txt file.
    Parameters
    ----------
    fp : str
        file path of data

    Return
    ----------
    ndarray
        2d array of (traj_nums * traj_len)
    """
    dat = []
    with open(fp, 'r') as f:
        m = 0
        lines = f.readlines()
        for idx, line in enumerate(lines):
            tmp = line.split()
            dat += [[int(t) for t in tmp]]
    return np.asarray(dat, dtype='int64')


def random_user_id():
    chars = list(string.ascii_uppercase + string.digits)
    choice = np.random.choice(len(chars), 32)
    return ''.join([chars[i] for i in choice])


def traj_to_geo(traj):
    """

    :param traj:
    :return:
    """
    seq_len = traj.shape[0]
    line = random_user_id() + '\t'
    for i in range(seq_len):
        a = traj[i]
        a = 10082 if a>= 10083 else a
        #print(traj[i])
        x = gps[a][0]
        y = gps[a][1]
        if i < 24:
            line += '%f,%f,2016-01-01 %s:00:00;' % (x, y, str(i).zfill(2))
        elif 24<=i<48:
            line += '%f,%f,2016-01-02 %s:00:00;' % (x, y, str(i % 24).zfill(2))
        elif 48<=i<72:
            line += '%f,%f,2016-01-03 %s:00:00;' % (x, y, str(i % 24).zfill(2))
        elif 72<=i<96:
            line += '%f,%f,2016-01-04 %s:00:00;' % (x, y, str(i % 24).zfill(2))
        elif 96<=i<120:
            line += '%f,%f,2016-01-05 %s:00:00;' % (x, y, str(i % 24).zfill(2))
        elif 120<=i<144:
            line += '%f,%f,2016-01-06 %s:00:00;' % (x, y, str(i % 24).zfill(2))
        elif 144<=i<168:
            line += '%f,%f,2016-01-07 %s:00:00;' % (x, y, str(i % 24).zfill(2))
    return line


def locate_geopoint_index(x, y):

    dist = []
    for xy in gps:
        dist.append((xy[0] - x)**2 + (xy[1] - y)**2)
    dist = np.array(dist)
    return np.where(dist == dist.min())[0][0]


def geo_to_traj(geo):

    loc_times = geo.split('\t')[1].split(';')
    traj = []
    for lt in loc_times:
        x, y, _ = lt.split(',')
        traj.append(locate_geopoint_index(x, y))
    traj = np.array(traj)
    return traj


def geostream_to_trajs(stream_lines):

    trajs = []
    for line in stream_lines:
        traj = geo_to_traj(line)
        trajs.append(traj)
    trajs = np.array(traj)
    return trajs


def trajs_to_streams(trajs):
    """

    :param trajs:
    :return:
    """
    total_num = trajs.shape[0]
    seq_len = trajs.shape[1]
    lines = []
    for i in range(total_num):
        lines.append(traj_to_geo(trajs[i]))
    return lines


if __name__ == '__main__':

    # convert real.data to timegeo-master input format
    dat = read_data_from_file(data_path+'/real.data')
    lines = trajs_to_streams(dat)
    with open(data_path+'/res', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
