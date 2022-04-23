import pdb
from . import only_map
from . import eval_map

import numpy as np


gps = []
with open('') as f:
    lines = f.readlines()
for l in lines:
    gps.append([float(x) for x in l.split()])


def locate_geopoint_index(x, y):

    dist = []
    for xy in gps:
        dist.append((xy[0] - x)**2 + (xy[1] - y)**2)
    dist = np.array(dist)
    return np.where(dist == dist.min())[0][0]


rhythm = eval(open('./rhythm').readline())
deltar = eval(open('./deltar').readline())
infos = only_map.main(filter_mode='user', fp='')
print(infos)


print("===============SIMULATE===============")
geos = eval_map.main(rhythm_global=rhythm, deltar_global=deltar,map_mode='simulate', infos=infos)

print("===============ANALYZE===============")
with open('', 'w') as f:
    for geo in geos:
        stays = geo['stays']
        poss = []
        for stay in stays:
            pos = locate_geopoint_index(stay[1][0], stay[1][1])
            st = stay[1][2]
            et = stay[1][3]
            times = int((et - st) / 3600) + 1
            for t in range(times):
                poss.append(pos)
        f.write(' '.join([str(s) for s in poss]))
        f.write('\n')


