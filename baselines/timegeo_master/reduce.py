import sys
from .Timegeo_core import *
from operator import itemgetter
from itertools import groupby


def read_from_mapper(file):
    for line in file:
        yield line.strip('\n').split('\t')


def main(run_mode):
    for key, value in groupby(read_from_mapper(sys.stdin), itemgetter(0)):
        if run_mode == 'rhythm':
            info = [0] * 7 * 24 * (60 / time_slot)
        elif run_mode == 'deltar':
            info = [0] * max_explore_range
        for v in value:
            for i, b in enumerate(eval(v[1])):
                info[i] += b
        total = float(sum(info))
        c = [x / total for x in info]
        print(c)


if __name__ == '__main__':
    main(run_mode=sys.argv[1])
