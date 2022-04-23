import sys
from .Timegeo_core import *


def read_from_input(file):
    for line in file:
        yield line


def read_from_file(file):
    with open(file) as f:
        lines = f.readlines()
    for line in lines:
        yield line


def main(rhythm_global, deltar_global,map_mode, infos):
    geos = []
    for info in infos:
        if map_mode == 'predict':
            predict = predict_evaluate(info, rhythm_global, deltar_global, run_mode='streaming')
            geos.append(predict)

        elif map_mode == 'simulate':
            info2 = simulate_traces(info, rhythm_global, run_mode='streaming')
            geos.append(info2)
    return geos


if __name__ == '__main__':
    rhythm = eval(open('./rhythm').readline())
    deltar = eval(open('./deltar').readline())
    main(rhythm_global=rhythm, deltar_global=deltar,map_mode='simulate')
