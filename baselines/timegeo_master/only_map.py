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


def main(filter_mode, fp):
    infos = []
    for line in read_from_file(fp):
        if filter_mode == 'user':  # based on original traces
            try:
                info = preprocessing(line)
                if info['user_stay'] <= active_user_stay:
                    pass
                else:
                    infos.append(info)
                    # sys.stderr.write('succeed!   '+line.split('\t')[0]+'\n')
            except:
                sys.stderr.write('failed!   ' + line.split('\t')[0] + '\n')
        elif filter_mode == 'home':  # based on pre_processing info
            try:
                info = eval(line)
                if info['home_stay'] < active_home_stay:
                    pass
                else:
                    print("%s" % info)
                    sys.stderr.write('succeed!')
            except:
                sys.stderr.write(line)
                sys.stderr.write('failed!   ' + info['user_id'] + '\n')
        elif filter_mode == 'commuter':  # based on home info
            try:
                info = eval(line)
                if info['work_stay'] < active_work_stay:
                    pass
                else:
                    print("%s" % info)
            except:
                sys.stderr.write('failed!   ' + info['user_id'] + '\n')
        elif filter_mode == 'noncommuter':  # based on home info
            try:
                info = eval(line)
                if info['work_stay'] == 0:
                    print("%s" % info)
                else:
                    pass
            except:
                sys.stderr.write('failed!   ' + info['user_id'] + '\n')
    return infos

if __name__ == '__main__':
    main(filter_mode='user', fp='../example.data')
