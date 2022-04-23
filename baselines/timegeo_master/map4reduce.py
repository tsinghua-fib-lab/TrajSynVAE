import sys
from .Timegeo_core import *


def read_from_input(file):
    for line in file:
        yield line


def main(run_mode):
    for line in read_from_input(sys.stdin):
        info = eval(line.strip('\n'))
        if run_mode == 'rhythm':
            try:
                rhythm = global_rhythm(info)
                print("%s%s%s" % (rhythm[0], '\t', rhythm[1]))
            except:
                sys.stderr.write('failed!   ' + info['user_id'] + '\n')
        elif run_mode == 'deltar':
            try:
                delta_r = global_displacement(info)
                print("%s%s%s" % (delta_r[0], '\t', delta_r[1]))
            except:
                sys.stderr.write('failed!   ' + info['user_id'] + '\n')


if __name__ == '__main__':
    main(run_mode=sys.argv[1])
