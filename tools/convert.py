#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import argparse
import subprocess
import os
from os import path
import multiprocessing as mp
from multiprocessing import Value
try:
    # Python 3
    from itertools import filterfalse
except ImportError:
    # Python 2
    from itertools import ifilterfalse

def print_progress(iteration, total, decimals=1, bar_length=50):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)

    sys.stdout.write('\r |%s| %s%s (%s of %s)' %
                     (bar, percents, '%', str(iteration), str(total)))
    sys.stdout.flush()
   
def is_not_h5m(file):
    return not file.endswith('h5m')

def is_older_than_vtk(file):
    file_vtk = path.splitext(file)[0] + ".vtk"
    if path.exists(file_vtk):
        return path.getmtime(file) < path.getmtime(file_vtk)
    return False

def mb_convert(file):
    file = path.splitext(file)[0]
    p = subprocess.Popen(["mbconvert", file + ".h5m", file + ".vtk"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = p.communicate()
    lock.acquire()
    if p.returncode:
        print('\n' + err.decode())
    n.value += 1
    print_progress(n.value, N.value)
    lock.release()

def init(l):
    global lock
    lock = l

def filterfalse_comp(fun, file_list):
    try:
        # Python 3
        return filterfalse(fun, file_list)
    except NameError:
        # Python 2
        return ifilterfalse(fun, file_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Convert multiple h5m files to vtk using mbconvert and multiprocessing")
    parser.add_argument(
        "file", help="list of h5m files or a regexp mask", nargs='+')
    parser.add_argument("-np", help="number of processes", type=int, default=1)
    args = parser.parse_args()

    file_list = list(filterfalse_comp(is_not_h5m, args.file))
    if not len(file_list):
        print("No h5m files found with the given name/mask")
        exit()

    file_list = list(filterfalse_comp(is_older_than_vtk, file_list))
    if not len(file_list):
        print("All found h5m files are older than corresponding vtk files")
        exit()

    N = Value('i', len(file_list))
    n = Value('i', 0)

    l = mp.Lock()
    pool = mp.Pool(processes=args.np, initializer=init, initargs=(l,))

    print_progress(n.value, N.value)
    pool.map(mb_convert, file_list)
    pool.close()
    pool.join()

    print('\n')

    exit()