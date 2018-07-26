'''
General functions
'''
import os
import time
import subprocess
import logging
import psutil
import shlex
import numpy as np
import cv2

CONST_LOC = os.path.dirname(os.path.realpath(__file__))
# Kinect Intrinsics
PRIM_X = 256.92
PRIM_Y = 204.67
FLNT = 365.98
# Senz3d Intrinsics
'''
PRIM_X = 317.37514566554989
PRIM_Y = 246.61273826510859
FLNT = 595.333159044648 / (30.48 / 1000.0)
'''

def get_real_coordinate(sensor_x, sensor_y, sensor_d):
    '''
    Use pinhole camera model
    '''
    real_z = sensor_d / 1000.0;
    real_x = (sensor_x - PRIM_X) * real_z / FLNT
    real_y = (sensor_y - PRIM_Y) * real_z / FLNT
    return np.array([real_x, real_y, real_z])

def initialize_logger(obj, log_lev=None):
    '''
    Given a logger, initialize it using default settings
    '''
    obj.logger = logging.getLogger(obj.__class__.__name__)
    if not getattr(obj.logger, 'handler_set', None):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            '%(name)s-%(funcName)s()(%(lineno)s)-%(levelname)s:%(message)s'))
        obj.logger.addHandler(stream_handler)
        obj.logger.handler_set = True
    if log_lev is not None:
        obj.logger.setLevel(log_lev)
    obj.logger.propagate = False


def find_nonzero(arr):
    '''
    Finds nonzero elements positions
    '''
    return np.fliplr(cv2.findNonZero(arr).squeeze())

def timeit(func):
    '''
    Decorator to time extraction
    '''
    def wrapper(self, *arg, **kw):
        '''
        Wrapper to measure execution time
        '''
        time1 = time.time()
        res = func(self, *arg, **kw)
        time2 = time.time()
        self.time.append(time2-time1)
        del self.time[:-5000]
        return res
    return wrapper

def run_on_external_terminal(command):
    '''
    Caution: the window closes if command is executed.
    '''
    args = shlex.split('/usr/bin/x-terminal-emulator -e ' + command)
    try:
        return subprocess.Popen(args).pid
    except OSError:
        raise

def check_if_running(name):
    '''
    Retuns true if a process exists, whose name contains the input argument
    '''
    return any(name in p.name() for p in psutil.process_iter())

def terminate_process(inp, use_pid=False):
    '''
    Terminates any process, whose name contains inp if use_pid is False, else
    whose pid is the one given as input.
    '''
    for proc in psutil.process_iter():
        # check whether the process name matches
        if use_pid and proc.pid == inp:
            proc.kill()
            return True
        elif inp in proc.name():
            proc.kill()
            return True
    return False
