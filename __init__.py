import os, time
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

import logging
def initialize_logger(obj, log_lev=None):
    obj.logger = logging.getLogger(obj.__class__.__name__)
    if not getattr(obj.logger, 'handler_set', None):
        CH = logging.StreamHandler()
        CH.setFormatter(logging.Formatter(
            '%(name)s-%(funcName)s()(%(lineno)s)-%(levelname)s:%(message)s'))
        obj.logger.addHandler(CH)
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
    def wrapper(self,*arg, **kw):
        t1 = time.time()
        res = func(self,*arg, **kw)
        t2 = time.time()
        self.time.append(t2-t1)
        del self.time[:-5000]
        return res
    return wrapper
