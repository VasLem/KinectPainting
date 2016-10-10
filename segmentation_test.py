'''Functions for identifying moving object in almost static scene'''
#import imp
'''try:
    imp.find_module('numbapro')
    WITH_CUDA = True
    from numbapro import cuda
except ImportError:
    WITH_CUDA = False
'''
import time
import numpy as np
import cv2
import class_objects as co

import cProfile,pstats, StringIO
'''
def select_values(inp, s, out):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if (tid < out.size) :
        out[tid]=inp[s[tid]]
'''
def scene_segmentation():
    '''Segment background'''
    pr= cProfile.Profile()
    pr.enable()
    if isinstance(co.segclass.segment_values, str):
        co.segclass.segment_values = co.data.depth_im.copy()
    print np.max(co.segclass.segment_values)
    val=0.5 
    thres =np.minimum(np.ceil(val * co.segclass.segment_values),5)
    time1 = time.clock()
    co.segclass.array_instances = np.zeros(
        (9, co.meas.imy + 2, co.meas.imx + 2),dtype=np.uint8)
    co.segclass.array_instances[0, :-2, :-
                                2] = co.segclass.segment_values
    co.segclass.array_instances[1, :-2, 1:-1] =\
    co.segclass.segment_values
    co.segclass.array_instances[2, :-2, 2:] =\
            co.segclass.segment_values
    co.segclass.array_instances[3, 1:-1, :-2] =\
            co.segclass.segment_values
    co.segclass.array_instances[4, 1:-1, 2:] =\
            co.segclass.segment_values
    co.segclass.array_instances[5, 2:, :-2] =\
            co.segclass.segment_values
    co.segclass.array_instances[6, 2:, 1:-1] =\
            co.segclass.segment_values
    co.segclass.array_instances[7, 2:, 2:] = \
            co.segclass.segment_values
    co.segclass.array_instances[8, 1:-1, 1:-1] = \
            co.segclass.segment_values
    time2 = time.clock()
    differences = np.abs(co.segclass.array_instances -
                         co.segclass.array_instances[8, :, :])
    differences[8, 1:-1, 1:-1] = thres
    #differences[differences == 0] = 1
    time3 = time.clock()
    minargs = np.argmin(differences, axis=0).ravel()
    time4 = time.clock()
    '''if WITH_CUDA:
        d_array_instances = cuda.to_device(co.segclass.array_instances)
        
        d_tmp_segment_values=cuda.to_device(tmp_segment_values)
        d_array_instances=cuda.to_device(co.segclass.array_instances.ravel())
        out_segment_values = np.zeros_like(tmp_segment_values)
        d_out_segment_values=cuda.to_device(out_segment_values)
        threads_per_block=9
        number_of_blocks=(co.meas.imx+2)*(co.meas.imy+2)
        select_values[number_of_blocks, threads_per_block](d_array_instances,d_tmp_segment_values,d_out_segment_values)
        d_out_segment_values.copy_to_host(tmp_segment_values)
    else:
    '''
    
    tmp_segment_values =\
    np.take(co.segclass.array_instances.ravel(),minargs*(co.meas.imx+2)*(co.meas.imy+2)+nprange).reshape(co.meas.imy + 2, co.meas.imx + 2)
    time5 = time.clock()
    
    co.segclass.segment_values = tmp_segment_values[1:-1, 1:-1]
    
    #cv2.imshow('segments', co.segclass.segment_values)
    #cv2.waitKey(100)
    print "Loop:", co.meas.im_count
    print time2 - time1, 's'
    print time3 - time2, 's'
    print time4 - time3, 's'
    print time5 - time4, 's'
    #cv2.imshow('segments',co.segclass.segment_values) 
    pr.disable()
    s = StringIO.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
#    print s.getvalue()
co.data.depth_im = cv2.imread('test.jpg')[:, :, 0]
print co.data.depth_im.shape
co.data.depth_im = (((co.data.depth_im - np.min(co.data.depth_im)) / \
    float(np.max(co.data.depth_im) -
          np.min(co.data.depth_im)))*255).astype(np.uint8)
co.meas.imy, co.meas.imx = co.data.depth_im.shape
nprange=np.arange((co.meas.imx+2)*(co.meas.imy+2))
print 'im_shape', co.meas.imy, co.meas.imx
#def loop():
#scene_segmentation()

for c in range(100):
    co.meas.im_count += 1
    scene_segmentation()

difference = np.abs(co.segclass.segment_values - co.data.depth_im)
print np.unique(difference)
from numpy import histogram
print histogram(difference,bins=difference.max()-1)
cv2.imshow('segments',co.segclass.segment_values)
cv2.imshow('difference',
               (255*((difference -
               np.min(difference))/float(np.max(difference)-np.min(difference)))).astype(np.uint8))
            
cv2.waitKey(0)

