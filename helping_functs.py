
import cv2
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy import signal

def initiate_objects(num):
    return tuple([DataObj()]*num)

def im_convert(fil, path, flag):
    if flag=='Depth':
        img=cv2.imread(path+'/'+fil)
        img=(256*256*img[:, :, 0]+256*img[:, :, 1]+img[:, :, 2])/float(256*256*256)
    elif flag=='Color':
        initimg=cv2.imread(path+'/'+fil)
        img=np.zeros([424, 512, 3])
        for c in range(3):
            img[:, :, c]=cv2.resize(initimg[:, :, c], (512, 424))/float(256)
        return(img)

def im_load(path, flag):
    contents=os.listdir(path) # returns list
    ctime= map(lambda fil:os.stat(path+'/'+fil)[8], contents)
    contents=[x for (y, x) in sorted(zip(ctime, contents))]
    if flag=='Depth':
        imdata=np.transpose(np.concatenate([map(lambda fil:im_convert(fil, path, flag), contents)]), (1, 2, 0))
    elif flag=='Color':
        imdata=np.transpose(np.concatenate([map(lambda fil:im_convert(fil, path, flag), contents)]), (1, 2, 3, 0))
    return(imdata)


def hist(image):
    hbins=30
    plt.hist(image.ravel()*256, 256, [1, 256])
    plt.show()

def diff(a, b):
    if sum(list(b.shape))/2>1:
        return [np.sqrt((i[0]-j[0])**2+(i[1]-j[1])**2) for i, j in zip(np.array(a), np.array(b))]
    else:
        if sum(list(b.shape))==2:
            return [np.sqrt((i[0]-b[0])**2+(i[1]-b[1])**2) for i in a]
        else:
            return [np.sqrt((i-b)**2) for i in a]


def interpolate(points, winsize):
    interpolated_data=[]
    for count, i in enumerate(range(0, points.shape[0], (winsize-1)/2)):
        interpolated_data.append(np.mean(points[max(0, i-(winsize-1)/2):(min(i+(winsize-1)/2, points.shape[0]-1)+1)], axis=0))
    return np.array(interpolated_data)


def compute_angles(points):
    return np.array([np.arctan2(n[1], n[0]) for n in points[1:]-points[:-1]])
