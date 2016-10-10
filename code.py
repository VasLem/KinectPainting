
import cv2
import os
import numpy as np
import math
import yaml
import time
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import Moving_Object_Detection_Alg as moda
import Palm_Detection_Alg as pda
import itertools as it
from scipy import signal






with open("config.yaml", 'r') as stream:
    try:
        cg=yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


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



r=cg['read']
if r=='d':
    path_depth=cg['path_depth']
    path_color=cg['path_color']
    data_depth=im_load(path_depth, 'Depth')
    data_color=im_load(path_color, 'Color')
    np.save(cg['save_depth'], data_depth)
    np.save(cg['save_color'], data_color)
elif r=='f':
    data_depth=np.load(cg['save_depth']+'.npy')
    data_color=np.load(cg['save_color']+'.npy')
elif r=='k':
    print 'no Kinect yet'





s=cg['save']
if s=='y':
    np.save(cg['save_depth'], data_depth)
    np.save(cg['save_color'], data_color)


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


n=9
# Desired n
#n=2*cg['framerate']
Noise_Model=moda.Init_Noise_Model(data_depth[:, :, :n]);
Found_Object_Im=np.zeros(data_depth.shape)
aver_count=0
im=data_depth[:, :, 0]

min1=np.min(im[im>0])
im[im==min1]=1
min2=np.min(im[im>0])
least_resolution=min2-min1
depth_thres=least_resolution*cg['depth_thres']
aver_depth=0
lap_thres=cg['lap_thres']
depth_mem=[]
maxcontoursnumber=3
imy, imx=im.shape

def interpolate(points, winsize):
    interpolated_data=[]
    for count, i in enumerate(range(0, points.shape[0], (winsize-1)/2)):
        interpolated_data.append(np.mean(points[max(0, i-(winsize-1)/2):(min(i+(winsize-1)/2, points.shape[0]-1)+1)], axis=0))
    return np.array(interpolated_data)


def compute_angles(points):
    return np.array([np.arctan2(n[1], n[0]) for n in points[1:]-points[:-1]])



for c in range(n,data_depth.shape[2]):
    im=data_depth[:, :, c]
    #start1=time.time()
    final_mask, arm_contour, sorted_contours, matched=moda.Find_Moving_Object(im, Noise_Model, depth_thres, lap_thres, aver_depth, maxcontoursnumber, depth_mem, aver_count)
    #end1=time.time()

    final=np.ones((im.shape[0], im.shape[1], 3))
    final[final_mask>0, 0]=im[final_mask>0]
    final[final_mask>0, 1]=im[final_mask>0]
    final[final_mask>0, 2]=im[final_mask>0]

    x, y, w, h = cv2.boundingRect(arm_contour)
    arm_contour=arm_contour.squeeze().astype(int)
    #start2 = time.time()
    wristpoints,hand_contour=pda.detect_wrist(arm_contour,imx,imy)
    #end2=time.time()
    if not wristpoints:
        continue
    #print "Find_Moving_Object time",(end1-start1)
    #print "detect_wrist time",(end2-start2)
    #print "total time ",end2-start1

    hand_hull=cv2.convexHull(hand_contour,returnPoints=False)
    hand_convexity_defects=cv2.convexityDefects(hand_contour,hand_hull)

    x_hand, y_hand, w_hand, h_hand = cv2.boundingRect(hand_contour)
    gauss_win_size=7
    sigma=3
    Rg=range(-(gauss_win_size-1)/2,(gauss_win_size-1)/2+1)
    Y,X=np.meshgrid(Rg,Rg)
    log=-1/(math.pi*sigma**4)*(1-(X**2+Y**2)/(2*sigma**2))*np.exp(-(Y**2+X**2)/(2*sigma**2))



    hand_im=im[y_hand:y_hand+h_hand,x_hand:x_hand+w_hand]
    hand_im_contour=hand_contour-np.array([x_hand,y_hand])


    hand_shape=np.ones(hand_im.shape)
    cv2.drawContours(hand_shape, [hand_im_contour], 0, 0, -1)
    bool_tmp=hand_shape==1
    hand_shape[hand_shape==0]=hand_im[hand_shape==0]
    hand_shape[hand_shape==1]=hand_shape[hand_shape==1]*(np.max(hand_im[hand_shape<1])+np.min(hand_im[hand_shape<1]))/2
    hand_shape=(hand_shape-np.min(hand_shape))/(np.max(hand_shape)-np.min(hand_shape))


    res=signal.convolve2d(hand_shape,log,mode='same')
    res=(res-np.min(res))/(np.max(res)-np.min(res))
    res[bool_tmp]=0
    res=cv2.Canny((res*256).astype(np.uint8),25,50)
    _,contours, _ = cv2.findContours(res.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    per=[]
    for i, c1 in enumerate(contours):
        per.append(cv2.arcLength(c1,True))
    res=np.ones(hand_im.shape)
    cv2.drawContours(res,contours,per.index(max(per)),0,-1)



    #res=cv2.Laplacian(res,cv2.CV_64F)
    #
    #
    #res[res<0.5]=0
    cv2.imshow("hand_shape",hand_shape)
    cv2.imshow("res",res)


    cv2.drawContours(final, [hand_contour], 0, [0, 1, 1], -1)
    for defect in hand_convexity_defects.squeeze():
        if defect[3]/256 >5:
            cv2.circle(final, tuple(hand_contour[defect[2]].squeeze()),3,[1, 1, 0],-1)
    #cv2.circle(final, tuple(wristpoints[0]), 3, [1, 0, 1], -1)
    #cv2.circle(final, tuple(wristpoints[1]), 3, [0, 0, 1], -1)
    #font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(final, str(wristpoint1), tuple(wristpoint1), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    #cv2.circle(final, tuple(hand_centered_contour[0]), 3, [0, 0.5, 1], -1)
    #cv2.circle(final, tuple(contour_corners[0]), 3, [0, 1, 0], -1)
    #cv2.circle(final, tuple(contour_corners[1]), 3, [0, 1, 0], -1)
    #cv2.imshow('image2', hand_shape)


    #cv2.line(final, tuple(wristpoint[0]), tuple((wristpoint[0]+eigenvectors[0]*d).astype(int)), (1, 0, 1), 5)
    '''
    palm_shape=np.ones(im.shape)*min(im[hand_shape==255])

    cv2.drawContours(palm_shape, [palm_contour], 0, 0, -1)
    palm_shape[palm_shape==0]=im[palm_shape==0]

    x, y, w, h = cv2.boundingRect(palm_contour)
    # X and Y coordinates of points in the image, spaced by 10.
    (X, Y) = np.meshgrid(range(x, x+w), range(y, y+h))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, palm_shape[Y, X] , zdir='z', s=4, c='b', depthshade=True)
    # Plot points from the image.
    #plt.scatter(X, Y, hand_shape[Y, X])
    plt.show()'''







    #cv2.imshow('image', final)
    cv2.waitKey(1000/cg['framerate'])
