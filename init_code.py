'''Run this script first'''
import subprocess
from subprocess import check_output
import signal
import sys
import time
import os
import cPickle as pickle
#import statsmodels.robust.scale
import yaml
import numpy as np
import cv2
import rospy
import roslaunch
import message_filters as mfilters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import moving_object_detection_alg as moda
#import palm_detection_alg as pda
import helping_functs as hf
import class_objects as co

PROCESS = None



def get_pid(name):
    '''
    Retrieves pid from name
    '''
    return map(int, check_output(["pidof", name]).split())
 
def rosbag_handler(sig,frame):
    print 'Toggled ROSbag pause/continue'
    PROCESS.stdin.write(' ')

def signal_handler(sig, frame):
    '''
    Signal handler for CTRL-C interrupt (SIGINT)
    '''
    print '\nGot SIGINT'
    print 'Exiting...'
    if co.CONST['stream'] == 'live':
        PROCESS.stop()
    else:
        os.killpg(PROCESS.pid, signal.SIGINT)
    sys.exit(0)


def improve_final_mask():
    '''improve depth image result by comparing with color image'''
    # Missing variance measure. Needs debugging if going to be used
    co.models.med = np.median(
        co.data.color_im[co.masks.final_mask > 0][:], axis=0)
    x_coord, y_coord, co.meas.w, co.meas.h = cv2.boundingRect(
        co.contours.arm_contour)
    co.masks.color_mask = cv2.inRange(
        co.data.color_im[y_coord:y_coord +
                         co.meas.h, x_coord:x_coord + co.meas.w],
        -2 * co.models.var * co.models.med, 2 * co.models.var + co.models.med)
    co.masks.final_mask[y_coord:y_coord + co.meas.h, x_coord:x_coord + co.meas.w] = np.logical_and(
        co.masks.color_mask > 0, co.masks.final_mask[y_coord:y_coord + co.meas.h,
                                                     x_coord:x_coord + co.meas.w] > 0)
    return co.masks


def frame_process():
    '''main function for frame processing'''
    co.masks.final_mask, co.contours.arm_contour, \
        co.contours.sorted_contours, _ = moda.find_moving_object()
    if isinstance(co.masks.final_mask, str):
        print co.masks.final_mask
        return co.masks.final_mask
    #co.masks = improve_final_mask(co.contours, co.data, co.masks, co.meas, co.models)

    co.masks.rgb_final = np.ones((co.data.depth_im.shape + (3,)))
    co.masks.rgb_final[co.masks.final_mask > 0,
                       0] = co.data.depth_im[co.masks.final_mask > 0]
    co.masks.rgb_final[co.masks.final_mask > 0,
                       1] = co.data.depth_im[co.masks.final_mask > 0]
    co.masks.rgb_final[co.masks.final_mask > 0,
                       2] = co.data.depth_im[co.masks.final_mask > 0]
    co.points.x_coord, co.points.y_coord, co.meas.w, co.meas.h = cv2.boundingRect(
        co.contours.arm_contour)
    co.contours.arm_contour = np.array(
        co.contours.arm_contour).squeeze().astype(int)
    #start2 = time.time()
    # co.points.wristpoints, co.contours.hand = pda.detect_wrist(
    #    co.CONST)
    # end2=time.time()
    if isinstance(co.points.wristpoints, str):
        print co.points.wristpoints
        return co.points.wristpoints
    '''hulls = Hull()
    hulls.hand = cv2.convexHull(co.contours.hand, returnPoints=False)
    convexity_defects = Convexity_Defect()
    convexity_defects.hand = cv2.convexityDefects(co.contours.hand, hulls.hand)'''
    co.points.x_hand, co.points.y_hand, co.meas.w_hand, co.meas.h_hand = cv2.boundingRect(
        co.contours.hand)
    ''' co.CONST.gauss_win_size = 7
    sigma= 3
    Rg = range(-(gauss_win_size - 1) / 2, (gauss_win_size - 1) / 2 + 1)
    Y, X = np.meshgrid(Rg, Rg)
    log = -1 / (math.pi * sigma**4) * (1 - (X**2 + Y**2) /
                                       (2 * sigma**2)) * np.exp(-(Y**2 + X**2) / (2 * sigma**2))
    '''
    co.data.hand_im = co.data.depth_im[co.points.y_hand:co.points.y_hand + co.meas.h_hand,
                                       co.points.x_hand:co.points.x_hand + co.meas.w_hand]
    co.contours.cropped_hand = co.contours.hand - \
        np.array([co.points.x_hand, co.points.y_hand])
    co.data.hand_shape = np.ones(co.data.hand_im.shape)
    cv2.drawContours(co.data.hand_shape, [co.contours.cropped_hand], 0, 0, -1)
    co.data.hand_shape[co.data.hand_shape == 0] = co.data.hand_im[
        co.data.hand_shape == 0]
    co.data.hand_shape[co.data.hand_shape == 1] = co.data.hand_shape[co.data.hand_shape == 1] * \
        (np.max(co.data.hand_im[co.data.hand_shape < 1]) +
         np.min(co.data.hand_im[co.data.hand_shape < 1])) / 2
    if np.max(co.data.hand_shape) - np.min(co.data.hand_shape) == 0:
        return 'zero hand_shape found, probably nothing'
    co.data.hand_shape = (co.data.hand_shape - np.min(co.data.hand_shape)) / \
        (np.max(co.data.hand_shape) - np.min(co.data.hand_shape))
    co.data.color_hand = np.zeros(co.data.hand_shape.shape + (3,))
    co.data.cropped_color_im = co.data.color_im[
        co.points.y_hand:co.points.y_hand + co.meas.h_hand, co.points.x_hand:co.points.x_hand
        + co.meas.w_hand, :]
    co.data.color_hand[co.data.hand_shape < 1] = co.data.cropped_color_im[
        co.data.hand_shape < 1]

    # c_hand[np.tile(hand_shape[:,:,None],(1,1,3))>0]=color_im[ \
    # np.tile(hand_shape[:,:,None],(1,1,3))>0]

    '''
    cv2.drawContours(rgb_final, [hand_contour], 0, [0, 1, 1], -1)
    for defect in hand_convexity_defects.squeeze():
     if defect[3]/256 >5:
      cv2.circle(rgb_final, tuple(hand_contour[defect[2]].squeeze()),3,[1, 1, 0],-1)
    '''

    return co.data.hand_shape


def initialise_global_vars():
    '''Initialise some of the global class objects'''
    co.models.noise_model = moda.init_noise_model()
    co.edges.load_calib_data()
    co.counters.aver_count = 0
    co.data.depth_im = co.data.initial_im_set[:, :, 0]
    co.meas.min1 = np.min(co.data.depth_im[co.data.depth_im > 0])
    co.data.depth_im[co.data.depth_im ==
                     co.meas.min1] = np.max(co.data.depth_im)
    co.meas.min2 = np.min(co.data.depth_im[co.data.depth_im > 0])
    co.meas.least_resolution = co.meas.min2 - co.meas.min1
    co.thres.depth_thres = co.meas.least_resolution * co.CONST['depth_thres']
    co.meas.imy, co.meas.imx = co.data.depth_im.shape
    return co.counters, co.data, co.masks, co.meas, co.models, co.thres


class Kinect(object):
    '''Kinect Processing Class'''

    def __init__(self):
        co.meas.nprange = np.arange((co.meas.imy + 2) * (co.meas.imx + 2))
        self.initial_im_set_list = []
        self.vars = self.initial_im_set_list
        global PROCESS
        print 'Detection method is set to:', co.CONST['detection_method']
        if co.CONST['detection_method'] == 'segmentation':
            print 'Segmentation Data file is: ' +\
                co.CONST['segmentation_data'] + '.pkl'
            co.segclass.exists_previous_segmentation = os.path.isfile(
                co.CONST['segmentation_data'] + '.pkl')
            if co.segclass.exists_previous_segmentation:
                print 'Existing previous background segmentation'
                print 'Loading from memory...'
                (co.segclass, co.meas) = pickle.load(
                    open(co.CONST['segmentation_data'] + '.pkl', 'rb'))
                co.segclass.exists_previous_segmentation = True
                print 'Loaded previous setup'
        print 'Streaming is set to:', co.CONST['stream']
        if co.CONST['stream'] == 'live':
            print 'Initialising Kinect Stream...'
            node = roslaunch.core.Node("kinect2_bridge", "kinect2_bridge")
            # rospy.set_param('fps_limit',10)
            launch = roslaunch.scriptapi.ROSLaunch()
            launch.start()
            PROCESS = launch.launch(node)
            if PROCESS.is_alive():
                print 'Starting subscribers to Kinect..'
        elif co.CONST['stream'] == 'recorded':
            print 'Starting streaming of rosbag file:', co.CONST['bag_path']
            PROCESS = subprocess.Popen(
                'rosbag play -l -q ' +
                co.CONST['bag_path'],
                stdin=subprocess.PIPE, shell=True,
                preexec_fn=os.setsid)
            signal.signal(signal.SIGTSTP, rosbag_handler)
        signal.signal(signal.SIGINT, signal_handler)
        self.image_pub = rospy.Publisher("results_topic", Image, queue_size=1)
        self.depth_sub = mfilters.Subscriber(
            "/kinect2/qhd/image_depth_rect", Image)
        self.color_sub = mfilters.Subscriber(
            "/kinect2/qhd/image_color_rect", Image)

        self.image_ts = mfilters.TimeSynchronizer(
            [self.depth_sub, self.color_sub], 30)
        self.image_ts.registerCallback(
            self.callback)
        self.bridge = CvBridge()

    def detection_with_noise_model(self):
        '''
        Use noise model based moving object detection
        '''
        self.initial_im_set_list = self.vars
        if co.counters.im_number == 0:
            co.meas.imy, co.meas.imx = co.data.depth_im.shape
            co.meas.nprange = np.arange(
                (co.meas.imy + 2) * (co.meas.imx + 2))
            co.data.depth3d = np.tile(
                co.data.depth_im[:, :, None], (1, 1, 3))
        if co.counters.im_number <= co.CONST['calib_secs'] * co.CONST['framerate'] - 1:
            self.initial_im_set_list.append(co.data.depth_im)
        elif co.counters.im_number == co.CONST['calib_secs'] * co.CONST['framerate']:
            co.data.initial_im_set = np.rollaxis(
                np.array(self.initial_im_set_list), 0, 3)
            co.counters, co.data, co.masks, \
                co.meas, co.models, co.thres = initialise_global_vars()
            frame_process()
            cv2.imwrite('segments.jpg', co.segclass.segment_values)
        else:
            # exit()
            if co.counters.save_im_num > co.lims.max_im_num_to_save - 1:
                co.data.depth[co.counters.save_im_num %
                              co.lims.max_im_num_to_save] = co.data.depth_im
            else:
                co.data.depth.append(co.data.depth_im)
            co.counters.save_im_num += 1
            result = frame_process()
            if not isinstance(result, str):
                print 'SUCCESS'

        co.counters.im_number += 1
        if co.CONST['results'] == 'display':
            co.im_results.show_results('stdout', co.CONST['framerate'])
        return

    def detection_with_segmentation(self):
        '''
        Use scene segmentation and center of mass method to find moving object
        '''
        self.initial_im_set_list = self.vars
        co.data.depth3d = np.tile(co.data.depth_im[:, :, None], (1, 1, 3))

        co.data.uint8_depth_im = (255 * co.data.depth_im).astype(np.uint8)
        if co.counters.im_number == 0:
            co.data.reference_uint8_depth_im = co.data.uint8_depth_im.copy()
            co.meas.imy, co.meas.imx = co.data.depth_im.shape
            co.meas.nprange = np.arange((co.meas.imy + 2) * (co.meas.imx + 2))
            co.segclass.all_objects_im = np.zeros_like(co.data.depth_im)
        if co.counters.im_number < co.CONST['framerate'
                                        ] * co.CONST['calib_secs']:
            self.initial_im_set_list.append(co.data.depth_im)
        elif co.counters.im_number ==\
                co.CONST['framerate'] * co.CONST['calib_secs']:
            co.data.initial_im_set = np.rollaxis(
                np.array(self.initial_im_set_list), 0, 3)
            moda.extract_background_values()
        objects_mask = moda.detection_by_scene_segmentation()

        '''
        try:
            lapl=np.abs(cv2.Laplacian(co.data.depth_im*objects_mask,cv2.CV_64F))
            lapl=(lapl-np.max(lapl))/(np.max(lapl)-np.min(lapl))
            co.im_results.images.append(lapl)
        except TypeError:
            a=1
        '''
        co.counters.im_number += 1
        return

    def detection_with_mog2(self):
        co.data.depth3d = np.tile(co.data.depth_im[:, :, None], (1, 1, 3))
        moda.detection_by_mog2()
        co.counters.im_number+=1

    def callback(self, depth, color):
        '''callback function'''
        co.im_results.images = []
        try:
            co.data.depth_im = self.bridge.imgmsg_to_cv2(
                depth, desired_encoding="passthrough")
            co.data.color_im = self.bridge.imgmsg_to_cv2(
                color, desired_encoding="passthrough")
            co.data.depth_raw= co.data.depth_im.copy()
            co.data.depth_raw *= co.data.depth_raw < co.CONST['max_depth']*co.CONST['noise_thres']
            co.data.depth_im = (co.data.depth_im) / float(co.CONST['max_depth'])

        except CvBridgeError as err:
            print err

        if co.edges.exists_lim_calib_image:
            co.noise_proc.remove_noise()
            choices = {'segmentation': self.detection_with_segmentation,
                       'noise model': self.detection_with_noise_model,
                       'mog2': self.detection_with_mog2}
            choices.get(co.CONST['detection_method'], 'default')()
            if co.CONST['save'] == 'y':
                if co.counters.save_im_num > co.lims.max_im_num_to_save - 1:
                    co.data.depth[co.counters.save_im_num %
                                  co.lims.max_im_num_to_save] = co.data.depth_im
                else:
                    co.data.depth.append(co.data.depth_im)
                co.counters.save_im_num += 1

            if co.CONST['results'] == 'display':
                co.im_results.show_results(
                    'ros', [self.image_pub, self.bridge])
        else:
            if co.counters.im_number <= 10 * co.CONST['framerate'] - 1:
                self.initial_im_set_list.append(co.data.depth_im)
            elif co.counters.im_number == 10 * co.CONST['framerate']:
                co.edges.construct_calib_edges(self.initial_im_set_list,
                                               write=True)
                cv2.imshow('Calibration_Edges', co.edges.calib_edges)
                cv2.waitKey(0)
                sys.exit()
            co.counters.im_number += 1


def main():
    """Main Function"""
    co.lims.max_im_num_to_save = co.CONST['max_depth_im_num_to_save']
    co.thres.lap_thres = co.CONST['lap_thres']
    edges_path, frame_path = co.edges.load_calib_data()
    if co.CONST['read'] == 'd':
        co.paths.depth = co.CONST['path_depth']
        # path_color=co.CONST['path_color']
        co.data.depth = hf.im_load(co.paths.depth, 'Depth')
        #data_color=hf.im_load(path_color, 'Color')
    if co.CONST['read'] == 'f':
        try:
            co.data.depth = np.load(co.CONST['save_depth'])
        except (IOError,EOFError):
            raise Exception('No valid file saved as '+co.CONST['saved_depth'])
        if not co.edges.exists_lim_calib_image:
            ans = raw_input('No edges calibration files available, '+
                  'do you want to proceed without them?(Y,n)')
            if ans=='N' or ans=='n':
                exit()
            else:
                co.edges.construct_calib_edges(whole_im=True,
                                               img=co.data.depth[:,:,0])
        # data_color=np.load(co.CONST['save_color']+'.npy')
    elif co.CONST['read'] == 'k':
        Kinect()
        rospy.init_node('kinect_stream', anonymous=True)
        if not co.edges.exists_lim_calib_image:
            print 'No image exists for calibration of frame edges'
            print 'Please move randomly kinect in space' \
                'for the next 10 seconds (DONT BREAK IT) and wait for' \
                'program to exit'
        rospy.spin()
        print "Shutting down"
        if co.CONST['stream'] == 'live':
            PROCESS.stop()
        else:
            # PROCESS.terminate()
            os.killpg(PROCESS.pid, signal.SIGINT)
    if co.CONST['save'] == 'y':
        np.save(co.CONST['save_depth'], co.data.depth)
        #np.save(co.CONST['save_color'], data_color)
    if co.CONST['read'] != 'k':
        co.lims.init_n = 9
        # n=2*co.CONST['framerate']
        co.data.initial_im_set = co.data.depth[:, :, :co.lims.init_n]
        co.counters, co.data, co.masks, co.meas, co.models, co.thres = initialise_global_vars()
        for count in range(co.lims.init_n, np.array(co.data.depth).shape[2]):
            co.data.depth_im = np.array(co.data.depth)[:, :, count]
            frame_process()
            cv2.waitKey(1000 / co.CONST['framerate'])


if __name__ == '__main__':
    main()
