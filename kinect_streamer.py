import os, sys, traceback
import logging
import signal
import numpy as np
import cv2
import rospy
import roslaunch
import message_filters as mfilters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, TimeReference
from math import pi
import time
import class_objects as co
import extract_and_process_rosbag as epr
import classifiers as cs
import moving_object_detection_alg as moda
#import poses_to_actions as p2a
KINECT = None

class Time(object):
    def __init__(self):
        self.secs = 0
        self.nsecs = 0

def signal_handler(sig, frame):
    '''
    Signal handler for CTRL-C interrupt (SIGINT)
    '''
    LOG.info('Mean frame processing time: '+
                 str(KINECT.time) +' ms')
    KINECT.process.stop()
    sys.exit(0)


from canvas import KinectSubscriber

class KinectPreProcessor(KinectSubscriber):
    '''Kinect Processing Class'''

    MAX_TIMED_FRAMES = 10000
    def __init__(self):
        #Kinect requirements
        # Hand action recognition
        super(KinectProcessor, self).__init__('KinectProcessor',self,
                                              self.data_queue, self.loglevel,
                                              co.CONST['channel'], Image)
        self.mog2 = moda.Mog2()
        self.used_classifier = cs.construct_passive_actions_classifier(train=False, test=False,
                                                        visualize=False,
                                                        test_against_all=True,
                                                        for_app=True)
        self.open_kernel = np.ones((5, 5), np.uint8)
        self.cnt_timed_frames = 0
        self.time = 0
        ####
        node = roslaunch.core.Node("kinect2_bridge", "kinect2_bridge")
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        self.process = launch.launch(node)
        if self.process.is_alive():
            print('Starting subscribers to Kinect..')
        signal.signal(signal.SIGINT, signal_handler)
        self.image_sub = rospy.Subscriber(
            co.CONST['channel'], Image, self.callback, queue_size=10)
        self.image_pub = rospy.Publisher("hand", Image,queue_size=10)
        self.class_pub = rospy.Publisher("class", TimeReference,queue_size=10)
        self.skel_pub = rospy.Publisher("skeleton", Image,queue_size=10)
        rospy.init_node('kinect_stream', anonymous=True)
        self.bridge = CvBridge()
        ###


    def callback(self, data):
        '''
        Callback function, <data> is the depth image
        '''
        try:
            time1 = time.time()
            try:
                frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            except CvBridgeError as err:
                print(err)
                return
            mog2_res = self.mog2.run(False,
                                     frame.astype(np.float32))
            if mog2_res is None:
                return
            mask1 = cv2.morphologyEx(mog2_res.copy(), cv2.MORPH_OPEN, self.open_kernel)
            check_sum = np.sum(mask1 > 0)
            if not check_sum or check_sum == np.sum(frame > 0):
                return
            _, contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            cont_ind = np.argmax([cv2.contourArea(contour) for
                                  contour in contours])
            final_mask = np.zeros_like(mask1)
            cv2.drawContours(final_mask, contours, cont_ind, 1, -1)
            #cv2.imshow('test',mask1*255)
            #cv2.waitKey(10)
            frame = frame * final_mask
            scores_exist,_ = self.used_classifier.run_testing(frame,
                                            online=True)
            #DEBUGGING
            #cv2.imshow('test',(frame%256).astype(np.uint8))
            #cv2.waitKey(10)
            time2 = time.time()
            self.cnt_timed_frames += 1
            if self.cnt_timed_frames < self.MAX_TIMED_FRAMES:
                self.time += time2 - time1
            if (self.used_classifier.recognized_classes is not None
                and len(self.used_classifier.recognized_classes)>0
                and scores_exist):
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(
                    self.used_classifier.frames_preproc.hand_img, "passthrough"))
                msg = TimeReference()
                try:
                    msg.source = self.used_classifier.train_classes[
                        self.used_classifier.recognized_classes[-1]]
                except TypeError:
                    msg.source = self.used_classifier.recognized_classes[-1].name
                msg.time_ref = Time()
                msg.time_ref.secs = float(self.time/min(self.cnt_timed_frames, self.MAX_TIMED_FRAMES)
                                          if self.cnt_timed_frames else 0)
                self.class_pub.publish(msg)
                self.skel_pub.publish(self.bridge.cv2_to_imgmsg(
                             np.atleast_2d(np.array(self.used_classifier.
                             frames_preproc.skeleton.skeleton,np.int32))))
        except Exception as e:
             exc_type, exc_value, exc_traceback = sys.exc_info()
             traceback.print_exception(exc_type,
                                exc_value,
                                exc_traceback, limit=10, file=sys.stdout)





class Kinect_old(object):
    '''Kinect Processing Class'''

    MAX_TIMED_FRAMES = 10000
    def __init__(self):
        #Kinect requirements
        # Hand action recognition
        self.mog2 = moda.Mog2()
        self.used_classifier = cs.construct_passive_actions_classifier(train=False, test=False,
                                                        visualize=False,
                                                        test_against_all=True,
                                                        for_app=True)
        self.open_kernel = np.ones((5, 5), np.uint8)
        self.cnt_timed_frames = 0
        self.time = 0
        ####
        node = roslaunch.core.Node("kinect2_bridge", "kinect2_bridge")
        launch = roslaunch.scriptapi.ROSLaunch()
        launch.start()
        self.process = launch.launch(node)
        if self.process.is_alive():
            print('Starting subscribers to Kinect..')
        signal.signal(signal.SIGINT, signal_handler)
        self.image_sub = rospy.Subscriber(
            "/kinect2/sd/image_depth",Image,self.callback, queue_size=10)
        self.image_pub = rospy.Publisher("hand", Image,queue_size=10)
        self.class_pub = rospy.Publisher("class", TimeReference,queue_size=10)
        self.skel_pub = rospy.Publisher("skeleton", Image,queue_size=10)
        rospy.init_node('kinect_stream', anonymous=True)
        self.bridge = CvBridge()
        ###

    def callback(self, data):
        '''
        Callback function, <data> is the depth image
        '''
        try:
            time1 = time.time()
            try:
                frame = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            except CvBridgeError as err:
                print(err)
                return
            mog2_res = self.mog2.run(False,
                                     frame.astype(np.float32))
            if mog2_res is None:
                return
            mask1 = cv2.morphologyEx(mog2_res.copy(), cv2.MORPH_OPEN, self.open_kernel)
            check_sum = np.sum(mask1 > 0)
            if not check_sum or check_sum == np.sum(frame > 0):
                return
            _, contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            cont_ind = np.argmax([cv2.contourArea(contour) for
                                  contour in contours])
            final_mask = np.zeros_like(mask1)
            cv2.drawContours(final_mask, contours, cont_ind, 1, -1)
            # cv2.imshow('test',mask1*255)
            # cv2.waitKey(10)
            frame = frame * final_mask
            scores_exist,_ = self.used_classifier.run_testing(frame,
                                            online=True)
            # DEBUGGING
            # cv2.imshow('test',(frame%256).astype(np.uint8))
            # cv2.waitKey(10)
            time2 = time.time()
            self.cnt_timed_frames += 1
            if self.cnt_timed_frames < self.MAX_TIMED_FRAMES:
                self.time += time2 - time1
            if (self.used_classifier.recognized_classes is not None
                and len(self.used_classifier.recognized_classes)>0
                and scores_exist):
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(
                    self.used_classifier.frames_preproc.hand_img, "passthrough"))
                msg = TimeReference()
                try:
                    msg.source = self.used_classifier.train_classes[
                        self.used_classifier.recognized_classes[-1]]
                except TypeError:
                    msg.source = self.used_classifier.recognized_classes[-1].name
                msg.time_ref = Time()
                msg.time_ref.secs = float(self.time/min(self.cnt_timed_frames, self.MAX_TIMED_FRAMES)
                                          if self.cnt_timed_frames else 0)
                self.class_pub.publish(msg)
                self.skel_pub.publish(self.bridge.cv2_to_imgmsg(
                             np.atleast_2d(np.array(self.used_classifier.
                             frames_preproc.skeleton.skeleton,np.int32))))
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type,
                                      exc_value,
                                      exc_traceback, limit=10, file=sys.stdout)

LOG = logging.getLogger('__name__')
CH = logging.StreamHandler()
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel('INFO')
KINECT = Kinect()
def main():
    rospy.spin()
if __name__=='__main__':
    main()
