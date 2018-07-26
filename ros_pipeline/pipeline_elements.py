"""
Pipeline Elements Module
"""
#pylint: disable=arguments-differ
import shlex
from subprocess import Popen
import signal
import sys
import traceback
import numpy as np
import cv2
from ros_pipeline.base import ROSPipelineElement
from hand_segmentation_alg import HandExtractor as HExtractor
from hand_segmentation_alg import FindArmSkeleton
from moving_object_detection_alg import Mog2
import classifiers as cs

class KinectSubscriber(ROSPipelineElement):
    """
    Kinect Subscriber Pipeline Element
    """
    def __init__(self, loglevel='INFO'):
        '''
        Subscriber to Kinect bridge
        '''
        super(KinectSubscriber, self).__init__('Subscriber', loglevel)
        self.depth = None
        self.kinect_process = Popen(shlex.split(
            'roslaunch kinect2_bridge kinect2_bidge.launch'))

    def get(self, depth_message):
        self.depth = self.get_array(depth_message)

    def post(self):
        self.post_array('depth', self.depth)

    def stop(self):
        self.kinect_process.send_signal(signal.SIGINT)
        super(KinectSubscriber, self).stop()


class ArmExtractor(ROSPipelineElement):
    '''
    Preprocessor of supplied depth image
    '''
    def __init__(self, loglevel='INFO'):
        super(ArmExtractor, self).__init__('ArmExtractor', loglevel)
        self.arm = None
        self.mog2 = Mog2()
        self.open_kernel = np.ones((5, 5), np.uint8)

    def preprocess_depth(self, frame):
        '''
        preprocess given frame
        '''
        try:
            mog2_res = self.mog2.run(False,
                                     frame.astype(np.float32))
            if mog2_res is None:
                return False
            mask1 = cv2.morphologyEx(
                mog2_res.copy(),
                cv2.MORPH_OPEN,
                self.open_kernel)
            check_sum = np.sum(mask1 > 0)
            if not check_sum or check_sum == np.sum(frame > 0):
                return False
            _, contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
            cont_ind = np.argmax([cv2.contourArea(contour) for
                                  contour in contours])
            final_mask = np.zeros_like(mask1)
            cv2.drawContours(final_mask, contours, cont_ind, 1, -1)
            # cv2.imshow('test',mask1*255)
            # cv2.waitKey(10)
            return frame * final_mask
        except BaseException:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type,
                                      exc_value,
                                      exc_traceback, limit=10, file=sys.stdout)
            return False



class Skeletonizer(ROSPipelineElement):
    """
    Arm Skeletonizer Pipeline Element
    """
    def __init__(self, loglevel='INFO'):
        super(Skeletonizer, self).__init__(self, 'Skeletonizer', loglevel)
        self.skeleton = FindArmSkeleton()
        self.skeleton_found = False


    def get(self, arm, arm_contour):
        arm = self.get_array(arm)
        arm_contour = self.get_array(arm_contour)
        self.skeleton_found = self.skeleton.run(arm, arm_contour,
                                                'longest_ray')

    def post(self):
        if self.skeleton_found:
            self.post_array(
                'skeleton', np.array(
                    self.skeleton.skeleton, np.int32))
            self.post_array(
                'surrounding_skel', np.array(
                    self.skeleton.surrounding_skel, np.int32))





class HandExtractor(ROSPipelineElement):
    '''
    Hand Extractor from arm pipeline element
    '''
    def __init__(self, loglevel='INFO'):
        super(HandExtractor, self).__init__('HandExtractor', loglevel)
        self.hand_extractor = HExtractor()
        self.extracted_hand = False
        self.hand = None

    def get(self, arm, skeleton, surrounding_skel):
        arm = self.get_array(arm)
        skeleton = self.get_array(skeleton)
        surrounding_skel = self.get_array(surrounding_skel)
        self.extracted_hand = False
        if self.hand_extractor.run(arm, skeleton, surrounding_skel):
            self.extracted_hand = True
            self.hand = self.hand_extractor.hand_mask * arm

    def post(self):
        if self.extracted_hand:
            self.post_array('hand', self.hand)
            self.post_time_reference('angle', self.hand_extractor.angle)
            self.post_time_reference('center', self.hand_extractor.hand_start)


class GestureRecognition(ROSPipelineElement):
    """
    Gesture Recognition pipeline element
    """
    def __init__(self, loglevel='INFO'):
        super(GestureRecognition, self).__init__(
            'GestureRecogntion', loglevel)
        self.used_classifier = cs.construct_passive_actions_classifier(
            train=False, test=False,
            visualize=False,
            test_against_all=True,
            for_app=True)
        self.classified = False

    def get(self, hand, angle, center):
        scores_exist, _ = self.used_classifier.run_testing(hand,
                                                           derot_angle=angle,
                                                           derot_center=center,
                                                           online=True)
        self.classified = False
        if (self.used_classifier.recognized_classes is not None
                and self.used_classifier.recognized_classes
                and scores_exist):
            self.classified = True

    def post(self):
        try:
            self.post_time_reference('gesture',
                                     self.used_classifier.train_classes[
                                         self.used_classifier.recognized_classes[
                                             -1]])
        except TypeError:
            self.post_time_reference('gesture',
                                     self.used_classifier.recognized_classes[
                                         -1].name)
