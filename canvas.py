import logging
import sys
import traceback
from ast import literal_eval
import abc
import time
from math import pi, cos, sin
import numpy as np
import cv2
import rospy
import wx
from cv_bridge import CvBridge, CvBridgeError
# pylint: disable=unused-import
from sensor_msgs.msg import Image
from sensor_msgs.msg import TimeReference
# pylint: enable=unused-import
import threading
from subprocess import Popen
import shlex
import signal
import message_filters as mfilters
import class_objects as co
from kivy.multistroke import Recognizer
from kivy.clock import Clock as clock
from kivy.vector import Vector
from numpy.linalg import norm
LOGLEVEL = 'INFO'  # 'DEBUG'
LOG = logging.getLogger(__name__)
FORMAT = '%(funcName)20s(%(lineno)s)-%(levelname)s:%(message)s'
CH = logging.StreamHandler()
CH.setFormatter(logging.Formatter(FORMAT))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel('INFO')
SHAPE = (500, 900)

#  pylint: disable=W0221


def getbitmap(main_panel, img):
    image = wx.Image(img.shape[1], img.shape[0])
    image.SetData(img.tostring())
    wx_bitmap = image.ConvertToBitmap()
    return wx_bitmap


class Time(object):
    '''
    Class to keep time
    '''

    def __init__(self):
        self.secs = 0
        self.nsecs = 0


class CreateEvent(wx.PyCommandEvent):
    '''
    Event signaler
    '''

    def __init__(self, *args, **kwargs):
        wx.PyCommandEvent.__init__(self, *args, **kwargs)


EVT_ROS_TYPE = wx.NewEventType()
EVT_ROS = wx.PyEventBinder(EVT_ROS_TYPE, 1)

EVT_STRK_TYPE = wx.NewEventType()
EVT_STRK = wx.PyEventBinder(EVT_STRK_TYPE, 1)


class HandData(object):
    def __init__(self):
        self.hand = None
        self.skel = None
        self.class_name = None
        self.fps = None

    def add(self, hand, skel, class_name, fps):
        self.hand = hand
        self.skel = skel
        self.class_name = class_name
        self.fps = fps


class StrokeData(object):
    def __init__(self):
        self.gesture = None
        self.found_gesture = False
        self.strokes = None

    def add(self,
            gesture,
            found_gesture,
            strokes):
        self.gesture = gesture
        self.found_gesture = found_gesture
        self.strokes = strokes



class ROSPipelineElement(threading.Thread):

    def __init__(self, name, loglevel='INFO'):
        '''
        `name`:: key given in ROSNodes in config.yaml
        `loglevel`:: the logger level, defaults to `INFO`
        '''
        threading.Thread.__init__(self)
        name = co.CONST['ROSNodes'][name]['name']
        subscribe_channels = co.CONST['ROSNodes'][name]['subscribing_to']
        publish_channels = co.CONST['ROSNodes'][name]['publishing_to']
        self._name = name
        self._loglevel = loglevel
        (subscribe_channels,
         subscribe_types) = self._check_channels(subscribe_channels)

        (publish_channels,
         publish_types) = self._check_channels(publish_channels)
        if subscribe_channels is not None:
            self.image_ts = mfilters.TimeSynchronizer(
                [mfilters.Subscriber(channel,
                                     channel_type) for channel, channel_type in
                 zip(subscribe_channels, subscribe_types)], 30)
            self.image_ts.registerCallback(self.get_callback)
        if publish_channels is not None:
            self._publishers = {channel: rospy.Publisher(channel, channel_type)
                                for channel, channel_type
                                in zip(publish_channels, publish_types)}
        else:
            self._publishers = None
        self._stop_event = threading.Event()
        self.bridge = CvBridge()

    def _check_channels(self, channels):
        '''
        Check input channels integrity
        '''
        if channels is None:
            return channels, None
        assert isinstance(channels, (str, list)),\
            'Incorrect argument type for channels'
        if isinstance(channels, str):
            channels = [channels]
        channels_types = [literal_eval(co.CONST['ROSChannelsTypes'][channel])
                          for channel in channels]
        return channels, channels_types

    def get_publisher(self, channel):
        '''
        Get requested publisher object
        '''
        try:
            return self._publishers[channel]
        except KeyError:
            raise BaseException(
                'Invalid provided channel, publisher does not exist')
        except TypeError:
            if self._publishers is None:
                raise TypeError("No publishers exist"
                                ", wrong class initialization")

    def run(self):
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.spin()

    def _callback(self, *channels_data):
        self.get(*channels_data)
        self.post()

    def get(self, *channels_data):
        '''
        Callback to handle channels subscribers
        '''
        pass

    def post(self):
        '''
        Function to handle channels publishers
        '''
        pass

    def stop(self):
        '''
        Stop ROS Thread
        '''
        self._stop_event.set()

    def stopped(self):
        '''
        Check if ROS Thread is stopped
        '''
        return self._stop_event.is_set()

    def post_array(self, channel, array):
        '''
        Publish a numpy array to the given channel
        '''
        self.get_publisher(channel).publish(self.bridge.cv2_to_imgmsg(
            np.atleast_2d(array)))

    def get_array(self, ros_message):
        '''
        Get a numpy array from ros_message coming from a publisher,
        who used post_array(). If error shows up, ignore it.
        '''
        try:
            return self.bridge.imgmsg_to_cv2(ros_message,
                                             desired_encoding="passthrough")
        except CvBridgeError as err:
            print(err)
        return None

    def post_time_reference(self, channel, data):
        '''
        Publish a data as time reference source to the given channel
        '''
        msg = TimeReference()
        msg.source = data
        msg.time_ref = Time()
        t = rospy.Time.from_sec(time.time())
        seconds = t.to_sec()  # floating point
        nanoseconds = t.to_nsec()
        msg.time_ref.secs = seconds
        msg.time_ref.nsecs = nanoseconds
        self.get_publisher(channel).publish(msg)

    def get_time_reference(self, message):
        return message.source


class KinectSubscriber(ROSPipelineElement):
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
        self.stop()


class ArmExtractor(ROSPipelineElement):
    def __init__(self, loglevel='INFO'):
        '''
        Preprocessor of supplied depth image
        '''
        super(ArmExtractor, self).__init__('ArmExtractor', loglevel)
        self.arm = None

    def preprocess_depth(self, frame):
        try:
            mog2_res = self.mog2.run(False,
                                     frame.astype(np.float32))
            if mog2_res is None:
                return
            mask1 = cv2.morphologyEx(
                mog2_res.copy(),
                cv2.MORPH_OPEN,
                self.open_kernel)
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
            return frame * final_mask
        except BaseException:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type,
                                      exc_value,
                                      exc_traceback, limit=10, file=sys.stdout)


from hand_segmentation_alg import FindArmSkeleton

class Skeletonizer(ROSPipelineElement):
    def __init__(self, loglevel='INFO'):
        super(Skeletonizer, self).__init__(self, 'Skeletonizer', loglevel)
        self.skeleton = FindArmSkeleton()


    def get(self, arm, arm_contour):
        arm = self.get_array(arm)
        arm_contour = self.get_array(arm_contour)
        self.skeleton_found = self.skeleton.run(arm, arm_contour,
                                                'longest_ray')

    def post(self):
        if self.skeleton_found:
            self.post_time_reference('angle', self.skeleton.angle)
            self.post_time_reference('center', self.skeleton.hand_start)
            self.post_array(
                'skeleton', np.array(
                    self.skeleton.skeleton, np.int32))



from hand_segmentation_alg import HandExtractor


class HandExtractor(ROSPipelineElement):
    def __init__(self, loglevel='INFO'):
        super(HandExtractor, self).__init__('HandExtractor', loglevel)
        self.hand_extractor = HandExtractor()
        self.extracted_hand = False

    def get(self, arm, skeleton, surrounding_skel):
        arm = self.get_array(arm)
        skeleton = self.get_array(skeleton)
        surrounding_skel = self.get_array(surrounding_skel)
        self.extracted_hand = False
        if self.hand_extractor.run(*args):
            self.extracted_hand = True
            self.hand = self.hand_extractor.hand_mask * arm

    def post(self):
        self.post_array('hand', self.hand)


class GestureRecognition(ROSPipelineElement):
    def __init__(self, loglevel='INFO'):
        super(GestureRecognition, self).__init__(
            'GestureRecogntion', loglevel)
        self.used_classifier = cs.construct_passive_actions_classifier(
            train=False, test=False,
            visualize=False,
            test_against_all=True,
            for_app=True)

    def get(self, hand, skeleton):
        scores_exist, _ = self.used_classifier.run_testing(hand,
                                                           derot_angle=angle,
                                                           derot_center=center,
                                                           online=True)
        # DEBUGGING
        # cv2.imshow('test',(frame%256).astype(np.uint8))
        # cv2.waitKey(10)
        time2 = time.time()
        self.cnt_timed_frames += 1
        if self.cnt_timed_frames < self.MAX_TIMED_FRAMES:
            self.time += time2 - time1
        self.classified = False
        if (self.used_classifier.recognized_classes is not None
            and len(self.used_classifier.recognized_classes) > 0
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




class StrokeRecognition(ROSPipelineElement):
    def __init__(self, loglevel='INFO'):
        super(StrokeRecognition, self).__init__('StrokeRecognizer', loglevel)
        self.strokes = [[]]
        self.time = time.time()
        self.gdb = Recognizer()
        self.logger = logging.getLogger('StrokeRecognition')
        self.logger.setLevel(loglevel)
        self.logger.addHandler(CH)
        self.logger.info('Creating Gestures Templates..')
        circle = [[Vector(int(10 + 10 * cos(t)),
                          int(10 + 10 * sin(t)))
                   for t in np.linspace(0, 2 * pi, 8)]]
        self.gdb.add_gesture('Circle', circle, strokes_sensitive=True)
        # Horizontal or vertical lines give error
        self.gdb.add_gesture('Line', [[Vector(10, 60), Vector(40, 50)]],
                             strokes_sensitive=True, priority=50)
        self.gdb.add_gesture('Triangle', [[Vector(10, 10),
                                           Vector(15, 15),
                                           Vector(20, 20),
                                           Vector(20, 20),
                                           Vector(25, 15),
                                           Vector(30, 10),
                                           Vector(30, 10),
                                           Vector(20, 10),
                                           Vector(10, 10)]],
                             strokes_sensitive=False,
                             orientation_sensitive=False,
                             permute=False)
        self.gdb.add_gesture('Rectangle', [[Vector(10, 10),
                                            Vector(10, 15),
                                            Vector(10, 20),
                                            Vector(10, 20),
                                            Vector(15, 20),
                                            Vector(20, 20),
                                            Vector(20, 20),
                                            Vector(20, 15),
                                            Vector(20, 10),
                                            Vector(20, 10),
                                            Vector(15, 10),
                                            Vector(10, 10)]],
                             strokes_sensitive=False,
                             orientation_sensitive=False,
                             permute=False)
        self.logger.info('Templates created')
        self.found_gesture = False
        self.gesture = None
        self._data = data
        self._parent = parent
        self.ran = False
        self.gdb.bind(on_search_complete=self.search_stop)
        self.dist_thres = 0
        self.time_thres = 2

    def get(self, skel, class_name):
        if self._parent.initialized and not self._parent.write_mode:
            try:
                skel = self.bridge.imgmsg_to_cv2(skel,
                                                 desired_encoding='passthrough')
                action = class_name.source
                if action == 'Index':
                    self.ran = False
                    self.add_stroke_vector(skel[-1, -1, 1],
                                           skel[-1, -1, 0])
                    self.time = time.time()
                elif action == 'Palm':
                    self.add_stroke()
                if not self.ran and time.time() - self.time > self.time_thres:
                    self.ran = True
                    if len(self.strokes[0]) > 0:
                        self.progress = self.gdb.recognize(self.strokes[:-1])
                    # removing last empty sublist          ^
                    clock.tick()
                    self._data.add(gesture=self.gesture,
                                   found_gesture=self.found_gesture,
                                   strokes=self.strokes)
                    self.strokes = [[]]
                    evt = CreateEvent(EVT_STRK_TYPE, -1)
                    wx.PostEvent(self._parent, evt)
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                traceback.print_exception(exc_type,
                                          exc_value,
                                          exc_traceback, limit=2, file=sys.stdout)
                print(e)
        else:
            if len(self.strokes[0]) > 0:
                self.progress = self.gdb.recognize(self.strokes[:-1])
            self.strokes = [[]]

    def post(self):
        if self.found_gesture:
            self.post_time_reference('stroke', self.gesture)

    def search_stop(self, gdb, pt):
        best = pt.best
        dist = best['dist']
        self.gesture = best['name']
        if dist > self.dist_thres and self.gesture is not None:
            self.found_gesture = True
            self.logger.debug('Gesture found as ' + str(self.gesture) +
                              ' having cos distance ' + str(dist))
        else:
            self.found_gesture = False
            self.logger.debug('Gesture not found. \n\tBest candidate was ' +
                              str(self.gesture) + ' with cos distance:' + str(dist))
        self.logger.debug('\tThe number of strokes was ' +
                          str(len(self.strokes) - 1))
        self.logger.debug('\tStrokes:')
        for stroke in self.strokes[:-1]:
            self.logger.debug('\t\t' + str(stroke))

    def add_stroke(self):
        if len(self.strokes[-1]
               ) > 0:  # add stroke only if last stroke is not empty
            self.strokes.append([])

    def add_stroke_vector(self, xpoint, ypoint):
        if len(self.strokes[-1]) > 0:
            dist_check = norm(np.diff([np.array(self.strokes[-1][-1]),
                                       np.array([xpoint, ypoint])], axis=0)) < 10
        else:
            dist_check = True
        if dist_check:
            self.strokes[-1].append(Vector(xpoint, ypoint))


class KinectSubscriber(threading.Thread):
    def __init__(self, name, parent, data_queue, loglevel='INFO', channels=None,
                 channels_types=None):
        '''
        `name`:: arbitrary name for the subscriber. Must be unique
        `parent`:: the parent
        `data_queue`:: the provided data to the thread, must be a queue object
        `loglevel`:: the logger level, defaults to `INFO`
        `channels`:: the channels list
        `channels_types`:: the channels types , imported from sensor_msgs.msg
        '''
        threading.Thread.__init__(self)
        self._name = name
        self._parent = parent
        self._data = data_queue
        self._loglevel = loglevel
        assert channels is not None, "Input argument channels must be given"
        assert isinstance(name, str), "Given name is not a string"
        assert channels_types is not None, "Input arguments channels types must be given"
        if isinstance(channels, str):
            channels = [channels]
        if not isinstance(channels_types, list):
            channels_types = [channels_types]
        self.image_ts = mfilters.TimeSynchronizer(
            [mfilters.Subscriber(channel,
                                 channel_type) for channel, channel_type in
             zip(channels, channels_types)], 30)
        self.image_ts.registerCallback(self.get_callback)
        self.bridge = CvBridge()
        self._stop_event = threading.Event()

    def run(self):
        rospy.init_node(self._name, anonymous=True, disable_signals=True)
        rospy.spin()

    @abc.abstractmethod
    def get_callback(self, *channels_data):
        '''
        Callback to handle channels data
        '''
        pass

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()


"""
class OldROSThread(threading.Thread):
    def __init__(self, parent, data, loglevel):
        threading.Thread.__init__(self)
        self._parent = parent
        self._data = data
        self._loglevel = loglevel
        self.logger = logging.getLogger('ROSThread')
        self.logger.addHandler(CH)
        self.logger.setLevel(loglevel)
    def run(self):
        '''
        Overrides Thread.run . Call Thread.start(), not this.
        '''
        self.logger.info('Initializing ROS subscriber..')
        subscriber = ROSSubscriber(self._parent,
                                   self._data,
                                   self._loglevel)
        self.logger.info('Initializing node..')
        rospy.init_node('ROSThread', anonymous=True, disable_signals=True)
        self.logger.info('Waiting for input..')
        rospy.spin()
"""


class Canvas(wx.Window):
    def __init__(self, parent, data, *args, **kwargs):
        self.init = True
        wx.Window.__init__(self, parent, *args, **kwargs)
        [self.height, self.width] = data.shape[:2]
        self.SetMinSize(wx.Size(self.width, self.height))
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.data = data

    def on_paint(self, event):
        painter = wx.AutoBufferedPaintDC(self)
        self.scale_frame()
        painter.Clear()
        painter.DrawBitmap(
            getbitmap(self, self.data), 0, 0)

    def on_size(self, event):
        if not self.init:
            self.width, self.height = self.GetClientSize()
            self.scale_frame()
        self.init = False

    def scale_frame(self):
        r = min(self.height / float(self.data.shape[0]),
                self.width / float(self.data.shape[1]))
        dim = (int(self.data.shape[1] * r), int(self.data.shape[0] * r))
        self.data = cv2.resize(self.data, dim, interpolation=cv2.INTER_AREA)


class MainFrame(wx.Frame):
    def __init__(self, parent, id_, title, loglevel='INFO'):
        wx.Frame.__init__(self, parent, id_, title)
        self.Bind(EVT_ROS, self.on_ros_process)
        self.Bind(EVT_STRK, self.on_stroke_process)
        self.data = HandData()
        self.drawing_im = None
        self.canvas = None
        self.ros_thread = ROSThread(self, self.data, loglevel)
        self.ros_thread.start()
        self.stroke_data = StrokeData()
        self.stroke_recog_thread = StrokeRecognitionThread(self,
                                                           self.stroke_data,
                                                           loglevel)
        self.stroke_recog_thread.start()
        self.init_depth = 0
        self.draw = 0
        self.erase = 0
        self.size = 1
        self.prev_size = 1
        self.depths = []
        self.prev_gest = None
        self.frame = None
        self.stroke = []
        self.write_mode = True
        self.initialized = False
        self.max_size = 20
        self.min_size = 1
        self.max_depth = 800
        self.min_depth = 550
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.on_timer, self.timer)
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def on_close(self, event):
        if self.timer.IsRunning():
            self.timer.Stop()
        self.Destroy()

    def on_timer(self, event):
        if self.timer.IsRunning():
            self.timer.Stop()
        self.strokes.callback()

    def on_ros_process(self, event):
        try:
            inp = np.tile(self.data.hand[:, :, None] % 255, (1, 1, 3))
            if self.drawing_im is None:
                self.drawing_im = np.zeros_like(inp)
                self.temporary_im = np.zeros_like(inp)
            if self.init_depth == 0:
                self.init_depth = self.data.hand[self.data.skel[-1, -1, 0],
                                                 self.data.skel[-1, -1, 1]]
            try:
                ypnt = self.data.skel[-1, -1, 0]
                xpnt = self.data.skel[-1, -1, 1]
                tip = self.data.hand[ypnt - 20:ypnt + 20, xpnt - 20:xpnt + 20]
                dep = np.median(tip[tip != 0])
            except IndexError:
                print(self.data.skel)
            if np.isfinite(dep):
                self.depths.append(dep)
                del self.depths[:-20]
            else:
                if len(self.depths) == 0:
                    return
            dep = np.median(np.array(self.depths))
            init_dep = dep
            if dep < self.min_depth and dep != 0:  # erase everything if hand comes close to kinect
                self.drawing_im = np.zeros_like(inp)
                self.temporary_im = np.zeros_like(inp)
            else:
                dep = min(dep, self.max_depth)
                if dep != 0:
                    self.size = int((self.max_size - self.min_size) /
                                    (float(self.max_depth) - float(self.min_depth)) *
                                    (self.max_depth - dep) + self.min_size)
                self.size = min(self.size, self.max_size)
                self.size = max(self.size, self.min_size)
                if self.initialized:
                    if self.data.class_name == 'Punch':
                        cv2.circle(self.drawing_im, (self.data.skel[-1, -1, 1],
                                                     self.data.skel[-1, -1, 0]), self.size,
                                   [0, 0, 0], -1)
                    elif self.data.class_name == 'Index':
                        self.stroke.append([self.data.skel[-1, -1, 1],
                                            self.data.skel[-1, -1, 0]])
                        self.stroke = self.stroke[-4:]  # keep last 4 points
                        if self.write_mode:
                            color = [255, 255, 255]
                        else:
                            color = [255, 0, 0]
                        if self.prev_gest == 'Index':
                            if norm(
                                    np.diff(np.array(self.stroke[-2:]), axis=0)) < 10:
                                cv2.line(self.temporary_im,
                                         tuple(
                                             self.stroke[-2]), tuple(self.stroke[-1]),
                                         color, self.size)
                        else:
                            cv2.circle(self.temporary_im, (self.data.skel[-1, -1, 1],
                                                           self.data.skel[-1, -1, 0]), self.size,
                                       color, -1)
                    elif self.data.class_name == 'Tiger':
                        if self.prev_gest == 'Palm':
                            self.write_mode = not self.write_mode
                            if self.write_mode:
                                self.temporary_im = \
                                    np.zeros_like(self.drawing_im)

                else:
                    if self.data.class_name == 'Tiger':
                        self.initialized = True
                self.prev_gest = self.data.class_name
            if self.write_mode:
                self.drawing_im += self.temporary_im
                self.temporary_im = np.zeros_like(self.drawing_im)
            inp = (inp + self.drawing_im + self.temporary_im)
            inp = inp.astype(np.uint8)
            for link in self.data.skel:
                cv2.line(inp, tuple(link[0][::-1]), tuple(link[1][::-1]),
                         [255, 0, 0], 3)
            inp = cv2.flip(inp, -1)
            co.tag_im(inp, 'Gesture: ' + self.data.class_name +
                      '\nMode: ' + ('Free' if self.write_mode else 'Guided') +
                      '\nPen Size: ' + str(self.size), color=(255, 0, 0),
                      fontscale=0.7, thickness=2)
            co.tag_im(inp, 'Median depth: ' + str(init_dep) +
                      '\nFPS: ' + str(self.data.fps), loc='bot right',
                      fontscale=0.4, color=(255, 255, 255))
            if self.canvas is None:
                self.canvas = Canvas(self, inp, size=wx.Size(inp.shape[1],
                                                             inp.shape[0]))
                self.Fit()
            self.frame = inp
            self.canvas.data = inp
            self.canvas.Refresh(False)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type,
                                      exc_value,
                                      exc_traceback, limit=2, file=sys.stdout)

    def on_stroke_process(self, event):
        try:
            if self.stroke_data.found_gesture and not self.write_mode:
                points = []
                for stroke in self.stroke_data.strokes:
                    if len(stroke) == 0:
                        continue
                    points.append([])
                    points[-1].append([[v[0], v[1]]
                                       for v in stroke])
                    if self.stroke_data.gesture == 'Circle':
                        points = np.array(
                            [item for sublist in
                             points for item in sublist]).reshape(-1, 2)
                        center = np.int0(np.mean(points, axis=0))
                        radius = np.int0(norm(np.std(points, axis=0)))
                        cv2.circle(self.drawing_im, (center[0],
                                                     center[1]), radius,
                                   [255, 255, 255], self.size)
                    elif self.stroke_data.gesture == 'Line':
                        points = np.array(
                            [item for sublist in points for item in sublist])
                        cv2.line(self.drawing_im, tuple(points[0][0]), tuple(points[-1][-1]),
                                 [255, 255, 255], self.size)
                    elif self.stroke_data.gesture == 'Rectangle':
                        points = np.array(
                            [item for sublist in points for item in sublist])
                        rect = cv2.minAreaRect(points)
                        box = np.int0(cv2.boxPoints(rect))
                        cv2.drawContours(self.drawing_im, [box], 0,
                                         [255, 255, 255], self.size)
                    elif self.stroke_data.gesture == 'Triangle':
                        points = np.array(
                            [item for sublist in points
                             for item in sublist]).reshape(1, -1, 2)
                        triangle = np.int0(cv2.minEnclosingTriangle(
                            points)[1].squeeze())
                        cv2.drawContours(self.drawing_im,
                                         [triangle], 0,
                                         [255, 255, 255], self.size)
                    self.temporary_im = np.zeros_like(self.drawing_im)
            else:
                if self.write_mode:
                    self.drawing_im += self.temporary_im
                else:
                    self.temporary_im = np.zeros_like(self.drawing_im)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type,
                                      exc_value,
                                      exc_traceback, limit=2, file=sys.stdout)


def main():
    '''
    main function
    '''
    app = wx.App(0)
    frame = MainFrame(None, -1, 'Painter', loglevel=LOGLEVEL)
    frame.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
