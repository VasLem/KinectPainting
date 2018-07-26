'''
Module to create canvas
'''
#pylint: disable=R0903
import logging
import sys
import threading
from subprocess import Popen
import shlex
import signal
import traceback
import abc
import time
from math import pi, cos, sin
import numpy as np
from numpy.linalg import norm
import cv2
import rospy
from kivy.multistroke import Recognizer
from kivy.clock import Clock as clock
from kivy.vector import Vector
import wx
from ros_pipeline.base import ROSPipelineElement
import class_objects as co
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


def getbitmap(img):
    '''
    Get wx image bitmap using an input numpy array
    '''
    image = wx.Image(img.shape[1], img.shape[0])
    image.SetData(img.tostring())
    wx_bitmap = image.ConvertToBitmap()
    return wx_bitmap




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
            getbitmap(self.data), 0, 0)

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
        self.drawing_im = None
        self.canvas = None
        self.ros_thread = ROSThread(self, loglevel)
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
