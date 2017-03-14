import logging
LOG = logging.getLogger(__name__)
FORMAT = '%(funcName)20s(%(lineno)s)-%(levelname)s:%(message)s'
CH = logging.StreamHandler()
CH.setFormatter(logging.Formatter(FORMAT))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel('INFO')

import os, sys, traceback
import numpy as np
import wx
import cv2
import rospy
import roslaunch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, TimeReference
from geometry_msgs.msg import Point
import threading
import message_filters as mfilters
from math import exp
import class_objects as co
from kivy.multistroke import Recognizer
from kivy.multistroke import MultistrokeGesture
from kivy.clock import Clock as clock
from kivy.vector import Vector
import time
from math import pi,cos,sin
from numpy.linalg import norm
SHAPE = (500, 900)


def getbitmap(main_panel, img):
    '''
    numpy array to bitmap
    '''
    psize = main_panel.GetSize()
    '''
    if img.shape[0] != psize[0] or img.shape[1] != psize[1]:
        copy = cv2.resize(img, (psize[0], psize[1]))
    else:
        copy = img.copy()
    '''
    image = wx.Image(img.shape[1], img.shape[0])
    image.SetData(img.tostring())
    wx_bitmap = image.ConvertToBitmap()
    return wx_bitmap

EVT_ROS_TYPE = wx.NewEventType()
EVT_ROS = wx.PyEventBinder(EVT_ROS_TYPE, 1)
class CreateEvent(wx.PyCommandEvent):
    '''
    Event signaler
    '''
    def __init__(self, *args, **kwargs):
        wx.PyCommandEvent.__init__(self,*args,**kwargs)

EVT_STRK_TYPE = wx.NewEventType()
EVT_STRK = wx.PyEventBinder(EVT_STRK_TYPE, 1)
class StrokeEvent(wx.PyCommandEvent):
    '''
    Event to signal that data from ros is loaded
    '''
    def __init__(self, *args, **kwargs):
        wx.PyCommandEvent.__init__(self,*args,**kwargs)
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

class StrokeRecognitionThread(threading.Thread):
    def __init__(self,parent,data):
        threading.Thread.__init__(self)
        self._parent = parent
        self._data = data
    def run(self):
        '''
        Overrides Thread.run . Call Thread.start(), not this.
        '''
        print('Initializing Stroke Recognition subscriber..')
        subscriber = StrokeRecognition(self._parent, self._data)
        print('Waiting for strokes input..')

class StrokeRecognition(object):
    def __init__(self, parent, data):
        self.strokes = [[]]
        self.time = time.time()
        self.gdb = Recognizer()
        circle = [[Vector(int(10+10*cos(t)),
                                              int(10+10*sin(t)))
                                       for t in np.linspace(0,2*pi,8)]]
        self.gdb.add_gesture('Circle',circle, )
        #Horizontal or vertical lines give error
        self.gdb.add_gesture('Line', [[Vector(10,60),Vector(40,50)]],
                             strokes_sensitive=False)
        self.gdb.add_gesture('Triangle', [[Vector(10,10),
                                           Vector(15,15),
                                          Vector(20,20),
                                           Vector(25,15),
                                          Vector(30,10),
                                           Vector(20,10),
                                          Vector(10,10)]],
                            strokes_sensitive=False)
        self.gdb.add_gesture('Rectangle', [[Vector(10,10),
                                            Vector(10,15),
                                           Vector(10,20),
                                            Vector(15,20),
                                           Vector(20,20),
                                            Vector(20,15),
                                           Vector(20,10),
                                            Vector(15,10),
                                           Vector(10,10)]],
                            strokes_sensitive=False)
        self.found_gesture = False
        self.gesture = None
        self._data = data
        self._parent = parent
        self.ran = False
        self.gdb.bind(on_search_start=self.search_start)
        self.gdb.bind(on_search_complete=self.search_stop)
        self.dist_thres = 1000
        self.time_thres = 2
        self.skel_sub = mfilters.Subscriber(
            "skeleton", Image)
        self.clas_sub = mfilters.Subscriber(
            "class", TimeReference)
        self.image_ts = mfilters.TimeSynchronizer(
            [self.skel_sub, self.clas_sub], 30)
        self.image_ts.registerCallback(
            self.callback)
        self.bridge = CvBridge()
    def callback(self, skel, class_name):
        try:
            skel = self.bridge.imgmsg_to_cv2(skel,
                                             desired_encoding=
                                             'passthrough')
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
                print self.strokes[:-1]
                self.progress = self.gdb.recognize(self.strokes[:-1])
                #removing last empty sublist          ^
                self.progress.bind(on_progress=self.on_progress)
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
             print e
    def on_progress(self, *args):
        print(self.progress.progress) # = 0


    def search_start(self, gdb, pt):
        print("A search is starting with %d tasks" % (pt.tasks))
    def search_stop(self, gdb, pt):
        best = pt.best
        print ("Search ended (",pt.status,"). Best is ",best['name'],
        " (score ",
        best['score'],", distance", best['dist'],")")
        dist = best['dist']
        self.gesture = best['name']
        if dist < self.dist_thres:
            self.found_gesture = True

    def add_stroke(self):
        if len(self.strokes[-1])>0: #add stroke only if last stroke is not empty
            self.strokes.append([])
    def add_stroke_vector(self, xpoint, ypoint):
        self.strokes[-1].append(Vector(xpoint,ypoint))

class ROSThread(threading.Thread):
    def __init__(self, parent, data):
        threading.Thread.__init__(self)
        self._parent = parent
        self._data = data
    def run(self):
        '''
        Overrides Thread.run . Call Thread.start(), not this.
        '''
        print('Initializing ROS subscriber..')
        subscriber = ROSSubscriber(self._parent,
                                   self._data)
        print('Initializing node..')
        rospy.init_node('ROSThread', anonymous=True, disable_signals=True)
        print('Waiting for input..')
        rospy.spin()


class ROSSubscriber(object):
    def __init__(self, parent, data):
        self._parent = parent
        self._data = data
        self.hand_sub = mfilters.Subscriber(
            "hand", Image)
        self.skel_sub = mfilters.Subscriber(
            "skeleton", Image)
        self.clas_sub = mfilters.Subscriber(
            "class", TimeReference)
        self.image_ts = mfilters.TimeSynchronizer(
            [self.hand_sub, self.skel_sub,
             self.clas_sub], 30)
        self.image_ts.registerCallback(
            self.callback)
        self.bridge = CvBridge()
    def callback(self, hand, skel, class_name):
        hand = self.bridge.imgmsg_to_cv2(hand,
                                              desired_encoding=
                                              'passthrough')
        skel = self.bridge.imgmsg_to_cv2(skel,
                                              desired_encoding=
                                              'passthrough')
        self._data.add(hand, skel, class_name.source,
                       class_name.time_ref.secs)
        evt = CreateEvent(EVT_ROS_TYPE, -1)
        wx.PostEvent(self._parent, evt)

class Canvas(wx.Panel):
    def __init__(self, parent, data):
        wx.Panel.__init__(self, parent)
        [self.height, self.width] = data.shape[:2]
        self.SetMinSize(wx.Size(self.width, self.height))
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.data = data

    def on_paint(self, event):
        painter = wx.AutoBufferedPaintDC(self)
        painter.Clear()
        painter.DrawBitmap(
            getbitmap(self, self.data), 0, 0)

def tag_im(img, text):
    '''
    Tag top of img with description in red
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (0, 20), font, 0.5, (0, 0, 255), 2)

class MainFrame(wx.Frame):
    def __init__(self,parent, id_, title):
        wx.Frame.__init__(self,parent, id_, title)
        self.canvas_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.Bind(EVT_ROS, self.on_ros_process)
        self.Bind(EVT_STRK, self.on_stroke_process)
        self.data = HandData()
        self.drawing_im = None
        self.canvas = None
        self.ros_thread = ROSThread(self,self.data)
        self.ros_thread.start()
        self.stroke_data = StrokeData()
        self.stroke_recog_thread = StrokeRecognitionThread(self,
                                                           self.stroke_data)
        self.stroke_recog_thread.start()
        self.init_depth = 0
        self.draw = 0
        self.erase = 0
        self.size = 1
        self.prev_size = 1
        self.depths = []
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
        inp = np.tile(self.data.hand[:, :, None] % 255, (1, 1, 3))
        if self.drawing_im is None:
            self.drawing_im = np.zeros_like(inp)
            self.temporary_im = np.zeros_like(inp)
        if self.init_depth == 0:
            self.init_depth = self.data.hand[self.data.skel[-1, -1, 0],
                                             self.data.skel[-1, -1, 1]]
        try:
            dep = self.data.hand[self.data.skel[-1,-1,0],
                             self.data.skel[-1,-1,1]]
        except IndexError:
            print self.data.skel
        if dep!=0:
            if len(self.depths)<20:
                self.depths.append(dep)
            else:
                self.depths= self.depths[1:]+[dep]
            dep = np.median(np.array(self.depths))
        '''
        if dep != 0:
            self.size = int(2*(11-10*dep/float(self.init_depth)))
        self.size = min(self.size, 10)
        self.size = max(self.size , 1)
        '''
        if self.prev_size != self.size:
            LOG.info('Current Size:'+ str(self.size))
            self.prev_size = self.size
        if self.data.class_name == 'Punch':
            self.size = 5
            cv2.circle(self.drawing_im,(self.data.skel[-1, -1, 1],
                                            self.data.skel[-1, -1, 0]), self.size,
                           [0,0,0], -1)
        elif self.data.class_name == 'Index':
            self.size = 2
            cv2.circle(self.temporary_im,(self.data.skel[-1, -1, 1],
                                        self.data.skel[-1, -1, 0]), self.size,
                       [255,255,255], -1)

        inp = inp + self.drawing_im + self.temporary_im
        inp = inp.astype(np.uint8)
        for link in self.data.skel:
            cv2.line(inp, tuple(link[0][::-1]), tuple(link[1][::-1]),
                     [255, 0, 0], 3)
        inp = cv2.flip(inp, -1)
        co.tag_im(inp, 'Action: ' + self.data.class_name)
        co.tag_im(inp, 'FPS: ' + str(self.data.fps),loc='bot right',
                  fontscale=0.4, color=(255,255,255))
        if self.canvas is None:
            self.canvas = Canvas(self, inp)
            self.canvas_sizer.Add(self.canvas)
            self.SetSizerAndFit(self.canvas_sizer)
        else:
            self.canvas.data = inp
        self.canvas.Refresh(False)
    def on_stroke_process(self, event):
        if self.stroke_data.found_gesture:
            points = []
            for stroke in self.stroke_data.strokes:
                if len(stroke) == 0:
                    continue
                points.append([])
                points[-1].append([[v[0],v[1]]
                               for v in stroke])
            try:
                if self.stroke_data.gesture == 'Circle':
                    points = np.array(
                        [item for sublist in
                         points for item in sublist]).reshape(-1,2)
                    print points.shape
                    center = np.int0(np.mean(points,axis=0))
                    radius = np.int0(norm(np.std(points,axis=0)))
                    cv2.circle(self.drawing_im,(center[0],
                                            center[1]), radius,
                           [255,255,255], self.size)
                elif self.stroke_data.gesture == 'Line':
                    points = np.array(
                        [item for sublist in points for item in sublist])
                    cv2.line(self.drawing_im, tuple(points[0][0])
                             , tuple(points[-1][-1]),
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
                         for item in sublist]).reshape(1,-1,2)
                    triangle = np.int0(cv2.minEnclosingTriangle(
                        points)[1].squeeze())
                    cv2.drawContours(self.drawing_im,
                                     [triangle], 0,
                                     [255, 255, 255], self.size)
                self.temporary_im = np.zeros_like(self.drawing_im)
            except Exception as e:
                 exc_type, exc_value, exc_traceback = sys.exc_info()
                 traceback.print_exception(exc_type,
                                    exc_value,
                                    exc_traceback, limit=2, file=sys.stdout)
        else:
            self.drawing_im += self.temporary_im




def main():
    '''
    main function
    '''
    app = wx.App(0)
    frame = MainFrame(None, -1, 'Painter')
    frame.Show(True)
    app.MainLoop()

if __name__ == '__main__':
    main()
