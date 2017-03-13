import os, sys
import numpy as np
import wx
import cv2
import logging
import rospy
import roslaunch
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, TimeReference
import threading
import message_filters as mfilters
from math import exp
import class_objects as co
LOG = logging.getLogger(__name__)
FORMAT = '%(funcName)20s(%(lineno)s)-%(levelname)s:%(message)s'
CH = logging.StreamHandler()
CH.setFormatter(logging.Formatter(FORMAT))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel('INFO')
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

EVT_READY_TYPE = wx.NewEventType()
EVT_READY = wx.PyEventBinder(EVT_READY_TYPE, 1)
class ReadyEvent(wx.PyCommandEvent):
    '''
    Event to signal that data from ros is loaded
    '''
    def __init__(self, *args, **kwargs):
        wx.PyCommandEvent.__init__(self,*args,**kwargs)

class Data(object):
    def __init__(self):
        self.hand = None
        self.skel = None
        self.class_name = None
        self.fps = None
    def add(self, hand, skel, class_name,fps):
        self.hand = hand
        self.skel = skel
        self.class_name = class_name
        self.fps = fps

class ROSThread(threading.Thread):
    def __init__(self, parent, data):
        threading.Thread.__init__(self)
        self._parent = parent
        self._data = data
    def run(self):
        '''
        Overrides Thread.run . Call Thread.start(), not this.
        '''
        print('Initializing subscriber..')
        subscriber = ROSSubscriber(self._parent,
                                   self._data)
        print('Initializing node..')
        rospy.init_node('canvas', anonymous=True, disable_signals=True)
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
        evt = ReadyEvent(EVT_READY_TYPE, -1)
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
        self.Bind(EVT_READY, self.on_process)
        self.data = Data()
        self.drawing_im = None
        self.canvas = None
        self.ros_thread = ROSThread(self,self.data)
        self.ros_thread.start()
        self.init_depth = 0
        self.draw = 0
        self.erase = 0
        self.size = 1
        self.prev_size = 1
        self.depths = []
    def on_process(self, event):
        inp = np.tile(self.data.hand[:, :, None] % 255, (1, 1, 3))
        if self.drawing_im is None:
            self.drawing_im = np.zeros_like(inp)
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
            cv2.circle(self.drawing_im,(self.data.skel[-1, -1, 1],
                                        self.data.skel[-1, -1, 0]), self.size,
                       [255,255,255], -1)
        inp = inp + self.drawing_im
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
