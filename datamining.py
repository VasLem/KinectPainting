'''
Dataset farmer application
'''
# pylint: disable=unused-argument,too-many-ancestors,too-many-instance-attributes
# pylint: disable=too-many-arguments
import os
import sys
import logging
import errno
import time
import tempfile
import traceback
import signal
import subprocess
import shlex
from Queue import Queue
from sensor_msgs.msg import Image
import psutil
import wx
import wx.lib.mixins.listctrl as listmix
from skimage import io
import numpy as np
import yaml
import rosbag
import cv2
import class_objects as co
import moving_object_detection_alg as moda
import hand_segmentation_alg as hsa
import extract_and_process_rosbag as epr
import full_actions_registration as far
from __init__ import terminate_process
from __init__ import check_if_running
from __init__ import run_on_external_terminal
from canvas import KinectSubscriber
io.use_plugin('freeimage')
ID_LOAD_CSV = wx.NewId()
ID_BAG_PROCESS = wx.NewId()
ID_PROCESS_ALL = wx.NewId()
ID_PLAY = wx.NewId()
ID_STOP = wx.NewId()
ID_SAVE_CSV = wx.NewId()
ID_ADD = wx.NewId()
ID_MIN = wx.NewId()
ID_MAX = wx.NewId()
ID_REMOVE = wx.NewId()
ID_VIDEO_SAVE = wx.NewId()
ID_MASK_RECOMPUTE = wx.NewId()
ID_ACT_SAVE = wx.NewId()
ID_BAG_RECORD = wx.NewId()
ID_SAMPLES_SAVE = wx.NewId()
ID_START_RECORD = wx.NewId()
ID_STOP_RECORD = wx.NewId()
CURR_DIR = os.getcwd()
ACTIONS_SAVE_PATH = os.path.join(CURR_DIR, 'actions')
ROSBAG_WHOLE_RES_SAVE_PATH = os.path.join(CURR_DIR, 'whole_result')
START_COUNT = 0  # 300
STOP_COUNT = 0  # 600
EVT_ROS_TYPE = wx.NewEventType()
EVT_ROS = wx.PyEventBinder(EVT_ROS_TYPE, 1)

#pylint disable=R0903
class CreateEvent(wx.PyCommandEvent):
    '''
    Event signaler
    '''
    def __init__(self, *args, **kwargs):
        wx.PyCommandEvent.__init__(self, *args, **kwargs)


try:
    with open('descriptions.yaml', 'r') as inp:
        DESCRIPTIONS = yaml.load(inp)
except BaseException:
    DESCRIPTIONS = None


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def getbitmap(main_panel, img):
    '''
    numpy array to bitmap
    '''
    psize = main_panel.GetSize()
    if img is not None:
        if img.shape[0] != psize[0] or img.shape[1] != psize[1]:
            copy = cv2.resize(img, (psize[0], psize[1]))
        else:
            copy = img.copy()
        if len(copy.shape) == 2:
            copy = np.tile(copy[:, :, None], (1, 1, 3))
        if not isinstance(copy[0, 0, 0], np.uint8) or np.max(copy) == 1:
            if np.max(copy) > 5000:
                copy = copy % 256
            else:
                if np.any(copy):
                    copy = (copy / float(np.max(copy))) * 255
            copy = copy.astype(np.uint8)
        image = wx.Image(copy.shape[1], copy.shape[0])
        image.SetData(copy.tostring())
        wx_bitmap = image.ConvertToBitmap()
        return wx_bitmap
    else:
        return None


class SpinStepCtrl(wx.SpinCtrlDouble):

    def __init__(self, parent, *args, **kwargs):
        wx.SpinCtrlDouble.__init__(self, parent, id=wx.ID_ANY, value='0.00', pos=wx.DefaultPosition,
                                   size=wx.DefaultSize,
                                   inc=kwargs.get('step'),
                                   min=kwargs.get('min_val', 0.00),
                                   max=kwargs.get('max_val', 100.00))
        self.SetValue(kwargs.get('init_val', 0.00))


class NamedTextCtrl(wx.BoxSizer):

    def __init__(self, parent, name, orientation=wx.VERTICAL,
                 flags=wx.ALL):
        wx.BoxSizer.__init__(self, orientation)
        centered_label = wx.StaticText(parent, -1, name)
        if orientation == wx.VERTICAL:
            self.Add(centered_label, 0, wx.ALIGN_CENTER_HORIZONTAL)
        else:
            self.Add(centered_label, 0, wx.ALIGN_CENTER_VERTICAL)
        self.ctrl = wx.TextCtrl(parent, -1, name=name)
        self.Add(self.ctrl, 1, wx.EXPAND | wx.ALL)

    def GetValue(self):
        return self.ctrl.GetValue()

    def SetValue(self, value):
        return self.ctrl.SetValue(value)

    def Clear(self):
        self.ctrl.Clear()


class LabeledSpinStepCtrlSizer(wx.BoxSizer):

    def __init__(self, parent, name, orientation=wx.VERTICAL, step=0, init_val=0,
                 set_range=(0.0, 100.0)):
        self.name = name
        wx.BoxSizer.__init__(self, orientation)
        centered_label = wx.StaticText(parent, -1, name)
        self.Add(centered_label, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.ctrl = SpinStepCtrl(parent, step=step,
                                 init_val=init_val,
                                 min_val=set_range[0],
                                 max_val=set_range[1])
        self.Add(self.ctrl, flag=wx.EXPAND)

    def GetValue(self):
        return self.ctrl.GetValue()


class MOG2params(wx.Panel):
    '''
    Panel to experiment with mog2 parameters
    '''

    def __init__(self, parent, main_panel):
        wx.Panel.__init__(self, main_panel)
        self.bg_ratio = LabeledSpinStepCtrlSizer(self, init_val=co.CONST['bg_ratio'],
                                                 set_range=(0, 1),
                                                 step=0.05,
                                                 name='bg_ratio')
        self.var_thres = LabeledSpinStepCtrlSizer(
            self, init_val=co.CONST['var_thres'],
            set_range=(0, 100),
            step=1,
            name='var_thres')
        self.gmm_num = LabeledSpinStepCtrlSizer(self, init_val=co.CONST['gmm_num'],
                                                set_range=(1, 30),
                                                step=1,
                                                name='gmm_num')
        self.history = LabeledSpinStepCtrlSizer(self, init_val=co.CONST['history'],
                                                set_range=(0, 10000),
                                                step=100,
                                                name='history')
        box = wx.BoxSizer(wx.VERTICAL)
        box.AddMany((self.bg_ratio, self.var_thres,
                     self.gmm_num, self.history))
        self.SetSizer(box)


class StreamChoice(wx.Frame):
    def __init__(self, parent, id_,
                 data):
        wx.Frame.__init__(self, parent, id_, 'Select Stream')
        self.Center()
        sizer = wx.BoxSizer(wx.VERTICAL)
        name = 'Convert to video :'
        self._data = data
        self._parent = parent
        centered_label = wx.StaticText(self, -1, name)
        self.choice = wx.Choice(self, -1, choices=data.keys())
        self.choice.Bind(wx.EVT_CHOICE, self.on_choice)
        sizer.AddMany(([centered_label, wx.ALIGN_CENTER_HORIZONTAL],
                       self.choice))
        self.SetSizerAndFit(sizer)

    def on_choice(self, event):
        chosen = self.choice.GetStringSelection()
        path = os.path.join(co.CONST['results_fold'], 'Videos')
        makedir(path)
        dlg = wx.FileDialog(self, "Save stream as..",
                            path,
                            os.path.basename(self._data[chosen].name.split(':')[0]) +
                            '_' +
                            os.path.basename(chosen),
                            "*.avi", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        result = dlg.ShowModal()
        dlg.Destroy()
        self.Hide()
        if result == wx.ID_OK:
            try:
                path = dlg.GetPath()
                dlg = wx.ProgressDialog("Converting",
                                        "Progress",
                                        maximum=len(self._data[chosen].frames),
                                        parent=self,
                                        style=0
                                        | wx.PD_APP_MODAL
                                        | wx.PD_ESTIMATED_TIME
                                        | wx.PD_REMAINING_TIME
                                        | wx.PD_CAN_ABORT)
                video = None
                try:
                    framesize = None
                    for frame in self._data[chosen].frames:
                        if frame is not None:
                            framesize = frame.shape[:2][::-1]

                            break
                    if framesize is None:
                        dlg.Destroy()
                        dlg.Close()
                        self.Close()
                        return
                    video = cv2.VideoWriter(
                        path, cv2.VideoWriter_fourcc(*'MJPG'), 30, framesize)
                    for count in range(len(self._data[chosen].frames)):
                        frame = self._data[chosen].frames[count]
                        sync = self._data[chosen].sync[count]
                        if frame is not None:
                            if len(frame.shape) < 2 or len(frame.shape) > 3:
                                raise Exception('Invalid frame of shape ' +
                                                frame.shape + ' was given')
                            elif len(frame.shape) == 2:
                                inp = np.tile(frame[..., None], (1, 1, 3))
                            else:
                                inp = frame
                            video.write(inp.astype(np.uint8))
                        else:
                            video.write(np.zeros(framesize, np.uint8))
                        wx.Yield()
                        keepGoing, _ = dlg.Update(count)
                        if not keepGoing:
                            cv2.destroyAllWindows()
                            video.release()
                            dlg.Destroy()
                            dlg.Close()
                            self.CLose()
                            return
                finally:
                    if video is not None:
                        cv2.destroyAllWindows()
                        video.release()

            finally:
                dlg.Destroy()
                dlg.Close()
                self.Close()
        elif result == wx.ID_CANCEL:
            return


class FrameToVideoOperations(wx.Panel):

    def __init__(self, parent, main_panel, data):
        self._parent = parent
        self._data = data
        wx.Panel.__init__(self, main_panel)
        self.save_vid = wx.Button(
            self, ID_VIDEO_SAVE, "Save as Video")
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.save_vid, wx.ALIGN_CENTER_HORIZONTAL)
        self.SetSizerAndFit(sizer)
        self.Bind(wx.EVT_BUTTON, self.on_selection, id=ID_VIDEO_SAVE)

    def on_selection(self, event):
        if not self._data.keys():
            dlg = wx.MessageDialog(self._parent, 'Process a rosbag file first',
                                   'Error', wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
            return
        choice_frame = StreamChoice(None, -1, self._data)
        choice_frame.Show()


class EditableListCtrl(wx.ListCtrl, listmix.TextEditMixin):

    def __init__(self, parent, ID=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.TextEditMixin.__init__(self)


class TButton(wx.Button):

    def __init__(self, parent, id=-1, label=wx.EmptyString, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0, validator=wx.DefaultValidator,
                 name=wx.ButtonNameStr, descr=''):
        wx.Button.__init__(self, parent, id, label,
                           pos, size, style,
                           validator, name)
        if descr == '':
            if DESCRIPTIONS is not None:
                if label in DESCRIPTIONS.keys():
                    self.SetToolTip(wx.ToolTip(DESCRIPTIONS[label]))
        else:
            self.SetToolTip(wx.ToolTip(descr))


class ShapeTButton(wx.BitmapButton):
    def __init__(self, parent, img_path,
                 id=-1, label=wx.EmptyString, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0, validator=wx.DefaultValidator,
                 name=wx.ButtonNameStr, descr=''):
        image = wx.Image(img_path, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        super(ShapeTButton, self).__init__(parent, id, bitmap=image,
                                           pos=pos, size=size, style=style,
                                           validator=validator, name=name)
        if descr == '':
            if DESCRIPTIONS is not None:
                if label in DESCRIPTIONS.keys():
                    self.SetToolTip(wx.ToolTip(DESCRIPTIONS[label]))
        else:
            self.SetToolTip(wx.ToolTip(descr))


class TopicsNotebook(wx.Notebook):

    def __init__(self, parent, data, id=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize,
                 style=wx.NB_TOP, name=wx.NotebookNameStr, fps=20, start_frame_handler=None,
                 end_frame_handler=None, forced_frame_handler=None):
        '''
        data is a dictionary of RosbagStruct. data will change for lower memory
        consumption.
        '''
        wx.Notebook.__init__(self, parent, id, pos, size, style=wx.NB_TOP,
                             name=name)
        self.start_frame_handler = start_frame_handler
        self.end_frame_handler = end_frame_handler
        self.forced_frame_handler = forced_frame_handler
        self.fps = fps
        self.pages = []
        max_len = 0
        for topic in data.keys():
            max_len = max(np.max(data[topic].sync) + 1, max_len)
        for topic_count, topic in enumerate(data):
            if data[topic].frames:
                inp = [None] * max_len
                for count, sync_count in enumerate(data[topic].sync):
                    inp[sync_count] = data[topic].frames[count]
                self.pages.append(VideoPanel(self, inp, fps,
                                             start_frame_handler,
                                             end_frame_handler,
                                             forced_frame_handler))
            else:
                txtPanel = wx.Panel(self)
                wx.StaticText(txtPanel, id=-1, label="This topic  was not saved in memory",
                              style=wx.ALIGN_CENTER, name="")
                self.pages.append(txtPanel)
            label = topic
            self.InsertPage(topic_count, self.pages[-1], label)

    def change_video(self, data):
        self.pages = []
        max_len = 0
        for topic in data.keys():
            max_len = max(len(data[topic].frames), max_len)
        for topic_count, topic in enumerate(data.keys()):
            if data[topic].frames:
                inp = [None] * max_len
                for count, sync_count in enumerate(data[topic].sync):
                    inp[sync_count] = data[topic].frames[count]
                try:
                    self.pages[topic_count] = VideoPanel(self, inp, self.fps,
                                                         self.start_frame_handler,
                                                         self.end_frame_handler,
                                                         self.forced_frame_handler)
                except IndexError:
                    self.pages.append(VideoPanel(self, inp, self.fps,
                                                 self.start_frame_handler,
                                                 self.end_frame_handler,
                                                 self.forced_frame_handler))
            else:
                txtPanel = wx.Panel(self)
                wx.StaticText(txtPanel, id=-1, label="This topic  was not saved in memory",
                              style=wx.ALIGN_CENTER, name="")
                try:
                    self.pages[topic_count] = txtPanel
                except IndexError:
                    self.pages.append(txtPanel)
            label = topic
            self.RemovePage(topic_count)
            self.InsertPage(topic_count, self.pages[topic_count], label)


def checkclass(obj, clas):
    if isinstance(obj, clas) or issubclass(obj.__class__, clas):
        return 1
    else:
        return 0


def wx_generic_binder(widget, function):
    '''
    TextCtrl(wx.EVT_TEXT) and Slider(wx.EVT_SLIDER) are supported for now.
    '''
    if widget is not None:
        if checkclass(widget, wx.TextCtrl):
            widget.Bind(wx.EVT_TEXT, function)
        elif checkclass(widget, wx.Slider):
            widget.Bind(wx.EVT_SLIDER, function)
        else:
            raise NotImplementedError


class VideoPanel(wx.Panel):
    '''
    A video panel implementation
    '''

    def __init__(self, parent, data, fps=20, start_frame_handler=None,
                 end_frame_handler=None, forced_frame_handler=None,
                 asp_rat=None):
        '''
        data is a list of frames. If a frame is missing,
        the entry is None
        '''
        wx.Panel.__init__(self, parent, wx.NewId())
        self.width = None
        for frame in data:
            if frame is not None:
                [self.height, self.width] = frame.shape[:2]
                break
        if self.width is None:
            raise Exception('Input frames data is full of None')
        if asp_rat is not None:
            self.height = int(self.height * asp_rat)
            self.width = int(self.width * asp_rat)
        self.SetMinSize(wx.Size(self.width, self.height))
        self.playing = 0
        self.count = 0
        self.start = 0
        self.start_frame_handler = start_frame_handler
        self.end_frame_handler = end_frame_handler
        self.forced_frame_handler = forced_frame_handler
        wx_generic_binder(self.start_frame_handler,
                          lambda event: self.handle_start(event))
        wx_generic_binder(self.end_frame_handler,
                          lambda event: self.handle_end(event))
        wx_generic_binder(self.forced_frame_handler,
                          lambda event: self.handle_forced(event))
        self.Bind(wx.EVT_PAINT,
                  lambda event: self.on_playing(event))
        self.end = len(data)
        self.data = data
        self.forced = 0
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.timer = wx.Timer(self)
        self.fps = fps
        self.replay = True
        self.timer.Start(self.fps)
        wx.CallLater(200, self.SetFocus)
        self.img = None

    def play(self, start=0, end=0):
        '''
        start video from start to end
        '''
        if end == 0:
            end = len(self.data)
        if not self.playing:
            if self.start_frame_handler is not None:
                self.start = self.start_frame_handler.GetValue()
            else:
                self.start = start
            if self.end_frame_handler is not None:
                self.end = self.end_frame_handler.GetValue()
            else:
                self.end = end
            self.playing = 1

    def handle_start(self, event):
        self.start = self.start_frame_handler.GetValue()
        event.Skip()

    def handle_end(self, event):
        self.end = self.end_frame_handler.GetValue()
        event.Skip()

    def handle_forced(self, event):
        self.count = self.forced_frame_handler.GetValue()
        self.forced = 1
        event.Skip()

    def change_start(self, ind):
        '''
        change starting frame, causes video to replay
        '''
        self.start = ind
        self.count = self.start

    def change_end(self, ind):
        '''
        change ending frame
        '''
        self.end = ind

    def force_frame_change(self, ind):
        '''
        Force frame to change
        '''
        self.count = ind

    def update_drawing(self):
        '''
        enable video
        '''
        self.Refresh(False)

    def on_timer(self, event):
        '''
        set timer for video playing
        '''
        self.update_drawing()

    def on_playing(self, event):
        '''
        play video of frames described by sliders
        '''
        painter = wx.AutoBufferedPaintDC(self)
        painter.Clear()
        if self.end_frame_handler is not None:
            self.end = self.end_frame_handler.GetValue()
        if self.forced and not self.playing:
            if self.count <= self.end:
                if self.data[self.count] is not None:
                    self.img = self.data[self.count]
            self.forced = False

        if self.playing:
            if self.count < self.end:
                if self.data[self.count] is not None:
                    self.img = self.data[self.count]
                self.count += 1
            else:
                if self.replay:
                    self.count = self.start
                else:
                    self.playing = 0
        if self.img is not None:
            painter.DrawBitmap(
                getbitmap(self, self.img), 0, 0)

    def pause(self):
        '''
        pause video
        '''
        if self.playing:
            self.playing = 0

    def stop(self):
        '''
        stop video
        '''
        if self.playing:
            self.playing = 0
            self.count = self.start


class KinectPreview(wx.Panel):
    '''
    Kinect Channel Previewer
    '''
    def __init__(self, parent, *args, **kwargs):
        self.init = True
        wx.Panel.__init__(self, parent, *args, **kwargs)
        [self.height, self.width] = [250, 250]
        self.SetMinSize(wx.Size(self.width, self.height))
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.data = np.zeros((self.height, self.width))
        self.timer = wx.Timer(self)
        self.replay = True
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)



    def update_drawing(self):
        '''
        enable video
        '''
        self.Refresh(False)

    def on_timer(self, event):
        '''
        set timer for video playing
        '''
        self.update_drawing()

    def set_size(self, wx_size):
        self.SetInitialSize(wx_size)

    def set_data(self, data):
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


class KinectRecorder(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        self.loglevel = 'INFO'
        self.init = True
        wx.Window.__init__(self, parent, *args, **kwargs)
        self.Bind(EVT_ROS, self.on_ros_process)
        self.Bind(wx.EVT_SIZE, self.on_size)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.Bind(wx.EVT_BUTTON, self.on_start_record,
                  id=ID_START_RECORD)
        self.Bind(wx.EVT_BUTTON, self.on_stop_record,
                  id=ID_STOP_RECORD)

        self.inter_pnl = wx.Panel(self, wx.NewId())
        self.kinect_preview_pnl = KinectPreview(self)
        self.record_button = ShapeTButton(self,
                                          os.path.join(
                                              co.CONST['AppData'],
                                              'Buttons',
                                              'record.png'),
                                          id=ID_START_RECORD)
        self.stop_button = ShapeTButton(self,
                                        os.path.join(
                                            co.CONST['AppData'],
                                            'Buttons',
                                            'stop.png'),
                                        id=ID_STOP_RECORD)
        inter_box = wx.BoxSizer(wx.VERTICAL)
        inter_box.AddMany([(self.record_button, 0),
                           (self.stop_button, 0)])
        self.inter_pnl.SetSizer(inter_box)
        main_box = wx.BoxSizer(wx.VERTICAL)
        main_box.AddMany([(self.inter_pnl, 0),
                          (self.kinect_preview_pnl, 0)])
        self.SetSizerAndFit(main_box)

        self.ros_thread = None
        self.started_roscore = 0
        self.started_kinect2_bridge = 0
        self.depths = []
        self.rosbag_record = None
        self.data_queue = Queue()
        self.data_queue.put(HandData())
        self.start_kinect()

    def on_size(self, event):
        if not self.init:
            self.width, self.height = self.GetClientSize()
            self.scale_frame()
        self.init = False

    def scale_frame(self):
        data = self.data_queue.get()
        r = min(self.height / float(data.hand.shape[0]),
                self.width / float(data.hand.shape[1]))
        dim = (int(data.hand.shape[1] * r), int(data.hand.shape[0] * r))
        data.hand = cv2.resize(data.hand, dim, interpolation=cv2.INTER_AREA)
        data.skel = np.dstack((dim[0] * data.skel[:, :, 0],
                               dim[1] * data.skel[:, :, 1]))

    def on_ros_process(self, event):
        self.scale_frame()
        data = self.data_queue.get()
        try:
            inp = np.tile(data.hand[:, :, None] % 255, (1, 1, 3))
            try:
                ypnt = data.skel[-1, -1, 0]
                xpnt = data.skel[-1, -1, 1]
                tip = data.hand[ypnt - 20:ypnt + 20, xpnt - 20:xpnt + 20]
                dep = np.median(tip[tip != 0])
            except IndexError:
                print data.skel
            if np.isfinite(dep):
                self.depths.append(dep)
                del self.depths[:-20]
            else:
                if len(self.depths) == 0:
                    return
            dep = np.median(np.array(self.depths))
            init_dep = dep
            inp = inp.astype(np.uint8)
            for link in data.skel:
                cv2.line(inp, tuple(link[0][::-1]), tuple(link[1][::-1]),
                         [255, 0, 0], 3)
            inp = cv2.flip(inp, -1)
            co.tag_im(inp, 'Median Hand Tip depth: ' + str(init_dep) +
                      '\nFPS: ' + str(data.fps), loc='bot right',
                      fontscale=0.4, color=(255, 255, 255))
            if self.kinect_preview_pnl.size is None:
                self.kinect_preview_pnl.set_size(wx.Size(inp.shape[1],
                                                         inp.shape[0]))
                self.Fit()
            self.frame = inp
            self.kinect_preview_pnl.set_data(data)
            self.Refresh(False)
            self.data_queue.put(data)
        except Exception as e:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            traceback.print_exception(exc_type,
                                      exc_value,
                                      exc_traceback, limit=2, file=sys.stdout)

    def on_start_record(self, evt):
        temp_file = tempfile.NamedTemporaryFile('rw', dir=co.CONST[
            'rosbag_temporary_location'])
        temp_file.close()
        self.temp_path = os.path.join(co.CONST['rosbag_temporary_location'],
                                      temp_file.name)
        if self.rosbag_record is None:
            self.rosbag_record = RosbagRecord(co.CONST['channel'],
                                              self.temp_path)
        self.rosbag_record.__enter__()

    def on_stop_record(self, evt):
        self.rosbag_record.__exit__(None, None, None)
        self.save_as()

    def on_close(self):
        self.ros_thread.stop()
        while not self.ros_thread.stopped():
            wx.BusyInfo("Closing, please wait...")
            time.sleep(1)
        self.stop_kinect()
        self.MakeModal(False)
        self.destroy()

    def start_kinect(self):
        if not check_if_running('roscore'):
            self.started_roscore = run_on_external_terminal('roscore')
        if not check_if_running('kinect2_bridge'):
            self.started_kinect2_bridge = run_on_external_terminal(
                "roslaunch kinect2_bridge kinect2_bridge.launch")
        self.ros_thread = KinectSubscriber('KinectSubscriber', self,
                                           self.data_queue, self.loglevel,
                                           co.CONST['channel'], Image)
        self.ros_thread.start()

    def stop_kinect(self):
        if self.started_roscore:
            terminate_process(self.started_roscore, use_pid=True)
            self.started_roscore = 0
        if self.started_kinect2_bridge:
            terminate_process(self.started_kinect2_bridge, use_pid=True)
            self.started_kinect2_bridge = 0
        self.ros_thread.stop()

    def save_as(self):
        '''
        save rosbag as
        '''
        dlg = wx.FileDialog(self, "Save rosbag as..",
                            co.CONST['rosbag_location'],
                            '',
                            "*.csv", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            path = dlg.GetPath()
            os.rename(self.temp_path, path)
        elif result == wx.ID_CANCEL:
            return


class RosbagRecord(object):
    def __init__(self, channel, path):
        self.channel = channel
        self.path = path

    def __enter__(self):
        args = shlex.split('rosbag record -o' + self.path + ' ' +
                           self.channel)
        self.process = subprocess.Popen(args)

    def __exit__(self, exc_type, exc, traceback):
        self.process.send_signal(signal.SIGINT)
        while self.process.poll() is None:
            time.sleep(1)


class MainFrame(wx.Frame):
    '''
    Main Processing Window
    '''

    def __init__(self, parent, id_, title, farm_key='depth', reg_key='hand',
                 data={}):
        wx.Frame.__init__(self, parent, id_, title)
        self.Center()
        self.farm_key = farm_key
        self.reg_key = reg_key
        self.data = data
        if self.data:
            found = 0
            for key in self.data.keys():
                if self.farm_key in key:
                    self.farm_key = key
                    found = 1
                    break
            if not found:
                raise Exception(
                    'invalid key_string given (default is \'depth\')')
            self.height = self.data[self.farm_key].frames[0].shape[0]
            self.width = self.data[self.farm_key].frames[0].shape[1]
            self.frames_len = len(self.data[self.farm_key].frames)
        else:
            self.height = 0
            self.width = 0
            self.frames_len = 1
        self.actionfarming = far.ActionFarmingProcedure(
            self.farm_key, self.reg_key)
        self.rosbag_process = epr.DataProcess()
        self.rosbag_process.set_keys(self.farm_key, self.reg_key)
        self.main_panel = wx.Panel(self, wx.NewId())
        self.inter_pnl = wx.Panel(self.main_panel, wx.NewId())
        self.act_pnl = wx.Panel(self.main_panel, -1)
        self.mog_pnl = None
        #self.mog_pnl = MOG2params(self, self.main_panel)
        self.mog_pnl = FrameToVideoOperations(self, self.main_panel, self.data)
        # self.recompute_mog2 = wx.Button(
        #    self.main_panel, ID_MASK_RECOMPUTE, "Calculate hand masks")
        #self.Bind(wx.EVT_BUTTON, self.on_recompute_mask, id=ID_MASK_RECOMPUTE)

        lbl_list = ['Memory', 'File']
        self.rbox = wx.RadioBox(self.inter_pnl,
                                label='Save processed result in', choices=lbl_list,
                                majorDimension=1, style=wx.RA_SPECIFY_ROWS)
        self.rbox.SetSelection(0)
        self.append = wx.CheckBox(self.act_pnl, -1, 'Append to existent data')
        self.samples_cb = wx.CheckBox(self.act_pnl, label='Montage Samples')
        self.append.SetValue(1)
        self.load_csv = TButton(self.act_pnl, ID_LOAD_CSV, 'Load csv')
        self.record_bag = TButton(
            self.inter_pnl, ID_BAG_RECORD, 'Record bag')
        self.process_bag = TButton(
            self.inter_pnl, ID_BAG_PROCESS, 'Process bag')
        self.process_all = TButton(
            self.inter_pnl, ID_PROCESS_ALL, 'Process all actions at once')
        self.play = TButton(self.inter_pnl, ID_PLAY, 'Play')
        self.stop = TButton(self.inter_pnl, ID_STOP, 'Stop')
        self.add = TButton(self.act_pnl, ID_ADD, 'Add')
        self.save_csv = TButton(self.act_pnl, ID_SAVE_CSV, 'Save csv')
        self.act_save = TButton(self.act_pnl, ID_ACT_SAVE, 'Save actions')
        self.remove = TButton(self.act_pnl, ID_REMOVE, 'Remove')
        self.txt_inp = NamedTextCtrl(self.act_pnl, 'Action Name:',
                                     wx.HORIZONTAL, wx.ALL | wx.EXPAND)
        self.slider_min = wx.Slider(self.inter_pnl, ID_MIN, 0, 0,
                                    self.frames_len - 1, size=(600, -1),
                                    style=wx.SL_VALUE_LABEL)
        self.slider_max = wx.Slider(self.inter_pnl, ID_MAX, self.frames_len - 1, 0,
                                    self.frames_len - 1, size=(600, -1),
                                    style=wx.SL_VALUE_LABEL)
        self.min = 0
        self.count = 0
        self.working = 0
        self.saved = 1
        self.bag_path = None
        self.lst = EditableListCtrl(self.act_pnl, -1, style=wx.LC_REPORT,
                                    size=(-1, 200))
        self.lst.InsertColumn(0, 'Action Name')
        self.lst.InsertColumn(1, 'Starting Indexes')
        self.lst.InsertColumn(2, 'Ending Indexes')
        self.lst.SetColumnWidth(0, 200)
        self.lst.SetColumnWidth(1, 200)
        self.lst.SetColumnWidth(2, 200)
        self.lst.Bind(wx.EVT_LIST_ITEM_FOCUSED, self.on_lst_item_select)
        # DEBUGGING
        act_sizer = wx.StaticBoxSizer(wx.VERTICAL, self.act_pnl, 'Actions')
        act_top_but_sizer = wx.BoxSizer(wx.HORIZONTAL)
        act_top_but_sizer.AddMany([
            (self.load_csv, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL),
            (self.save_csv, 1, wx.EXPAND | wx.ALL)])
        act_mid_but_sizer = wx.BoxSizer(wx.HORIZONTAL)
        act_mid_but_sizer.AddMany([
            (self.add, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL),
            (self.remove, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL)])
        act_bot_but_sizer = wx.BoxSizer(wx.HORIZONTAL)
        act_bot_but_sizer.AddMany([
            (self.act_save, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL),
            (self.append, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL),
            (self.samples_cb, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL)])
        act_sizer.AddMany([(act_top_but_sizer, 0, wx.EXPAND |
                            wx.ALIGN_LEFT | wx.ALL),
                           (self.txt_inp, 0, wx.EXPAND |
                            wx.ALIGN_LEFT | wx.ALL),
                           (act_mid_but_sizer, 0, wx.EXPAND |
                            wx.ALIGN_LEFT | wx.ALL),
                           (self.lst, 0, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL),
                           (act_bot_but_sizer, 0, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL)])
        self.act_pnl.SetSizer(act_sizer)
        vid_ctrl_box = wx.StaticBoxSizer(wx.HORIZONTAL,
                                         self.inter_pnl,
                                         'Video Controls')
        vid_ctrl_box.AddMany([(self.play, 0), (self.stop, 0)])
        process_box = wx.StaticBoxSizer(wx.HORIZONTAL,
                                        self.inter_pnl,
                                        'Rosbag file processing')
        process_box.AddMany([(self.rbox, 0), (self.record_bag, 0),
                             (self.process_bag, 0),
                             (self.process_all, 0)])
        mis_box = wx.StaticBoxSizer(wx.HORIZONTAL,
                                    self.inter_pnl,
                                    'Starting Frame')
        mas_box = wx.StaticBoxSizer(wx.HORIZONTAL,
                                    self.inter_pnl,
                                    'Ending Frame')
        mas_box.Add(self.slider_max, 0)
        mis_box.Add(self.slider_min, 0)
        slid_box = wx.StaticBoxSizer(wx.VERTICAL,
                                     self.inter_pnl,
                                     'Frames Partition')
        slid_box.AddMany([(mis_box, 0), (mas_box, 0)])
        inter_box = wx.BoxSizer(wx.VERTICAL)
        inter_box.AddMany([(process_box, 0),
                           (vid_ctrl_box, 0),
                           (slid_box, 0)])
        self.sb = self.CreateStatusBar()
        self.nb = None
        self.nb_box = None
        self.min_pnl = None
        self.max_pnl = None
        self.inter_pnl.SetSizer(inter_box)
        self.rgt_box = None
        self.lft_box = wx.BoxSizer(wx.VERTICAL)
        self.lft_box.AddMany([(self.inter_pnl), (self.act_pnl)])
        if self.mog_pnl is not None:
            sboxSizer = wx.StaticBoxSizer(wx.VERTICAL, self.main_panel,
                                          'Stream Postprocessing')
            sboxSizer.Add(self.mog_pnl, wx.ALIGN_CENTER_HORIZONTAL, 1)
            #sboxSizer.AddMany([(self.mog_pnl, 1), (self.recompute_mog2)])
            self.lft_box.Add(sboxSizer)
        self.main_box = wx.BoxSizer(wx.HORIZONTAL)
        self.main_box.AddMany([(self.lft_box, 1)])
        self.main_panel.SetSizer(self.main_box)
        self.framesizer = wx.BoxSizer(wx.HORIZONTAL)
        self.framesizer.Add(self.main_panel)
        self.SetSizerAndFit(self.framesizer)
        self.Bind(wx.EVT_BUTTON, self.on_actions_save, id=ID_ACT_SAVE)
        self.Bind(wx.EVT_BUTTON, self.on_load_csv, id=ID_LOAD_CSV)
        self.Bind(wx.EVT_BUTTON, self.on_process, id=ID_BAG_PROCESS)
        self.Bind(wx.EVT_BUTTON, self.on_record, id=ID_BAG_RECORD)
        self.Bind(wx.EVT_CHECKBOX, self.on_samples_save)
        self.Bind(wx.EVT_BUTTON, self.on_process_all, id=ID_PROCESS_ALL)
        self.Bind(wx.EVT_BUTTON, self.on_play, id=ID_PLAY)
        self.Bind(wx.EVT_BUTTON, self.on_stop, id=ID_STOP)
        self.Bind(wx.EVT_BUTTON, self.on_add, id=ID_ADD)
        self.Bind(wx.EVT_BUTTON, self.on_save_csv, id=ID_SAVE_CSV)
        self.Bind(wx.EVT_BUTTON, self.on_remove, id=ID_REMOVE)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.slider_min.Bind(wx.EVT_SLIDER, self.on_min_slider)
        self.slider_max.Bind(wx.EVT_SLIDER, self.on_max_slider)
        self.csv_name = 'actions.csv'
        wx.CallLater(200, self.SetFocus)
        self._buffer = None
        self.exists_hand_count = None

    def on_record(self, event):
        '''
        record rosbag and process on the fly or after the record
        '''
        self.kinect_recorder = KinectRecorder(self)
        self.kinect_recorder.Show()
        self.kinect_recorder.SetFocus()
        return True

    def on_lst_item_select(self, event):
        ind = self.lst.GetFocusedItem()
        try:
            self.txt_inp.SetValue(self.lst.GetItem(ind, 0).GetText())
        except wx._core.wxAssertionError:
            pass

    def update_drawing(self, main_panel):
        '''
        enable video
        '''
        main_panel.Refresh(False)

    def define_panels_params(self):
        self.min_pnl.SetMinSize(wx.Size(self.width / 4, self.height / 4))
        self.max_pnl.SetMinSize(wx.Size(self.width / 4, self.height / 4))
        self.min_pnl.Bind(wx.EVT_PAINT,
                          lambda event: self.on_frame_change(event,
                                                             self.slider_min,
                                                             self.min_pnl))
        self.max_pnl.Bind(wx.EVT_PAINT,
                          lambda event: self.on_frame_change(event,
                                                             self.slider_max,
                                                             self.max_pnl))
        self.Layout()
        self.Refresh()

    def on_hand_image(self, event, main_panel):
        painter = wx.AutoBufferedPaintDC(self.hnd_pnl)
        painter.Clear()
        try:
            painter.DrawBitmap(getbitmap(main_panel,
                                         self.data[self.reg_key].
                                         frames[self.exists_hand_count]), 0, 0)
        except TypeError:
            self.hnd_pnl.SetBackgroundColour(wx.BLACK)

    def on_recompute_mask(self, event):
        co.chhm.reset()
        dlg = wx.ProgressDialog("Recalculating hand mask",
                                "",
                                maximum=self.frames_len,
                                parent=self,
                                style=0
                                | wx.PD_APP_MODAL
                                | wx.PD_CAN_ABORT
                                | wx.PD_ESTIMATED_TIME
                                | wx.PD_REMAINING_TIME)
        try:
            gmm_num = self.mog_pnl.gmm_num.GetValue()
            bg_ratio = self.mog_pnl.bg_ratio.GetValue()
            var_thres = self.mog_pnl.var_thres.GetValue()
            history = self.mog_pnl.history.GetValue()
            self.rosbag_process.set_mog2_parameters(gmm_num,
                                                    bg_ratio,
                                                    var_thres,
                                                    history)
            logging.debug('gmm_num=' + str(gmm_num) +
                          '\nbg_ratio=' + str(bg_ratio) +
                          '\nvar_thres=' + str(var_thres) +
                          '\nhistory=' + str(history))
            self.rosbag_process.register_hand(self.farm_key, self.reg_key,
                                              overwrite=True, dialog=dlg)
        finally:
            dlg.Destroy()
        return 1

    def on_samples_save(self, event):
        cb = event.GetEventObject()
        if 'Samples' in cb.GetLabel():
            self.rosbag_process.save_samples = int(cb.GetValue()) * 9

    def on_actions_save(self, event):
        self.actionfarming.run(
            self.lst,
            self.bag_path,
            append=self.append.GetValue())

    def on_frame_change(self, event, slider, main_panel):
        '''
        change frame when slider value changes
        '''
        try:
            bmp = getbitmap(main_panel, self.data[
                self.farm_key].frames[slider.GetValue()])
        except KeyError:
            return
        if bmp is not None:
            painter = wx.AutoBufferedPaintDC(main_panel)
            painter.Clear()
            painter.DrawBitmap(bmp, 0, 0)

    def on_min_slider(self, event):
        '''
        update corresponding main_panel when slider value changes
        '''
        _max = self.slider_max.GetValue()
        _min = self.slider_min.GetValue()
        self.slider_max.SetValue(max(_min, _max))
        if self.min_pnl is not None:
            self.update_drawing(self.min_pnl)

    def on_max_slider(self, event):
        '''
        update corresponding main_panel when slider value changes
        '''
        _max = self.slider_max.GetValue()
        _min = self.slider_min.GetValue()
        self.slider_min.SetValue(min(_min, _max))
        if self.max_pnl is not None:
            self.update_drawing(self.max_pnl)

    def on_play(self, event):
        '''
        what to do when Play is pressed
        '''
        # The following should work some time soon, stackoverflow question sent
        if self.nb is not None:
            self.nb.pages[self.nb.GetSelection()].play()
        '''
        if self.nb is not None:
            self.nb.play()
        '''

    def on_stop(self, event):
        '''
        what to do when Stop is pressed
        '''
        if self.nb is not None:
            self.nb.pages[self.nb.GetSelection()].stop()
        '''
        if self.nb is not None:
            self.nb.stop()
        '''

    def on_close(self, event):
        '''
        what to do when window is closed
        '''
        if not self.saved:
            dlg = wx.MessageDialog(self,
                                   'Unsaved Changes, do you want to save them?',
                                   'Exiting', wx.YES_NO | wx.CANCEL)
            result = dlg.ShowModal()
            dlg.Destroy()
            if result == wx.ID_YES:
                self.save_as()
            elif result == wx.ID_CANCEL:
                return
        if self.nb is not None:
            self.nb.pages[self.nb.GetSelection()].timer.Stop()
        '''
        if self.nb is not None:
            self.nb.timer.Stop()
        '''
        self.Destroy()
        return

    def get_list_actions(self):
        count = self.lst.GetItemCount()
        actions = []
        for row in range(count):
            item = self.lst.GetItem(row, 0).GetText()
            actions.append(item)
        return actions

    def on_add(self, event):
        '''
        add item to list, after Add is pressed
        '''
        action_name = self.txt_inp.GetValue()
        if not action_name:
            self.sb.SetStatusText('No addition, missing entry name')
            return
        _min = self.slider_min.GetValue()
        _max = self.slider_max.GetValue()
        if _min > _max:
            self.sb.SetStatusText('No addition, sliders are wrongly set')
            return
        actions = self.get_list_actions()
        try:
            action_index = actions.index(action_name)
            self.lst.SetItem(action_index, 1, self.lst.GetItem(action_index,
                                                               1).GetText()
                             + ',' + str(_min))
            self.lst.SetItem(action_index, 2, self.lst.GetItem(action_index,
                                                               2).GetText()
                             + ',' + str(_max))
        except ValueError:
            self.lst.InsertItem(len(actions), action_name)
            self.lst.SetItem(len(actions), 1, str(_min))
            self.lst.SetItem(len(actions), 2, str(_max))
        # self.txt_inp.Clear()
        self.saved = 0

    def on_remove(self, event):
        '''
        remove selected item from list
        '''
        index = self.lst.GetFocusedItem()
        _min = self.lst.GetItem(index, 1).GetText()
        _max = self.lst.GetItem(index, 2).GetText()
        if ',' not in _min:
            self.lst.DeleteItem(index)
        else:
            minl = _min.split(',')
            maxl = _max.split(',')
            if len(minl) != len(maxl):
                self.sb.SetStatusText(
                    'No removal, min and max have unequal length')
                return
            self.lst.SetItem(index, 1, ",".join(minl[:-1]))
            self.lst.SetItem(index, 2, ",".join(maxl[:-1]))

    def load_csv_file(self, path):
        self.lst.DeleteAllItems()
        with open(path, 'r') as inp:
            for i, line in enumerate(inp):
                item_num = self.lst.GetItemCount()
                inp_items = line.split(':')
                if '\n' in inp_items:
                    inp_items.remove('\n')
                for j, item in enumerate(inp_items):
                    if j == 0:
                        self.lst.InsertItem(item_num, str(item))
                    else:
                        self.lst.SetItem(item_num, j, item)
        return

    def on_load_csv(self, event):
        dlg = wx.FileDialog(self, "Load action list..", os.getcwd(),
                            self.csv_name,
                            "*.csv", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.load_csv_file(dlg.GetPath())
        elif result == wx.ID_CANCEL:
            return

    def load_rosbag(self):
        dlg = wx.FileDialog(self, "Load rosbag..",
                            co.CONST['rosbag_location'], '',
                            "*.bag", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.bag_path = dlg.GetPath()
            self.csv_name = os.path.splitext(
                os.path.basename(self.bag_path))[0] + '.csv'
            self.default_csv_path = os.path.join(
                co.CONST['ground_truth_fold'], self.csv_name)
            if os.path.exists(self.default_csv_path):
                self.load_csv_file(self.default_csv_path)
            return 1
        elif result == wx.ID_CANCEL:
            return result

    def on_process(self, event):
        res = self.load_rosbag()
        if res == wx.ID_CANCEL:
            return res
        baglength = rosbag.Bag(self.bag_path).get_message_count()
        if STOP_COUNT:
            baglength = STOP_COUNT - START_COUNT
        if START_COUNT:
            baglength = baglength - START_COUNT
        dlg = wx.ProgressDialog("Processing rosbag file",
                                "",
                                maximum=baglength,
                                parent=self,
                                style=0
                                | wx.PD_APP_MODAL
                                | wx.PD_ESTIMATED_TIME
                                | wx.PD_REMAINING_TIME
                                | wx.PD_CAN_ABORT)
        try:
            gmm_num = self.mog_pnl.gmm_num.GetValue()
            bg_ratio = self.mog_pnl.bg_ratio.GetValue()
            var_thres = self.mog_pnl.var_thres.GetValue()
            history = self.mog_pnl.history.GetValue()
        except BaseException:
            gmm_num = co.CONST['gmm_num']
            bg_ratio = co.CONST['bg_ratio']
            var_thres = co.CONST['var_thres']
            history = co.CONST['history']
        self.rosbag_process.set_mog2_parameters(gmm_num, bg_ratio, var_thres,
                                                history)
        rbox_sel = self.rbox.GetSelection()
        append = self.append.GetValue()
        self.data.clear()
        self.data.update(self.rosbag_process.run(self.bag_path,
                                                 dialog=dlg, low_ram=1 - rbox_sel,
                                                 save_res=rbox_sel, append=append))
        lengths = [len(self.data[key].frames) for key in self.data.keys()]
        baglength = max(lengths)
        dlg.Destroy()
        dlg.Close()
        self.slider_min.SetMax(baglength - 1)
        self.slider_max.SetMax(baglength - 1)
        self.slider_min.SetValue(START_COUNT)
        self.slider_max.SetValue(baglength - 1)
        found = 0
        if self.farm_key not in self.data.keys():
            for key in self.data.keys():
                if self.farm_key in key:
                    self.farm_key = key
                    found = 1
                    break
        else:
            found = 1
        if not found:
            raise Exception('invalid farm_key given (default is \'depth\')')
        self.height, self.width = self.data[
            self.farm_key].frames[0].shape[:2]
        self.frames_len = len(self.data[self.farm_key].frames)
        if self.nb is None:
            self.nb_box = wx.StaticBox(self.main_panel, -1, 'Data',
                                       style=wx.BORDER_RAISED)
            self.nb_sizer = wx.StaticBoxSizer(self.nb_box, wx.HORIZONTAL)
            self.nb = TopicsNotebook(self.main_panel, self.data,
                                     start_frame_handler=self.slider_min,
                                     end_frame_handler=self.slider_max,
                                     forced_frame_handler=self.slider_min)
            self.nb_sizer.Add(self.nb, 1)
            self.ref_box = wx.StaticBox(self.main_panel, -1, 'Reference Frames',
                                        style=wx.BORDER_RAISED)
            self.ref_sizer = wx.StaticBoxSizer(self.ref_box, wx.HORIZONTAL)
            self.min_pnl_sizer = wx.StaticBoxSizer(wx.VERTICAL, self.main_panel,
                                                   'Starting Frame')
            self.max_pnl_sizer = wx.StaticBoxSizer(wx.VERTICAL, self.main_panel,
                                                   'Ending Frame')
            self.min_pnl = wx.Panel(self.main_panel, -1)
            self.min_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
            self.max_pnl = wx.Panel(self.main_panel, -1)
            self.max_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
            self.min_pnl_sizer.Add(self.min_pnl, 1, wx.EXPAND)
            self.max_pnl_sizer.Add(self.max_pnl, 1, wx.EXPAND)
            self.ref_sizer.Add(self.min_pnl_sizer, 1, wx.SHAPED | wx.ALIGN_CENTER |
                               wx.CENTER)
            self.ref_sizer.Add(self.max_pnl_sizer, 1, wx.SHAPED | wx.ALIGN_CENTER |
                               wx.CENTER)
            self.define_panels_params()
            self.rgt_box = wx.StaticBox(self.main_panel, -1)
            self.rgt_sizer = wx.StaticBoxSizer(self.rgt_box, wx.VERTICAL)
            self.rgt_sizer.AddMany([(self.nb_sizer, 1, wx.EXPAND),
                                    (self.ref_sizer, 1, wx.SHAPED | wx.EXPAND)])
            self.main_box.Add(self.rgt_sizer, 1)
            self.main_box.Layout()
            self.SetSizerAndFit(self.framesizer)
        else:
            self.nb.change_video(self.data)
        return 1

    def save_as(self):
        '''
        save actions list as csv
        '''
        dlg = wx.FileDialog(self, "Save action list as..",
                            co.CONST['ground_truth_fold'],
                            self.csv_name,
                            "*.csv", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'w') as inp:
                for i in range(self.lst.GetItemCount()):
                    for j in range(self.lst.GetColumnCount()):
                        inp.write(self.lst.GetItem(i, j).GetText())
                        inp.write(":")
                    inp.write('\n')
        elif result == wx.ID_CANCEL:
            return

    def on_save_csv(self, event):
        '''
        what to do when Save as.. is pressed
        '''
        self.save_as()
        self.saved = 1

    def on_process_all(self, event):
        ground_truths = [os.path.splitext(fil)[0] for fil
                         in os.listdir(co.CONST['ground_truth_fold'])
                         if fil.endswith('.csv')]
        rosbags = [os.path.splitext(fil)[0] for fil in
                   os.listdir(co.CONST['rosbag_location'])
                   if fil.endswith('.bag')]
        to_process = [rosbag for rosbag in rosbags if (rosbag in ground_truths
                                                       and not
                                                       rosbag.startswith('test'))]
        ground_truths = [os.path.join(
            co.CONST['ground_truth_fold'], name + '.csv') for name in to_process]
        rosbags = [os.path.join(
            co.CONST['rosbag_location'], name + '.bag') for name in to_process]
        count = 0
        for rosbag, ground_truth in zip(rosbags, ground_truths):
            self.actionfarming.run(ground_truth, rosbag, append=count > 0)
            count += 1


LOG = logging.getLogger(__name__)
FORMAT = '%(funcName)20s(%(lineno)s)-%(levelname)s:%(message)s'
CH = logging.Handler()
CH.setFormatter(logging.Formatter(FORMAT))
LOG.addHandler(CH)
LOG.setLevel(logging.DEBUG)


def main():
    '''
    main function
    '''
    logging.basicConfig(format=FORMAT)
    app = wx.App(0)
    frame = MainFrame(None, -1, 'Data Mining')
    frame.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
