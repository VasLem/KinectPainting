'''
One-shot dataset farmer, using single rosbag file
'''
# pylint: disable=unused-argument,too-many-ancestors,too-many-instance-attributes
# pylint: disable=too-many-arguments
import os
import time
import cPickle as pickle
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import wx
import rosbag
import cv2
import class_objects as co

ID_PLAY = wx.NewId()
ID_STOP = wx.NewId()
ID_SAVE = wx.NewId()
ID_ADD = wx.NewId()
ID_MIN = wx.NewId()
ID_MAX = wx.NewId()
ID_REMOVE = wx.NewId()


def getbitmap(img, scale=0):
    '''
    numpy array to bitmap
    '''
    if scale:
        copy = cv2.resize(img, (0, 0), fx=scale, fy=scale)
    else:
        copy = img.copy()
    if len(copy.shape) == 2:
        copy = np.tile(copy[:, :, None], (1, 1, 3))
    if not isinstance(copy[0, 0, 0], np.uint8):
        if np.max(copy) > 5000:
            copy = copy % 256
        else:
            copy[:, :, 0] = copy[:, :, 0] / float(np.max(copy[:, :, 0]) * 255)
            copy[:, :, 1] = copy[:, :, 1] / float(np.max(copy[:, :, 1]) * 255)
            copy[:, :, 2] = copy[:, :, 2] / float(np.max(copy[:, :, 2]) * 255)
        copy = copy.astype(np.uint8)
    image = wx.Image(copy.shape[1], copy.shape[0])
    image.SetData(copy.tostring())
    wx_bitmap = image.ConvertToBitmap()
    return wx_bitmap



class MainFrame(wx.Frame):
    '''
    Main Processing Window
    '''
    def __init__(self, parent, id_, title, processed_rosbag, key):
        frames = processed_rosbag[key].frames
        wx.Frame.__init__(self, parent, id_, title)
        self.height = frames[0].shape[0]
        self.width = frames[0].shape[1]
        self.panel = wx.Panel(self, wx.NewId())
        self.pnl1 = wx.Panel(self.panel, wx.NewId())
        self.pnl2 = wx.Panel(self.panel, wx.NewId(), size=(self.width,
                                                           self.height))
        self.pnl2.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.pnl3 = wx.Panel(self.panel, -1, style=wx.SIMPLE_BORDER)
        self.pnl4 = wx.Panel(self.panel, -1,
                             style=wx.SIMPLE_BORDER, size=(self.width / 4,
                                                           self.height / 4))
        self.pnl4.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.pnl5 = wx.Panel(self.panel, -1, style=wx.SIMPLE_BORDER,
                             size=(self.width / 4, self.height / 4))
        self.pnl5.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.play = wx.Button(self.pnl1, ID_PLAY, 'Play')
        self.stop = wx.Button(self.pnl1, ID_STOP, 'Stop')
        self.add = wx.Button(self.pnl1, ID_ADD, 'Add')
        self.save = wx.Button(self.pnl1, ID_SAVE, 'Save as..')
        self.remove = wx.Button(self.pnl1, ID_REMOVE, 'Remove')
        self.txt_inp = wx.TextCtrl(self.pnl1, -1, name='Action Name')
        self.frames = frames

        self.slider_min = wx.Slider(self.pnl1, ID_MIN, 0, 0,
                                    len(frames) - 1, size=(600, -1))
        self.slider_max = wx.Slider(self.pnl1, ID_MAX, len(frames) - 1, 0,
                                    len(frames) - 1, size=(600, -1))
        self.min = 0
        self.count = 0
        self.working = 0
        self.saved = 1
        self.lst = wx.ListCtrl(self.pnl3, -1, style=wx.LC_REPORT,
                               size=(-1, 200))
        self.lst.InsertColumn(0, 'Action Name')
        self.lst.InsertColumn(1, 'Starting Index')
        self.lst.InsertColumn(2, 'Ending Index')
        self.lst.SetColumnWidth(0, 200)
        self.lst.SetColumnWidth(1, 200)
        self.lst.SetColumnWidth(2, 200)
        hbox = wx.BoxSizer(wx.VERTICAL)
        hbox.Add(self.lst, 1, wx.VERTICAL)
        self.pnl3.SetSizer(hbox)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox1.Add(self.play, 1, wx.ALIGN_CENTER)
        hbox1.Add(self.stop, 1, wx.ALIGN_CENTER)
        hbox1.Add(self.add, 1, wx.ALIGN_CENTER)
        hbox1.Add(self.txt_inp, 1, wx.ALIGN_CENTER)
        hbox1.Add(self.remove, 1, wx.ALIGN_CENTER)
        hbox1.Add(self.save, 1, wx.ALIGN_CENTER)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2.Add(self.slider_min, 1)
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        hbox3.Add(self.slider_max, 1)
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(hbox1, 1, wx.ALIGN_CENTER)
        vbox.Add(hbox2, 1, wx.ALIGN_CENTER)
        vbox.Add(hbox3, 1, wx.ALIGN_CENTER)
        self.pnl1.SetSizer(vbox)
        verpanelsizer1 = wx.BoxSizer(wx.VERTICAL)
        verpanelsizer1.Add(self.pnl1, 1)
        verpanelsizer1.Add(self.pnl3, 1)
        horpanelsizer1 = wx.BoxSizer(wx.HORIZONTAL)
        horpanelsizer1.Add(self.pnl4, 1, wx.ALIGN_CENTER)
        horpanelsizer1.Add(self.pnl5, 1, wx.ALIGN_CENTER)
        verpanelsizer2 = wx.BoxSizer(wx.VERTICAL)
        verpanelsizer2.Add(self.pnl2, 1)
        verpanelsizer2.Add(horpanelsizer1, 1)
        horpanelsizer = wx.BoxSizer(wx.HORIZONTAL)
        horpanelsizer.Add(verpanelsizer1, 1)
        horpanelsizer.Add(verpanelsizer2, 1)
        self.panel.SetSizer(horpanelsizer)
        framesizer = wx.BoxSizer(wx.HORIZONTAL)
        framesizer.Add(self.panel)
        self.SetSizer(framesizer)
        self.Fit()
        self.Bind(wx.EVT_BUTTON, self.on_play, id=ID_PLAY)
        self.Bind(wx.EVT_BUTTON, self.on_stop, id=ID_STOP)
        self.Bind(wx.EVT_BUTTON, self.on_add, id=ID_ADD)
        self.Bind(wx.EVT_BUTTON, self.on_save, id=ID_SAVE)
        self.Bind(wx.EVT_BUTTON, self.on_remove, id=ID_REMOVE)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.slider_min.Bind(wx.EVT_SLIDER,
                             lambda event: self.on_slider(event, self.pnl4))
        self.slider_max.Bind(wx.EVT_SLIDER,
                             lambda event: self.on_slider(event, self.pnl5))

        self.pnl2.Bind(wx.EVT_PAINT, self.on_playing)
        self.pnl4.Bind(wx.EVT_PAINT,
                       lambda event: self.on_frame_change(event,
                                                          self.slider_min,
                                                          self.pnl4))
        self.pnl5.Bind(wx.EVT_PAINT,
                       lambda event: self.on_frame_change(event,
                                                          self.slider_max,
                                                          self.pnl5))
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.timer = wx.Timer(self)
        self.timer.Start(20)

        self.pnl2.Bind(wx.EVT_SIZE, self.on_size)
        wx.CallLater(200, self.SetFocus)
        self._buffer = None
    def on_frame_change(self, event, slider, panel):
        '''
        change frame when slider value changes
        '''
        painter = wx.AutoBufferedPaintDC(panel)
        painter.Clear()
        painter.DrawBitmap(getbitmap(self.frames[slider.GetValue()], 0.25), 0, 0)

    def on_slider(self, event, panel):
        '''
        update corresponding panel when slider value changes
        '''
        self.update_drawing(panel)

    def on_size(self, event):
        '''
        update size of video when size of window changes
        '''
        width, height = self.GetClientSize()
        self._buffer = wx.Bitmap(width, height)
        self.update_drawing(self.pnl2)

    def update_drawing(self, panel):
        '''
        enable video
        '''
        panel.Refresh(False)

    def on_timer(self, event):
        '''
        set timer for video playing
        '''
        self.update_drawing(self.pnl2)

    def on_play(self, event):
        '''
        what to do when Play is pressed
        '''
        if not self.working:
            self.count = self.slider_min.GetValue()
            self.working = 1

    def on_playing(self, event):
        '''
        play video of frames described by sliders
        '''
        if self.working:
            new_min = self.slider_min.GetValue()
            if new_min != self.min:
                self.count = new_min
                self.min = new_min
            painter = wx.AutoBufferedPaintDC(self.pnl2)
            painter.Clear()
            if self.count < self.slider_max.GetValue():
                painter.DrawBitmap(getbitmap(self.frames[self.count]), 0, 0)
                self.count += 1
            else:
                self.working = 0

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
        self.timer.Stop()
        self.Destroy()

    def on_stop(self, event):
        '''
        what to do when Stop is pressed
        '''
        if self.working:
            self.working = 0
            self.count = self.slider_min.GetValue()

    def on_add(self, event):
        '''
        add item to list, after Add is pressed
        '''
        if not self.txt_inp.GetValue():
            return
        num_items = self.lst.GetItemCount()
        self.lst.InsertItem(num_items, self.txt_inp.GetValue())
        self.lst.SetItem(num_items, 1, str(self.slider_min.GetValue()))
        self.lst.SetItem(num_items, 2, str(self.slider_max.GetValue()))
        self.txt_inp.Clear()
        self.saved = 0

    def on_remove(self, event):
        '''
        remove selected item from list
        '''
        index = self.lst.GetFocusedItem()
        self.lst.DeleteItem(index)

    def save_as(self):
        '''
        save actions list as csv
        '''
        dlg = wx.FileDialog(self, "Save action list as..", os.getcwd(),
                            "actions.csv",
                            "*.csv", wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'w') as inp:
                for i in range(self.lst.GetItemCount()):
                    for j in range(self.lst.GetColumnCount()):
                        inp.write(self.lst.GetItem(i, j).GetText())
                        inp.write(",")
                    inp.write('\n')
        elif result == wx.ID_CANCEL:
            return

    def on_save(self, event):
        '''
        what to do when Save as.. is pressed
        '''
        self.save_as()
        self.saved = 1


class RosbagStruct(object):
    '''
    class to hold extracted rosbag single topic
    '''
    def __init__(self, name=''):
        self.name = name
        self.frames = []
        self.sync = []
        self.timestamps = []


def rosbag_process(bag_path=co.CONST['bag_path'], filt=True, convstr='color',
                   convtype='rgb8', save=False, load=False, pathname='extracted_bag'):
    '''
    Save rosbag frames to a dictionary according to the topic. Using as reference the
    newest topic with the closest timestamp , a synchronization vector is made. Each
    dictionary contains:
    ->a sublist, containing every captured frame
    ->the corresponding synchronization vector
    ->the corresponing ROS timestamp vector
    if color_filt, any topic containing string convstr  is converted to
    convtype
    if save, dictionary is saved to file pathname
    if load, dictionary is loaded from file pathname
    '''
    loaded = 0
    if load:
        try:
            with open(pathname, 'r') as inp:
                bagdata = pickle.load(inp)
            loaded = 1
        except (IOError, EOFError):
            print 'No file available, repeating process'
    if not loaded:
        bridge = CvBridge()
        bagdata = {}
        prev_topic = ''
        sync_count = 0
        for topic, msg, timestamp in rosbag.Bag(co.CONST['bag_path']).read_messages():
            if filt:
                if convstr in topic:
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg, convtype)
                    except CvBridgeError as err:
                        print err
                else:
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough')
                    except CvBridgeError as err:
                        print err
            else:
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough')
                except CvBridgeError as err:
                    print err
            try:
                bagdata[topic].frames.append(cv_image)
            except (AttributeError, KeyError):
                bagdata[topic] = RosbagStruct(topic)
                bagdata[topic].frames.append(cv_image)

            try:
                if topic == prev_topic or bagdata[topic].sync[-1] == sync_count:
                    sync_count += 1
            except IndexError:
                if topic == prev_topic:
                    sync_count += 1

            bagdata[topic].timestamps.append(timestamp)
            bagdata[topic].sync.append(sync_count)
        if save:
            with open(pathname, 'w') as out:
                pickle.dump(bagdata, out)
    return bagdata


def main():
    '''
    main function
    '''
    bagdata = rosbag_process(save=False, load=False)
    for key in bagdata.keys():
        if 'depth' in key:
            dep_key = key
        elif 'color' in key:
            col_key = key
    app = wx.App(0)
    frame = MainFrame(None, -1, 'Data Mining', bagdata, dep_key)
    frame.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
