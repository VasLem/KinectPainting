'''
One-shot dataset farmer, using single rosbag file
'''
# pylint: disable=unused-argument,too-many-ancestors,too-many-instance-attributes
# pylint: disable=too-many-arguments
import logging
FORMAT='%(funcName)20s(%(lineno)s)-%(levelname)s:%(message)s'
logging.basicConfig(format=FORMAT)
import os
import time
import cPickle as pickle
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import wx
import rosbag
import cv2
import class_objects as co
import moving_object_detection_alg as moda
import hand_segmentation_alg as hsa
logging.getLogger().setLevel(logging.DEBUG)
ID_LOAD = wx.NewId()
ID_LOADBAG = wx.NewId()
ID_BAG_PROCESS = wx.NewId()
ID_PLAY = wx.NewId()
ID_STOP = wx.NewId()
ID_SAVE = wx.NewId()
ID_ADD = wx.NewId()
ID_MIN = wx.NewId()
ID_MAX = wx.NewId()
ID_REMOVE = wx.NewId()
ID_MASK_RECOMPUTE = wx.NewId()
bagdata={}
STOP_COUNT = 0

def getbitmap(main_panel,img):
    '''
    numpy array to bitmap
    '''
    psize = main_panel.GetSize()
    if img.shape[0]!=psize[0] or img.shape[1]!=psize[1]:
        copy = cv2.resize(img, (psize[0],psize[1]))
    else:
        copy = img.copy()
    if len(copy.shape) == 2:
        copy = np.tile(copy[:, :, None], (1, 1, 3))
    if not isinstance(copy[0, 0, 0], np.uint8) or np.max(copy)==1:
        if np.max(copy) > 5000:
            copy = copy % 256
        else:
            copy = (copy / float(np.max(copy))) * 255
        copy = copy.astype(np.uint8)
    image = wx.Image(copy.shape[1], copy.shape[0])
    image.SetData(copy.tostring())
    wx_bitmap = image.ConvertToBitmap()
    return wx_bitmap


class SpinStepCtrl(wx.SpinCtrlDouble):
    def __init__(self,parent, *args, **kwargs):
        wx.SpinCtrlDouble.__init__(self,parent,id=wx.ID_ANY,value='0.00',pos=wx.DefaultPosition,
                             size=wx.DefaultSize,
                             inc=kwargs.get('step'), 
                             min=kwargs.get('min_val',0.00),
                             max=kwargs.get('max_val',100.00))
        self.SetValue(kwargs.get('init_val',0.00))



class LabeledSpinStepCtrlSizer(wx.BoxSizer):
    def __init__(self, parent, name, orientation=wx.VERTICAL, step=0, init_val=0,
				 set_range=(0.0,100.0)):
        self.name=name
        wx.BoxSizer.__init__(self,orientation)
        centeredLabel = wx.StaticText(parent, -1,name)
        self.Add(centeredLabel, flag=wx.ALIGN_CENTER_HORIZONTAL)
        self.ctrl = SpinStepCtrl(parent, step=step, 
									init_val=init_val,
									min_val=set_range[0],
									max_val=set_range[1])
        self.Add(self.ctrl, flag=wx.EXPAND)
     
    def GetValue(self):
        return self.ctrl.GetValue()

class MOG2params(wx.Panel):
    def __init__(self, parent, main_panel):
        wx.Panel.__init__(self, main_panel)
        self.bg_ratio = LabeledSpinStepCtrlSizer(self, init_val=co.CONST['bg_ratio'],
                                                    set_range=(0,1),
                                                    step=0.05,
                                                    name='bg_ratio')
        self.var_thres = LabeledSpinStepCtrlSizer(self, init_val=co.CONST[
            'var_thres'],
                                                    set_range=(0,100),
                                                    step=1,
                                                    name='var_thres')
        self.gmm_num = LabeledSpinStepCtrlSizer(self, init_val=co.CONST['gmm_num'],
                                                    set_range=(1,30),
                                                    step=1,
                                                    name='gmm_num')
        self.history = LabeledSpinStepCtrlSizer(self, init_val=co.CONST['history'],
                                                    set_range=(0,10000),
                                                    step=100,
                                                    name='history')
        box = wx.BoxSizer(wx.VERTICAL)
        box.AddMany((self.bg_ratio,self.var_thres,self.gmm_num,self.history))
        self.SetSizer(box)


class MainFrame(wx.Frame):
    '''
    Main Processing Window
    '''
    def __init__(self, parent, id_, title, processed_rosbag=None, key_string=None):
        wx.Frame.__init__(self, parent, id_, title)
        global bagdata
        if key_string is not None:
            self.key_string = key_string
        else:
            self.key_string = 'depth'
        self.key = None
        if processed_rosbag is not None:
            for key in processed_rosbag.keys():
                if self.key_string in key:
                    self.key = key
            if self.key is not None:
                bagdata = processed_rosbag
        if bagdata:
            self.height = bagdata[self.key].frames[0].shape[0]
            self.width = bagdata[self.key].frames[0].shape[1]
            self.frames_len = len(bagdata[self.key].frames)
        else:
            self.height = 0
            self.width =0
            self.frames_len = 1
        self.main_panel = wx.Panel(self, wx.NewId())
        self.inter_pnl = wx.Panel(self.main_panel, wx.NewId())
        self.vid_pnl = wx.Panel(self.main_panel, wx.NewId())
        self.vid_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.rep_pnl = wx.Panel(self.main_panel, -1, style=wx.SIMPLE_BORDER)
        self.min_pnl = wx.Panel(self.main_panel, -1,
                             style=wx.SIMPLE_BORDER)
        self.min_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.max_pnl = wx.Panel(self.main_panel, -1, style=wx.SIMPLE_BORDER)
        self.hnd_pnl = None
        self.mog_pnl = None
        self.hnd_pnl = wx.Panel(self.main_panel, -1, style = wx.SIMPLE_BORDER)
        self.hnd_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.mog_pnl = MOG2params(self,self.main_panel)
        self.recompute_mog2 = wx.Button(self.main_panel, ID_MASK_RECOMPUTE, "Calculate hand masks")
        self.Bind(wx.EVT_BUTTON, self.on_recompute_mask, id=ID_MASK_RECOMPUTE)


        self.max_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)


        self.load = wx.Button(self.inter_pnl, ID_LOAD, 'Load csv')
        self.loadbag = wx.Button(self.inter_pnl, ID_LOADBAG, 'Load bag')
        self.process_bag = wx.Button(self.inter_pnl, ID_BAG_PROCESS, 'Process bag')
        self.play = wx.Button(self.inter_pnl, ID_PLAY, 'Play')
        self.stop = wx.Button(self.inter_pnl, ID_STOP, 'Stop')
        self.add = wx.Button(self.inter_pnl, ID_ADD, 'Add')
        self.save = wx.Button(self.inter_pnl, ID_SAVE, 'Save csv')
        self.remove = wx.Button(self.inter_pnl, ID_REMOVE, 'Remove')
        self.txt_inp = wx.TextCtrl(self.inter_pnl, -1, name='Action Name')
        self.slider_min = wx.Slider(self.inter_pnl, ID_MIN, 0, 0,
                                    self.frames_len - 1, size=(600, -1))
        self.slider_max = wx.Slider(self.inter_pnl, ID_MAX, self.frames_len - 1, 0,
                                    self.frames_len - 1, size=(600, -1))
        self.min = 0
        self.count = 0
        self.working = 0
        self.saved = 1
        self.bag_path = None
        self.lst = wx.ListCtrl(self.rep_pnl, -1, style=wx.LC_REPORT,
                               size=(-1, 200))
        self.lst.InsertColumn(0, 'Action Name')
        self.lst.InsertColumn(1, 'Starting Index')
        self.lst.InsertColumn(2, 'Ending Index')
        self.lst.SetColumnWidth(0, 200)
        self.lst.SetColumnWidth(1, 200)
        self.lst.SetColumnWidth(2, 200)
        lst_box = wx.BoxSizer(wx.VERTICAL)
        lst_box.Add(self.lst, 1, wx.VERTICAL)
        self.rep_pnl.SetSizer(lst_box)
        load_box = wx.BoxSizer(wx.HORIZONTAL)
        load_box.AddMany([(self.load,1),
                       (self.loadbag,1),
                       (self.process_bag,1)])
        but_box = wx.BoxSizer(wx.HORIZONTAL)
        but_box.AddMany([(self.play,1),(self.stop,1),
                       (self.add,1),(self.txt_inp,1),
                       (self.remove,1),(self.save, 1)])
        mis_box = wx.BoxSizer(wx.HORIZONTAL)
        mis_box.Add(self.slider_min, 1)
        mas_box = wx.BoxSizer(wx.HORIZONTAL)
        mas_box.Add(self.slider_max, 1)
        inter_box = wx.BoxSizer(wx.VERTICAL)
        inter_box.AddMany([(load_box,1),
                           (but_box,1),
                      (mis_box,1),
                      (mas_box,1)])
        self.inter_pnl.SetSizer(inter_box)
        self.lft_box = wx.BoxSizer(wx.VERTICAL)
        self.lft_box.AddMany([(self.inter_pnl,1),(self.rep_pnl,1)])
        if self.mog_pnl is not None:
            self.lft_box.AddMany([(self.mog_pnl,1),
                              (self.recompute_mog2,1)])
        ref_box = wx.BoxSizer(wx.HORIZONTAL)
        ref_box.Add(self.min_pnl, 1, wx.ALIGN_CENTER)
        if self.hnd_pnl is not None:
            ref_box.Add(self.hnd_pnl, 1, wx.ALIGN_CENTER)
        ref_box.Add(self.max_pnl, 1, wx.ALIGN_CENTER)
        self.rgt_box = wx.BoxSizer(wx.VERTICAL)
        self.rgt_box.AddMany([(self.vid_pnl, 1),
                              (ref_box, 1)])
        self.main_box = wx.BoxSizer(wx.HORIZONTAL)
        self.main_box.AddMany([(self.lft_box, 1),
                               (self.rgt_box, 1)])
        self.main_panel.SetSizer(self.main_box)
        self.framesizer = wx.BoxSizer(wx.HORIZONTAL)
        self.framesizer.Add(self.main_panel)
        self.SetSizer(self.framesizer)
        self.Fit()
        self.Bind(wx.EVT_BUTTON, self.on_load, id=ID_LOAD)
        self.Bind(wx.EVT_BUTTON, self.on_load_rosbag, id=ID_LOADBAG)
        self.Bind(wx.EVT_BUTTON, self.on_process, id=ID_BAG_PROCESS)
        self.Bind(wx.EVT_BUTTON, self.on_play, id=ID_PLAY)
        self.Bind(wx.EVT_BUTTON, self.on_stop, id=ID_STOP)
        self.Bind(wx.EVT_BUTTON, self.on_add, id=ID_ADD)
        self.Bind(wx.EVT_BUTTON, self.on_save, id=ID_SAVE)
        self.Bind(wx.EVT_BUTTON, self.on_remove, id=ID_REMOVE)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.slider_min.Bind(wx.EVT_SLIDER,
                             lambda event: self.on_slider(event, self.min_pnl))
        self.slider_max.Bind(wx.EVT_SLIDER,
                             lambda event: self.on_slider(event, self.max_pnl))
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.timer = wx.Timer(self)
        self.timer.Start(20)

        self.vid_pnl.Bind(wx.EVT_SIZE, self.on_size)
        wx.CallLater(200, self.SetFocus)
        self._buffer = None
        self.exists_hand_count = None
        if processed_rosbag is not None:
            self.define_panels_params()

    def define_panels_params(self):
        self.vid_pnl.SetMinSize(wx.Size(self.width,self.height))
        self.min_pnl.SetMinSize(wx.Size(self.width/4,self.height/4))
        self.max_pnl.SetMinSize(wx.Size(self.width/4,self.height/4))
        self.vid_pnl.Bind(wx.EVT_PAINT,
                             lambda event: self.on_playing(event, self.vid_pnl))
        self.min_pnl.Bind(wx.EVT_PAINT,
                       lambda event: self.on_frame_change(event,
                                                          self.slider_min,
                                                          self.min_pnl))
        self.max_pnl.Bind(wx.EVT_PAINT,
                       lambda event: self.on_frame_change(event,
                                                          self.slider_max,
                                                          self.max_pnl))
        self.hnd_pnl.Bind(wx.EVT_PAINT,
                       lambda event: self.on_hand_image(event, self.hnd_pnl))
        self.hnd_pnl.SetMinSize(wx.Size(self.width/4,self.height/4))
        self.Layout()
        self.Refresh()


    def on_hand_image(self, event, main_panel):
        painter = wx.AutoBufferedPaintDC(self.hnd_pnl)
        painter.Clear()
        try:
            painter.DrawBitmap(getbitmap(main_panel,
                                         bagdata['hand'].frames[self.exists_hand_count]), 0, 0)
        except TypeError:
            self.hnd_pnl.SetBackgroundColour(wx.BLACK)

    def on_recompute_mask(self, event):
        global bagdata
        co.chhm.reset()
        dep_key=None
        hand_key=None
        while True:
            for key in bagdata.keys():
                if 'depth' in key:
                    dep_key = key
                elif 'hand' in key:
                    hand_key = key
            if dep_key is None:
                res = self.on_process(None)
                if res==wx.ID_CANCEL:
                    return res
            else:
                break
        if hand_key is None:
            hand_key = 'hand'
        dlg = wx.ProgressDialog("Recalculating hand mask",
                               "",
                               maximum = self.frames_len,
                               parent = self,
                               style = 0
                               | wx.PD_APP_MODAL
                               | wx.PD_CAN_ABORT
                               | wx.PD_ESTIMATED_TIME
                               | wx.PD_REMAINING_TIME)
        try:
            gmm_num=self.mog_pnl.gmm_num.GetValue()
            bg_ratio=self.mog_pnl.bg_ratio.GetValue()
            var_thres=self.mog_pnl.var_thres.GetValue()
            history=self.mog_pnl.history.GetValue()

            logging.debug('gmm_num='+str(gmm_num) +
                          '\nbg_ratio='+str(bg_ratio) +
                          '\nvar_thres='+str(var_thres) +
                          '\nhistory='+str(history))
            for count in range(len(bagdata[dep_key].frames)):
                moda.detection_by_mog2(False,bagdata[dep_key].frames[count],
                                       gmm_num=gmm_num,
                                       bg_ratio=bg_ratio,
                                       var_thres=var_thres,
                                       history=history)
                hand_patch, hand_patch_pos, mask = hsa.main_process_upgraded(
                    co.meas.found_objects_mask.astype(
                        np.uint8))
                if hand_patch is not None:
                    try:
                        bagdata[hand_key].frames.append(
                            mask*bagdata[dep_key].frames[count])
                    except (AttributeError, KeyError):
                        bagdata[hand_key] = RosbagStruct('hand')
                        bagdata[hand_key].frames.append(mask*bagdata[dep_key].frames[count])
                    bagdata[hand_key].info.append(hand_patch_pos)
                    bagdata[hand_key].timestamps.append(bagdata[dep_key].timestamps[count])
                    bagdata[hand_key].sync.append(bagdata[dep_key].sync[count])
                wx.Yield()
                keepGoing,_=dlg.Update(count)
                if not keepGoing:
                    dlg.Destroy()
                    return wx.ID_CANCEL
        finally:
            dlg.Destroy()
        co.chhm.print_stats()
        return 1


    def on_frame_change(self, event, slider, main_panel):
        '''
        change frame when slider value changes
        '''
        painter = wx.AutoBufferedPaintDC(main_panel)
        painter.Clear()
        painter.DrawBitmap(getbitmap(main_panel, bagdata[self.key].frames[slider.GetValue()]), 0, 0)

    def on_slider(self, event, main_panel):
        '''
        update corresponding main_panel when slider value changes
        '''
        self.update_drawing(main_panel)

    def on_size(self, event):
        '''
        update size of video when size of window changes
        '''
        width, height = self.GetClientSize()
        self._buffer = wx.Bitmap(width, height)
        self.update_drawing(self.vid_pnl)

    def update_drawing(self, main_panel):
        '''
        enable video
        '''
        main_panel.Refresh(False)

    def on_timer(self, event):
        '''
        set timer for video playing
        '''
        self.update_drawing(self.vid_pnl)
        if self.exists_hand_count is not None:
            self.update_drawing(self.hnd_pnl)


    def on_play(self, event):
        '''
        what to do when Play is pressed
        '''
        if not self.working:
            self.count = self.slider_min.GetValue()
            self.working = 1

    def on_playing(self, event, main_panel):
        '''
        play video of frames described by sliders
        '''
        if self.working:
            new_min = self.slider_min.GetValue()
            if new_min != self.min:
                self.count = new_min
                self.min = new_min
            painter = wx.AutoBufferedPaintDC(self.vid_pnl)
            painter.Clear()
            if self.count < self.slider_max.GetValue():
                painter.DrawBitmap(getbitmap(main_panel,bagdata[self.key].frames[self.count]), 0, 0)
                if self.hnd_pnl is not None:
                    try:
                        self.exists_hand_count = bagdata[
                            'hand'].sync.index(bagdata[self.key].sync[self.count])
                    except (ValueError, KeyError):
                        self.exists_hand_count = None
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


    def on_load(self, event):
        dlg = wx.FileDialog(self, "Load action list..", os.getcwd(),
                            "actions.csv",
                            "*.csv", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            path = dlg.GetPath()
            with open(path, 'r') as inp:
                for i,line in enumerate(inp):
                    item_num=self.lst.GetItemCount()
                    inp_items=line.split(',')
                    if '\n' in inp_items:
                        inp_items.remove('\n')
                    for j,item in enumerate(inp_items):
                        if j==0:
                            self.lst.InsertItem(item_num, str(item))
                        else:
                            self.lst.SetItem(item_num, j, item)
        elif result == wx.ID_CANCEL:
            return

    def on_load_rosbag(self, event):
        dlg = wx.FileDialog(self, "Load rosbag..",
                            co.CONST['rosbag_location'],'',
                            "*.bag", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
        result = dlg.ShowModal()
        dlg.Destroy()
        if result == wx.ID_OK:
            self.bag_path = dlg.GetPath()
            return 1
        elif result == wx.ID_CANCEL:
            return result
    def on_process(self, event):
        if self.bag_path is None:
            res = self.on_load_rosbag(None)
            if res == wx.ID_CANCEL:
                return res
        baglength =  rosbag.Bag(self.bag_path).get_message_count()
        if STOP_COUNT:
            baglength = STOP_COUNT
        dlg = wx.ProgressDialog("Processing rosbag file",
                                "",
                                maximum = baglength,
                                parent = self,
                                style = 0
                                | wx.PD_APP_MODAL
                                | wx.PD_ESTIMATED_TIME
                                | wx.PD_REMAINING_TIME
                                | wx.PD_CAN_ABORT)
        gmm_num=self.mog_pnl.gmm_num.GetValue()
        bg_ratio=self.mog_pnl.bg_ratio.GetValue()
        var_thres=self.mog_pnl.var_thres.GetValue()
        history=self.mog_pnl.history.GetValue()
        rosbag_process(parent=dlg, gmm_num=gmm_num,
                      bg_ratio=bg_ratio,
                      var_thres=var_thres,
                      history=history)
        dlg.Destroy()
        dlg.Close()
        self.slider_min.SetMax(baglength-1)
        self.slider_max.SetMax(baglength-1)
        self.slider_max.SetValue(baglength-1)
        if self.key is None:
            for key in bagdata.keys():
                if 'depth' in key:
                    self.key = key
                    break
        self.height, self.width = bagdata[self.key].frames[0].shape[:2]
        self.frames_len = len(bagdata[self.key].frames)
        self.define_panels_params()
        return 1


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
        self.info = []


def rosbag_process(bag_path=co.CONST['train_bag_path'], filt=True, convstr='color',
                   convtype='rgb8', save=False, load=False,
                   keep_depth_only=True,
                   pathname='extracted_bag', low_ram=True,
                   detect_hand=True,parent = None,
                  gmm_num=co.CONST['gmm_num'],bg_ratio=co.CONST['bg_ratio'],
                  var_thres = co.CONST['var_thres'],history=co.CONST['history']):
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
    global bagdata
    bagdata= {}
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
        prev_topic = ''
        sync_count = 0
        count=0
        logging.debug('gmm_num='+str(gmm_num) +
                      '\nbg_ratio='+str(bg_ratio) +
                      '\nvar_thres='+str(var_thres) +
                      '\nhistory='+str(history))
        for topic, msg, timestamp in rosbag.Bag(bag_path).read_messages():
            #Reading image Stage
            if keep_depth_only:
                if 'depth' in topic:
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg, 'passthrough')
                    except CvBridgeError as err:
                        print err
                else:
                    continue
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
            #Saving stage. Images are saved as grayscale and uint8 for less RAM
            if low_ram:
                copy = cv_image.copy()
                if len(copy.shape)==3:
                    copy=np.mean(copy,axis=2)
                if not isinstance(copy[0, 0], np.uint8) or np.max(copy)==1:
                    if np.max(copy) > 5000:
                        copy = copy % 256
                    else:
                        copy = (copy / float(np.max(copy))) * 255
                    copy = copy.astype(np.uint8)
            else:
                copy = cv_image

            try:
                bagdata[topic].frames.append(copy)
            except (AttributeError, KeyError):
                bagdata[topic] = RosbagStruct(topic)
                bagdata[topic].frames.append(copy)
            try:
                if topic == prev_topic or bagdata[topic].sync[-1] == sync_count:
                    sync_count += 1
            except IndexError:
                if topic == prev_topic:
                    sync_count += 1

            bagdata[topic].timestamps.append(timestamp)
            bagdata[topic].sync.append(sync_count)
            if detect_hand:
                if 'depth' in topic:
                    if co.edges.calib_edges is None:
                        co.edges.load_calib_data(img=cv_image,whole_im=True)
                    co.data.depth_im = cv_image.copy()
                    co.noise_proc.remove_noise(6000, False)
                    mog2_res = moda.detection_by_mog2(False,
                                           gmm_num=gmm_num,
                                           bg_ratio=bg_ratio,
                                           var_thres=var_thres,
                                           history=history)
                    hand_patch, hand_patch_pos, masks = \
                            hsa.main_process_upgraded(
                        co.meas.found_objects_mask.astype(
                            np.uint8))
                    if hand_patch is not None:
                        outp = (masks[0]>0)*cv_image
                        try:
                            bagdata['hand'].frames.append(
                                outp)
                        except (AttributeError, KeyError):
                            bagdata['hand'] = RosbagStruct('hand')
                            bagdata['hand'].frames.append(outp)
                        bagdata['hand'].info.append(hand_patch_pos)
                        bagdata['hand'].timestamps.append(timestamp)
                        bagdata['hand'].sync.append(sync_count)
                    else:
                        outp = (mog2_res>0)*cv_image
            count+=1
            #DEBUGGING
            if count==STOP_COUNT:
                break
            if parent is not None:
                wx.Yield()
                keepGoing,_= parent.Update(count-1)
                if not keepGoing:
                    return outp
        co.chhm.print_stats()
        if save:
            with open(pathname, 'w') as out:
                pickle.dump(bagdata, out)
    return outp

pathname='/media/vassilis/Thesis/Datasets/PersonalFarm/TrainingActions/all.bag'
def main():
    '''
    main function
    '''
    app = wx.App(0)
    frame = MainFrame(None, -1, 'Data Mining')
    frame.Show(True)
    app.MainLoop()


if __name__ == '__main__':
    main()
