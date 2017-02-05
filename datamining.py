'''
One-shot dataset farmer, using single rosbag file
'''
# pylint: disable=unused-argument,too-many-ancestors,too-many-instance-attributes
# pylint: disable=too-many-arguments
import os
import logging
import warnings
import errno
import time
import wx
import wx.lib.mixins.listctrl as listmix
from skimage import io, exposure, img_as_uint
import numpy as np
import yaml
import rosbag
import cv2
from cv_bridge import CvBridge, CvBridgeError
import class_objects as co
import moving_object_detection_alg as moda
import hand_segmentation_alg as hsa
import extract_and_process_rosbag as epr
import full_actions_registration as far
io.use_plugin('freeimage')
logging.getLogger().setLevel(logging.DEBUG)
ID_LOAD_CSV = wx.NewId()
ID_BAG_PROCESS = wx.NewId()
ID_PLAY = wx.NewId()
ID_STOP = wx.NewId()
ID_SAVE_CSV = wx.NewId()
ID_ADD = wx.NewId()
ID_MIN = wx.NewId()
ID_MAX = wx.NewId()
ID_REMOVE = wx.NewId()
ID_MASK_RECOMPUTE = wx.NewId()
ID_ACT_SAVE = wx.NewId()
CURR_DIR = os.getcwd()
ACTIONS_SAVE_PATH = os.path.join(CURR_DIR, 'actions')
ROSBAG_WHOLE_RES_SAVE_PATH = os.path.join(CURR_DIR, 'whole_result')
START_COUNT = 0  # 300
STOP_COUNT = 0  # 600
try:
    with open('descriptions.yaml', 'r') as inp:
        DESCRIPTIONS = yaml.load(inp)
except:
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
            copy = (copy / float(np.max(copy))) * 255
        copy = copy.astype(np.uint8)
    image = wx.Image(copy.shape[1], copy.shape[0])
    image.SetData(copy.tostring())
    wx_bitmap = image.ConvertToBitmap()
    return wx_bitmap


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
        centeredLabel = wx.StaticText(parent, -1, name)
        if orientation == wx.VERTICAL:
            self.Add(centeredLabel, 0, wx.ALIGN_CENTER_HORIZONTAL)
        else:
            self.Add(centeredLabel, 0, wx.ALIGN_CENTER_VERTICAL)
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
        centeredLabel = wx.StaticText(parent, -1, name)
        self.Add(centeredLabel, flag=wx.ALIGN_CENTER_HORIZONTAL)
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
        self.pages = []
        max_len = 0
        for topic in data.keys():
            max_len = max(len(data[topic].frames), max_len)
        for topic in data.keys():
            if data[topic].frames:
                inp = [None] * max_len
                for count, sync_count in enumerate(data[topic].sync):
                    inp[sync_count] = data[topic].frames[count]
                self.pages.append(VideoPanel(self, inp, fps,
                                             start_frame_handler,
                                             end_frame_handler,
                                             forced_frame_handler))
                data[topic].frames = inp
                data[topic].sync = []
            else:
                txtPanel = wx.Panel(self)
                wx.StaticText(txtPanel, id=-1, label="This topic  was not saved in memory",
                              style=wx.ALIGN_CENTER, name="")
                self.pages.append(txtPanel)
            if len(topic) > 10:
                label = topic[-10:]
            else:
                label = topic
            self.AddPage(self.pages[-1], label)


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
        self.rosbag_process = epr.RosbagProcess()
        self.rosbag_process.set_keys(self.farm_key, self.reg_key)
        self.main_panel = wx.Panel(self, wx.NewId())
        self.inter_pnl = wx.Panel(self.main_panel, wx.NewId())
        self.act_pnl = wx.Panel(self.main_panel, -1)
        self.mog_pnl = None
        self.mog_pnl = MOG2params(self, self.main_panel)
        self.recompute_mog2 = wx.Button(
            self.main_panel, ID_MASK_RECOMPUTE, "Calculate hand masks")
        self.Bind(wx.EVT_BUTTON, self.on_recompute_mask, id=ID_MASK_RECOMPUTE)

        lbl_list = ['Memory', 'File']
        self.rbox = wx.RadioBox(self.inter_pnl,
                                label='Save processed result in', choices=lbl_list,
                                majorDimension=1, style=wx.RA_SPECIFY_ROWS)
        self.rbox.SetSelection(0)
        self.load_csv = TButton(self.act_pnl, ID_LOAD_CSV, 'Load csv')
        self.process_bag = TButton(
            self.inter_pnl, ID_BAG_PROCESS, 'Process bag')
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
        self.load_csv_file('actions.csv')
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
        act_bot_but_sizer.Add(
            self.act_save, 1, wx.EXPAND | wx.ALIGN_LEFT | wx.ALL)
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
        process_box.AddMany([(self.rbox, 0), (self.process_bag, 0)])
        mis_box = wx.StaticBoxSizer(wx.HORIZONTAL,
                                    self.inter_pnl,
                                    'Starting Frame')
        mas_box = wx.StaticBoxSizer(wx.HORIZONTAL,
                                    self.inter_pnl,
                                    'Ending Frame')
        mas_box.Add(self.slider_max, 0)
        mis_box.Add(self.slider_min, 0)
        inter_box = wx.BoxSizer(wx.VERTICAL)
        slid_box = wx.StaticBoxSizer(wx.VERTICAL,
                                     self.inter_pnl,
                                     'Frames Partition')
        slid_box.AddMany([(mis_box, 0), (mas_box, 0)])
        inter_box.AddMany([(process_box, 0),
                           (vid_ctrl_box, 0),
                           (slid_box, 0)])
        self.sb = self.CreateStatusBar()
        self.nb = None
        self.min_pnl = None
        self.max_pnl = None
        self.inter_pnl.SetSizer(inter_box)
        self.rgt_box = None
        self.lft_box = wx.BoxSizer(wx.VERTICAL)
        self.lft_box.AddMany([(self.inter_pnl), (self.act_pnl)])
        if self.mog_pnl is not None:
            sboxSizer = wx.StaticBoxSizer(wx.VERTICAL, self.main_panel, 'MOG2')
            sboxSizer.AddMany([(self.mog_pnl, 1), (self.recompute_mog2)])
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
        self.Bind(wx.EVT_BUTTON, self.on_play, id=ID_PLAY)
        self.Bind(wx.EVT_BUTTON, self.on_stop, id=ID_STOP)
        self.Bind(wx.EVT_BUTTON, self.on_add, id=ID_ADD)
        self.Bind(wx.EVT_BUTTON, self.on_save_csv, id=ID_SAVE_CSV)
        self.Bind(wx.EVT_BUTTON, self.on_remove, id=ID_REMOVE)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.slider_min.Bind(wx.EVT_SLIDER, self.on_min_slider)
        self.slider_max.Bind(wx.EVT_SLIDER, self.on_max_slider)

        wx.CallLater(200, self.SetFocus)
        self._buffer = None
        self.exists_hand_count = None

    def on_lst_item_select(self, event):
        ind = self.lst.GetFocusedItem()
        self.txt_inp.SetValue(self.lst.GetItem(ind, 0).GetText())

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

    def on_actions_save(self, event):
        self.actionfarming.run(self.lst)

    def on_frame_change(self, event, slider, main_panel):
        '''
        change frame when slider value changes
        '''
        painter = wx.AutoBufferedPaintDC(main_panel)
        painter.Clear()
        painter.DrawBitmap(getbitmap(main_panel, self.data[
            self.farm_key].frames[slider.GetValue()]), 0, 0)

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
                            "actions.csv",
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
            return 1
        elif result == wx.ID_CANCEL:
            return result

    def on_process(self, event):
        if self.bag_path is None:
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
        gmm_num = self.mog_pnl.gmm_num.GetValue()
        bg_ratio = self.mog_pnl.bg_ratio.GetValue()
        var_thres = self.mog_pnl.var_thres.GetValue()
        history = self.mog_pnl.history.GetValue()
        self.rosbag_process.set_mog2_parameters(gmm_num, bg_ratio, var_thres,
                                                history)
        rbox_sel = self.rbox.GetSelection()
        self.data = self.rosbag_process.run(dialog=dlg, low_ram=1 - rbox_sel,
                                               save_res=rbox_sel)
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
        if not found:
            raise Exception('invalid farm_key given (default is \'depth\')')
        self.height, self.width = self.data[
            self.farm_key].frames[0].shape[:2]
        self.frames_len = len(self.data[self.farm_key].frames)
        self.nb = TopicsNotebook(self, self.data,
                                 start_frame_handler=self.slider_min,
                                 end_frame_handler=self.slider_max,
                                 forced_frame_handler=self.slider_min)
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
        self.rgt_box = wx.BoxSizer(wx.VERTICAL)
        self.rgt_box.AddMany([(self.nb, 1, wx.EXPAND),
                              (self.ref_sizer, 1, wx.SHAPED | wx.EXPAND)])
        self.main_box.Add(self.rgt_box, 1)
        self.SetSizerAndFit(self.framesizer)
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


pathname = '/media/vassilis/Thesis/Datasets/PersonalFarm/TrainingActions/all.bag'


def main():
    '''
    main function
    '''
    FORMAT = '%(funcName)20s(%(lineno)s)-%(levelname)s:%(message)s'
    logging.basicConfig(format=FORMAT)
    app = wx.App(0)
    frame = MainFrame(None, -1, 'Data Mining')
    frame.Show(True)
    app.MainLoop()

if __name__ == '__main__':
    main()
