'''
One-shot dataset farmer, using single rosbag file
'''
# pylint: disable=unused-argument,too-many-ancestors,too-many-instance-attributes
# pylint: disable=too-many-arguments
import logging
FORMAT = '%(funcName)20s(%(lineno)s)-%(levelname)s:%(message)s'
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
import errno
import wx.lib.mixins.listctrl as listmix
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
ID_ACT_SAVE = wx.NewId()
bagdata = {}
START_COUNT = 0  # 300
STOP_COUNT = 0  # 600


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


class ActionFarmingProcedure(object):

    def __init__(self, farm_key, reg_key, bag_path=None):
        self.actions_name = []
        self.actions_start_ind = []
        self.actions_stop_ind = []
        self.actions_path = []
        self.processed_rosbag = None
        self.farm_key = farm_key
        self.reg_key = reg_key
        self.rosbag_process = RosbagProcess()
        self.rosbag_process.set_keys(self.farm_key, self.reg_key)
        self.bag_path = co.CONST['train_bag_path']
        self.save_path = co.CONST['actions_save_path']

    def reset(self):
        self.actions_path = []
        self.actions_name = []
        self.actions_stop_ind = []
        self.actions_start_ind = []

    def set_pathnames(self, rosbag_path=None, save_path=None):
        if save_path is not None:
            self.save_path = save_path
        if rosbag_path is not None:
            self.rosbag_path = rosbag_path

    def set_keys(self, farm_key=None, reg_key=None):
        self.farm_key = farm_key if farm_key is not None else self.farm_key
        self.reg_key = reg_key if reg_key is not None else self.farm_key

    def load_list_of_actions(self, actionslist):
        '''
        actionslist: string pathname or wx.ListCtrl object
        '''
        global bagdata
        if bagdata is None:
            raise Exception('bagdata is None')
        if self.farm_key is None or self.reg_key is None:
            raise Exception('run set_keys first')
        if isinstance(actionslist, wx.ListCtrl):
            for i in range(actionslist.GetItemCount()):
                for j in range(actionslist.GetColumnCount()):
                    item = actionslist.GetItem(i, j).GetText()
                    if item is None:
                        wx.MessageBox('List is not full, please fix it', 'Error',
                                      wx.OK | wx.ICON_ERROR)
                    if j == 0:
                        self.actions_name.append(item)
                    elif j == 1:
                        self.actions_start_ind.append(int(item))
                    elif j == 2:
                        self.actions_stop_ind.append(int(item))
        elif isinstance(actionslist, str):
            try:
                with open(actionslist, 'r') as inp:
                    for i, line in enumerate(inp):
                        inp_items = line.split(',')
                        if '\n' in inp_items:
                            inp_items.remove('\n')
                        if len(inp_items) != 3:
                            wx.MessageBox('List in file is wrong, please fix it', 'Error',
                                          wx.OK | wx.ICON_ERROR)
                        for j, item in enumerate(inp_items):
                            if j == 0:
                                self.actions_name.append(item)
                            elif j == 1:
                                self.actions_start_ind.append(int(j))
                            elif j == 2:
                                self.actions_stop_ind.append(int(j))
            except (IOError, EOFError) as e:
                print e
                raise Exception(str(actionslist) + ' is an invalid pathname')
        else:
            raise Exception(self.load_list_of_actions.__doc__)

    def create_directories(self, pathname=None):
        if pathname is None:
            pathname = co.CONST['actions_save_path']
        for action in self.actions_name:
            self.actions_path.append(os.path.join(pathname, action))
            makedir(self.actions_path[-1])

    def register_action(self, action_ind, show_dlg=False):
        start_ind = self.actions_start_ind[action_ind] - 5
        stop_ind = self.actions_stop_ind[action_ind]
        dialog = None
        if show_dlg:
            dialog = wx.ProgressDialog("Action: " +
                                       self.actions_name[action_ind],
                                       "Processing...",
                                       stop_ind - start_ind + 1,
                                       style=0
                                       | wx.PD_APP_MODAL
                                       | wx.PD_CAN_ABORT
                                       | wx.PD_ESTIMATED_TIME
                                       | wx.PD_REMAINING_TIME)
        action_data = self.rosbag_process.process(farm_key=self.farm_key,
                                                  reg_key=self.reg_key,
                                                  start=start_ind, stop=stop_ind, low_ram=False,
                                                  dialog=dialog)
        if dialog is not None:
            dialog.Update(0, 'Saving frames to: ' +
                          self.actions_path[action_ind])
        # DEBUGGING
        '''
        for frame in action_data[self.reg_key].frames:
            cv2.imshow('tmp',frame)
            cv2.waitKey(30)
        '''
        count = 0
        [os.remove(os.path.join(self.actions_path[action_ind], f)) for f
         in os.listdir(self.actions_path[action_ind]) if f.endswith('.png') or
         f.endswith('.txt')]

        for frame in action_data[self.reg_key].frames:
            cv2.imwrite(os.path.join(self.actions_path[action_ind], str(count) + '.png'),
                        frame.astype(np.float32), (cv2.IMWRITE_PNG_COMPRESSION, 9))
            with open(os.path.join(self.actions_path[action_ind],
                                   str(count) + '.txt'), 'w') as out:
                for item in action_data[self.reg_key].info[count]:
                    out.write("%s\n" % np.array_str(np.array(item)))
            count = count + 1
            if dialog is not None:
                wx.Yield()
                try:
                    keepGoing, _ = dialog.Update(count - 1)
                except wx._core.wxAssertionError:
                    dialog.Destroy()
                    dialog.Close()
                    dialog = None
            if not keepGoing and dialog is not None:
                dialog.Destroy()
                dialog.Close()
                return
        if dialog is not None:
            dialog.Destroy()
            dialog.Close()

    def run(self, actions_list, pathname=None, show_dlg=True):
        self.load_list_of_actions(actions_list)
        self.create_directories(pathname)
        for action_ind in range(len(self.actions_name)):
            self.register_action(action_ind, show_dlg)


class EditableListCtrl(wx.ListCtrl, listmix.TextEditMixin):

    def __init__(self, parent, ID=wx.ID_ANY, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, style=0):
        wx.ListCtrl.__init__(self, parent, ID, pos, size, style)
        listmix.TextEditMixin.__init__(self)


class MainFrame(wx.Frame):
    '''
    Main Processing Window
    '''

    def __init__(self, parent, id_, title, farm_key='depth', reg_key='hand'):
        wx.Frame.__init__(self, parent, id_, title)
        global bagdata
        self.farm_key = farm_key
        self.reg_key = reg_key
        if bagdata:
            found = 0
            for key in bagdata.keys():
                if self.farm_key in key:
                    self.farm_key = key
                    found = 1
                    break
            if not found:
                raise Exception(
                    'invalid key_string given (default is \'depth\')')
            self.height = bagdata[self.farm_key].frames[0].shape[0]
            self.width = bagdata[self.farm_key].frames[0].shape[1]
            self.frames_len = len(bagdata[self.farm_key].frames)
        else:
            self.height = 0
            self.width = 0
            self.frames_len = 1
        self.actionfarming = ActionFarmingProcedure(
            self.farm_key, self.reg_key)
        self.rosbag_process = RosbagProcess()
        self.rosbag_process.set_keys(self.farm_key, self.reg_key)
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
        self.hnd_pnl = wx.Panel(self.main_panel, -1, style=wx.SIMPLE_BORDER)
        self.hnd_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.mog_pnl = MOG2params(self, self.main_panel)
        self.recompute_mog2 = wx.Button(
            self.main_panel, ID_MASK_RECOMPUTE, "Calculate hand masks")
        self.Bind(wx.EVT_BUTTON, self.on_recompute_mask, id=ID_MASK_RECOMPUTE)

        self.max_pnl.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)

        self.load = wx.Button(self.inter_pnl, ID_LOAD, 'Load csv')
        self.loadbag = wx.Button(self.inter_pnl, ID_LOADBAG, 'Load bag')
        self.process_bag = wx.Button(
            self.inter_pnl, ID_BAG_PROCESS, 'Process bag')
        self.play = wx.Button(self.inter_pnl, ID_PLAY, 'Play')
        self.stop = wx.Button(self.inter_pnl, ID_STOP, 'Stop')
        self.add = wx.Button(self.inter_pnl, ID_ADD, 'Add')
        self.save = wx.Button(self.inter_pnl, ID_SAVE, 'Save csv')
        self.act_save = wx.Button(self.inter_pnl, ID_ACT_SAVE, 'Save actions')
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
        self.lst = EditableListCtrl(self.rep_pnl, -1, style=wx.LC_REPORT,
                                    size=(-1, 200))
        self.lst.InsertColumn(0, 'Action Name')
        self.lst.InsertColumn(1, 'Starting Index')
        self.lst.InsertColumn(2, 'Ending Index')
        self.lst.SetColumnWidth(0, 200)
        self.lst.SetColumnWidth(1, 200)
        self.lst.SetColumnWidth(2, 200)
        # DEBUGGING
        if os.path.exists('actions.csv'):
            with open('actions.csv', 'r') as inp:
                for i, line in enumerate(inp):
                    item_num = self.lst.GetItemCount()
                    inp_items = line.split(',')
                    if '\n' in inp_items:
                        inp_items.remove('\n')
                    for j, item in enumerate(inp_items):
                        if j == 0:
                            self.lst.InsertItem(item_num, str(item))
                        else:
                            self.lst.SetItem(item_num, j, item)
        lst_box = wx.BoxSizer(wx.VERTICAL)
        lst_box.Add(self.lst, 1, wx.VERTICAL)
        self.rep_pnl.SetSizer(lst_box)
        load_box = wx.BoxSizer(wx.HORIZONTAL)
        load_box.AddMany([(self.load, 1),
                          (self.loadbag, 1),
                          (self.process_bag, 1)])
        but_box = wx.BoxSizer(wx.HORIZONTAL)
        but_box.AddMany([(self.play, 1), (self.stop, 1),
                         (self.add, 1), (self.txt_inp, 1),
                         (self.remove, 1), (self.save, 1)])
        mis_box = wx.BoxSizer(wx.HORIZONTAL)
        mis_box.Add(self.slider_min, 1)
        mas_box = wx.BoxSizer(wx.HORIZONTAL)
        mas_box.Add(self.slider_max, 1)
        inter_box = wx.BoxSizer(wx.VERTICAL)
        inter_box.AddMany([(load_box, 1),
                           (but_box, 1),
                           (mis_box, 1),
                           (mas_box, 1)])
        self.inter_pnl.SetSizer(inter_box)
        self.lft_box = wx.BoxSizer(wx.VERTICAL)
        self.lft_box.AddMany([(self.inter_pnl, 1), (self.rep_pnl, 1)])
        if self.mog_pnl is not None:
            self.lft_box.AddMany([(self.mog_pnl, 1),
                                  (self.recompute_mog2, 1)])
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
        self.Bind(wx.EVT_BUTTON, self.on_actions_save, id=ID_ACT_SAVE)
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

    def define_panels_params(self):
        self.vid_pnl.SetMinSize(wx.Size(self.width, self.height))
        self.min_pnl.SetMinSize(wx.Size(self.width / 4, self.height / 4))
        self.max_pnl.SetMinSize(wx.Size(self.width / 4, self.height / 4))
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
        # self.hnd_pnl.Bind(wx.EVT_PAINT,
        #               lambda event: self.on_hand_image(event, self.hnd_pnl))
        self.hnd_pnl.SetMinSize(wx.Size(self.width / 4, self.height / 4))
        self.Layout()
        self.Refresh()

    def on_hand_image(self, event, main_panel):
        painter = wx.AutoBufferedPaintDC(self.hnd_pnl)
        painter.Clear()
        try:
            painter.DrawBitmap(getbitmap(main_panel,
                                         bagdata[self.reg_key].
                                         frames[self.exists_hand_count]), 0, 0)
        except TypeError:
            self.hnd_pnl.SetBackgroundColour(wx.BLACK)

    def on_recompute_mask(self, event):
        global bagdata
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
        co.chhm.print_stats()
        return 1

    def on_actions_save(self, event):
        self.actionfarming.run(self.lst)

    def on_frame_change(self, event, slider, main_panel):
        '''
        change frame when slider value changes
        '''
        painter = wx.AutoBufferedPaintDC(main_panel)
        painter.Clear()
        painter.DrawBitmap(getbitmap(main_panel, bagdata[
            self.farm_key].frames[slider.GetValue()]), 0, 0)

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
                try:
                    self.exists_hand_count = bagdata[
                        self.reg_key].sync.index(bagdata[self.farm_key].sync[self.count])
                except (ValueError, KeyError):
                    self.exists_hand_count = None
                if not self.exists_hand_count:
                    painter.DrawBitmap(getbitmap(main_panel, bagdata[self.farm_key].
                                                 frames[self.count]), 0, 0)
                else:
                    painter.DrawBitmap(getbitmap(main_panel, bagdata['hand'].
                                                 frames[self.exists_hand_count]), 0, 0)

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
                for i, line in enumerate(inp):
                    item_num = self.lst.GetItemCount()
                    inp_items = line.split(',')
                    if '\n' in inp_items:
                        inp_items.remove('\n')
                    for j, item in enumerate(inp_items):
                        if j == 0:
                            self.lst.InsertItem(item_num, str(item))
                        else:
                            self.lst.SetItem(item_num, j, item)
        elif result == wx.ID_CANCEL:
            return

    def on_load_rosbag(self, event):
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
            res = self.on_load_rosbag(None)
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
        self.rosbag_process.run(dialog=dlg)
        lengths = [len(bagdata[key].frames) for key in bagdata.keys()]
        baglength = max(lengths)
        dlg.Destroy()
        dlg.Close()
        self.slider_min.SetMax(baglength - 1)
        self.slider_max.SetMax(baglength - 1)
        self.slider_min.SetValue(START_COUNT)
        self.slider_max.SetValue(baglength - 1)
        found = 0
        if self.farm_key not in bagdata.keys():
            for key in bagdata.keys():
                if self.farm_key in key:
                    self.farm_key = key
                    found = 1
                    break
        if not found:
            raise Exception('invalid key_string given (default is \'depth\')')
        self.height, self.width = bagdata[self.farm_key].frames[0].shape[:2]
        self.frames_len = len(bagdata[self.farm_key].frames)
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

LO_THRES = 650
HI_THRES = 700


class RosbagProcess(object):

    def __init__(self):
        self.hands_memory = co.Memory()
        self.mog2 = moda.Mog2()
        self.skeleton = None
        self.hands_mask = None
        self.history = co.CONST
        self.gmm_num = None
        self.bg_ratio = None
        self.var_thres = None
        self.gmm_num = co.CONST['gmm_num']
        self.bg_ratio = co.CONST['bg_ratio']
        self.var_thres = co.CONST['var_thres']
        self.history = co.CONST['history']
        self.reg_key = None
        self.farm_key = None
        self.prev_frame = None

    def set_mog2_parameters(self, gmm_num=co.CONST['gmm_num'],
                            bg_ratio=co.CONST['bg_ratio'],
                            var_thres=co.CONST['var_thres'],
                            history=co.CONST['history']):
        self.gmm_num = gmm_num
        self.bg_ratio = bg_ratio
        self.var_thres = var_thres
        self.history = history

    def run(self, bag_path=co.CONST['train_bag_path'], filt=True,
            farm_key='depth', reg_key='hand', save=False, load=False,
            pathname='extracted_bag', low_ram=True,
            detect_hand=True, dialog=None):
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
        loaded = 0
        if load:
            try:
                with open(pathname, 'r') as inp:
                    bagdata = pickle.load(inp)
                loaded = 1
            except (IOError, EOFError):
                print 'No file available, repeating process'
        if not loaded:
            bagdata = self.process(bag_path, farm_key,
                                   reg_key, low_ram,
                                   detect_hand, dialog)
        if save:
            with open(pathname, 'w') as out:
                pickle.dump(bagdata, out)
        return None

    def process(self, bag_path=co.CONST['train_bag_path'], farm_key='depth',
                reg_key='hand',
                low_ram=True,
                detect_hand=True, dialog=None, start=START_COUNT, stop=STOP_COUNT):
        bagdata = {}
        bridge = CvBridge()
        prev_topic = ''
        sync_count = 0
        count = 0
        # Gather first num(=30) frames and find an initial background
        num = 30
        initial_im_set = []
        nnz_img = None
        for topic, msg, timestamp in rosbag.Bag(bag_path).read_messages():
                # Reading image Stage
            if farm_key in topic:
                try:
                    img = bridge.imgmsg_to_cv2(
                        msg, 'passthrough')
                except CvBridgeError as err:
                    print err
                initial_im_set.append(img)
                count += 1
                if nnz_img is None:
                    nnz_img = np.zeros_like(img)
                flag = img > 0
                nnz_img[flag] = img[flag]
                if count == num:
                    break
        initial_background = np.median(np.array(initial_im_set), axis=0)
        untrusty_pixels = ((initial_background <=
                            nnz_img - 10) + (initial_background >= nnz_img +
                                             10)).astype(bool)
        cv2.imwrite('untrusty_pixels.png', untrusty_pixels.astype(np.float32))
        cv2.imwrite('background_datamining.png', initial_background)
        count = 0
        self.mog2.reset()
        for topic, msg, timestamp in rosbag.Bag(bag_path).read_messages():
                # Reading image Stage
            if farm_key in topic:
                try:
                    cv_image = bridge.imgmsg_to_cv2(
                        msg, 'passthrough')
                except CvBridgeError as err:
                    print err
            else:
                continue
            filtered_cv_image = cv_image.copy()
            filtered_cv_image[0]
            cv_image = filtered_cv_image.copy()
            # from here on cv_image coresponds to farm_key frame
            if count < start:
                if count < 30:
                    # force background initialization from early frames
                    # Replacing most untrusty pixels with local information
                    for _ in range(2):
                        median_filtered_cv_image = cv2.medianBlur(
                            filtered_cv_image, 5)
                        filtered_cv_image[untrusty_pixels] = median_filtered_cv_image[
                            untrusty_pixels]
                    cv_image = filtered_cv_image
                    if count == 0:
                        self.mog2.initialize(3,
                                             self.bg_ratio,
                                             self.var_thres,
                                             30)
                    self.mog2.fgbg.apply(cv_image.astype(np.float32))
            if count >= start:
                # Saving stage. Images are saved as grayscale and uint8 for
                # less RAM
                # Replacing most untrusty pixels with local information
                for _ in range(2):
                    median_filtered_cv_image = cv2.medianBlur(
                        filtered_cv_image, 5)
                    filtered_cv_image[untrusty_pixels] = median_filtered_cv_image[
                        untrusty_pixels]
                cv_image = filtered_cv_image
                if low_ram:
                    copy = cv_image.copy()
                    if len(copy.shape) == 3:
                        copy = np.mean(copy, axis=2)
                    if not isinstance(copy[0, 0], np.uint8) or np.max(copy) == 1:
                        if np.max(copy) > 256:
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
                    if self.farm_key in topic:
                        self.register_hand(bagdata, topic, single=True,
                                           frame=cv_image,
                                           frame_sync=bagdata[topic].sync[-1],
                                           low_ram=low_ram)
                if dialog is not None:
                    wx.Yield()
                    keepGoing, _ = dialog.Update(count - start)
                    if not keepGoing:
                        break
            if stop != 0:
                if count == stop:
                    break
            count += 1
        return bagdata

    def set_keys(self, farm_key, reg_key):
        self.reg_key = reg_key
        self.farm_key = farm_key

    def register_hand(self, bagdata, farm_key=None, reg_key=None, overwrite=False,
                      rename_key=True, single=False, frame=None,
                      frame_sync=None, dialog=None, low_ram=False):
        '''
        farm_key : from which topic to get data
        reg_key : what name will have the topic to register hand data
        overwrite is True if overwrite is allowed, default is False
        rename_key is True if renaming key is allowed in case overwrite is False
        if single is True:
            frame: frame from which hand is registered
            frame_sync: frame sync to be added to hand dictionary sync entry
        '''
        if farm_key is None:
            if self.farm_key is None:
                raise Exception('farm_key is required')
            farm_key = self.farm_key
        if reg_key is None:
            if self.reg_key is None:
                raise Exception('reg_key is required')
            reg_key = self.reg_key
        if not single:
            if not bagdata:
                raise Exception('run first')
            topic_name = None
            for topic in bagdata.keys():
                if farm_key in topic:
                    topic_name = topic
                    break
            if topic_name is None:
                raise Exception('Invalid farm_key given')
            else:
                farm_key = topic_name
            if reg_key in bagdata.keys():
                if overwrite:
                    logging.info('Overwriting previous hand data..')
                else:
                    count = 0
                    if rename_key:
                        logging.info('Renaming given key..')
                        while True:
                            if reg_key + str(count) not in bagdata.keys():
                                reg_key = reg_key + str(count)
                                break
                    else:
                        raise Exception('reg_key given already exists. \n' +
                                        self.register_hand.__doc__)
            frames = bagdata[farm_key].frames
            frames_sync = bagdata[farm_key].sync
            self.mog2.reset()
        else:
            frames = [frame]
            frames_sync = [frame_sync]
        for frame, frame_sync in zip(frames, frames_sync):
            if self.hands_memory is None:
                self.hands_memory = co.Memory()
            if co.edges.calib_edges is None:
                co.edges.load_calib_data(img=frame, whole_im=True)
            if self.skeleton is None:
                self.skeleton = hsa.FindArmSkeleton(frame.copy())
            inp = frame.copy()
            mog2_res = self.mog2.run(False,
                                     inp.astype(np.float32),
                                     gmm_num=self.gmm_num,
                                     bg_ratio=self.bg_ratio,
                                     var_thres=self.var_thres,
                                     history=self.history)
            if self.prev_frame is None:
                self.prev_frame = mog2_res
            mog2_res = (mog2_res == 127).astype(np.uint8)
            kernel = np.ones((5, 5), np.uint8)
            mask1 = cv2.morphologyEx(mog2_res.copy(), cv2.MORPH_OPEN, kernel)
            check_sum = np.sum(mask1 > 0)
            if check_sum > 0 and check_sum < np.sum(frame > 0):
                _, cnts, _ = cv2.findContours(mask1,
                                              cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_NONE)
                cnts_areas = [cv2.contourArea(cnts[i]) for i in
                              xrange(len(cnts))]
                mask = None
                try:
                    if self.skeleton.run(
                            frame, cnts[np.argmax(cnts_areas)], 'longest_ray'):
                        mask = self.skeleton.hand_mask
                except:
                    pass
                if mask is not None:
                    if low_ram and frame.max() > 256:
                        res = (((mask * mog2_res) > 0) *
                               (frame /
                                float(co.CONST['max_depth'])) *
                               256).astype(np.uint8)
                    else:
                        res = 255 * (mask * mog2_res).astype(np.uint8)
                        res = (self.skeleton.draw_skeleton(
                            res, show=False)).astype(np.uint8)
                        #((mask) > 0) * frame
                    try:
                        bagdata[reg_key].frames.append(
                            (res).astype(np.uint8))
                    except (AttributeError, KeyError):
                        bagdata[reg_key] = RosbagStruct(reg_key)
                        bagdata[reg_key].frames.append((res).astype(np.uint8))
                    bagdata[reg_key].info.append(
                        self.skeleton.skeleton_widths)
                    bagdata[reg_key].sync.append(frame_sync)
            if dialog is not None:
                wx.Yield()
                keepGoing, _ = dialog.Update(self.mog2.frame_count)
                if not keepGoing:
                    dialog.Destroy()
                    return wx.ID_CANCEL


pathname = '/media/vassilis/Thesis/Datasets/PersonalFarm/TrainingActions/all.bag'


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
