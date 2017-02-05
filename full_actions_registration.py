import os
import logging
import warnings
import errno
import wx
from skimage import io, img_as_uint
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import class_objects as co
import moving_object_detection_alg as moda
import hand_segmentation_alg as hsa
import extract_and_process_rosbag as epr
CURR_DIR = os.getcwd()
ACTIONS_SAVE_PATH = os.path.join(CURR_DIR, 'actions')
def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
class ActionFarmingProcedure(object):

    def __init__(self, farm_key, reg_key):
        self.actions_name = []
        self.actions_start_ind = []
        self.actions_stop_ind = []
        self.actions_path = []
        self.processed_rosbag = None
        self.farm_key = farm_key
        self.reg_key = reg_key
        self.rosbag_process = epr.RosbagProcess()
        self.rosbag_process.set_keys(self.farm_key, self.reg_key)
        self.save_path = co.CONST['actions_save_path']
        try:
            makedir(self.save_path)
        except (IOError, EOFError):
            logging.warning('Invalid actions_save_path in ' +
                            'config.yaml, using' + ACTIONS_SAVE_PATH)
            self.save_path = ACTIONS_SAVE_PATH

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
                        self.actions_start_ind.append(item)
                    elif j == 2:
                        self.actions_stop_ind.append(item)
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
                                self.actions_start_ind.append(item)
                            elif j == 2:
                                self.actions_stop_ind.append(item)
            except (IOError, EOFError) as e:
                print e
                raise Exception(str(actionslist) + ' is an invalid pathname')
        else:
            raise Exception(self.load_list_of_actions.__doc__)

    def create_directories(self, pathname=None):
        if pathname is None:
            pathname = self.save_path
        for action, start in zip(self.actions_name, self.actions_start_ind):
            self.actions_path.append(os.path.join(pathname, action))
            makedir(self.actions_path[-1])
            for count in range(len(start.split(','))):
                makedir(os.path.join(self.actions_path[-1], str(count)))

    def register_action(self, action_ind, show_dlg=False):
        '''
        If action_ind string of indexes is given, beware that the folders
        created will not be automatically deleted before their refilling, so
        files from previous execution will remain undeleted.
        '''
        start_ind = self.actions_start_ind[action_ind]
        stop_ind = self.actions_stop_ind[action_ind]
        start = min([int(ind) for ind in start_ind.split(',')])
        end = max([int(ind) for ind in stop_ind.split(',')])
        dialog = None
        if show_dlg:
            dialog = wx.ProgressDialog("Action: " +
                                       self.actions_name[action_ind],
                                       "Processing...",
                                       end - start + 1,
                                       style=0
                                       | wx.PD_APP_MODAL
                                       | wx.PD_CAN_ABORT
                                       | wx.PD_ESTIMATED_TIME
                                       | wx.PD_REMAINING_TIME)
        action_data = self.rosbag_process.process(farm_key=self.farm_key,
                                                  reg_key=self.reg_key,
                                                  start=start_ind, stop=stop_ind, low_ram=False,
                                                  dialog=dialog)
        if self.farm_key not in action_data:
            for key in action_data:
                if self.farm_key in key:
                    self.farm_key = key
                    break
        if self.farm_key not in action_data:
            raise Exception("invalid farm_key given")
        if dialog is not None:
            dialog.Update(0, 'Saving frames to: ' +
                          self.actions_path[action_ind])
        count = 0
        [os.remove(os.path.join(self.actions_path[action_ind], f)) for f
         in os.listdir(self.actions_path[action_ind]) if f.endswith('.png') or
         f.endswith('.txt')]
        str_len = self.rosbag_process.str_len
        for frame in action_data[self.reg_key].frames:
            info = action_data[self.farm_key].info[count]
            filename = str(action_data[self.reg_key].sync[count]
                           ).zfill(str_len)
            if isinstance(info, int):
                filename = os.path.join(str(info), filename)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                io.imsave(os.path.join(self.actions_path[action_ind],
                                       filename + '.png'), img_as_uint(frame))
            with open(os.path.join(self.actions_path[action_ind],
                                   filename + '.txt'), 'w') as out:
                for item in action_data[self.reg_key].info[count]:
                    out.write("%s\n" % np.array_str(np.array(item)))
            if dialog is not None:
                wx.Yield()
                keep_going, _ = dialog.Update(count)
                if not keep_going:
                    dialog.Destroy()
                    dialog.Close()
                    return
            count = count + 1
        if dialog is not None:
            dialog.Destroy()
            dialog.Close()

    def run(self, actions_list, pathname=None, show_dlg=True):
        self.load_list_of_actions(actions_list)
        self.create_directories(pathname)
        for action_ind in range(len(self.actions_name)):
            self.register_action(action_ind, show_dlg)
