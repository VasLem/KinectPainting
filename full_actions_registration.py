import os
import logging
import warnings
import errno
import wx
from skimage import io, img_as_uint
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import class_objects as co
import moving_object_detection_alg as moda
import hand_segmentation_alg as hsa
import extract_and_process_rosbag as epr
import cPickle as pickle
CURR_DIR = os.getcwd()
ACTIONS_SAVE_PATH = os.path.join(CURR_DIR, 'actions')
LOG = logging.getLogger('__name__')

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
        self.rosbag_process = epr.DataProcess()
        self.rosbag_process.set_keys(self.farm_key, self.reg_key)
        self.save_path = co.CONST['actions_path']
        try:
            makedir(self.save_path)
        except (IOError, EOFError):
            LOG.warning('Invalid actions_path in ' +
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
        self.actions_name = []
        self.actions_start_ind = []
        self.actions_stop_ind = []
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
                        inp_items = line.rstrip('\n').split(':')
                        if '' in inp_items:
                            inp_items.remove('')
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

    def register_action(self, action_ind, inp, show_dlg=False, append=False):
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
            dialog = wx.ProgressDialog("Processing "+os.path.basename(inp)+
                                       '...',
                                       "Action: " +
                                       self.actions_name[action_ind],
                                       end - start + 1,
                                       style=0
                                       | wx.PD_APP_MODAL
                                       | wx.PD_CAN_ABORT
                                       | wx.PD_ESTIMATED_TIME
                                       | wx.PD_REMAINING_TIME)
        self.rosbag_process.reset()
        action_data = self.rosbag_process.process(inp,
                                                  farm_key=self.farm_key,
                                                  reg_key=self.reg_key,
                                                  start=start_ind, stop=stop_ind, low_ram=False,
                                                  dialog=dialog,
                                                  save_res=True,
                                                  save_path=os.path.join(self.save_path,
                                                                         self.actions_name[
                                                                            action_ind]),
                                                  append=append)
        if dialog is not None:
            dialog.Destroy()
            dialog.Close()

    def run(self, actions_list, inp, pathname=None, show_dlg=True, append=False):
        self.load_list_of_actions(actions_list)
        self.create_directories(pathname)
        for action_ind in range(len(self.actions_name)):
            self.register_action(action_ind, inp, show_dlg, append=append)
