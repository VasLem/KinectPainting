import logging
import os
import warnings
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
import yaml
from math import pi
from skimage import io, exposure, img_as_uint, img_as_float
from copy import copy
CURR_DIR = os.getcwd()
ROSBAG_WHOLE_RES_SAVE_PATH = os.path.join(CURR_DIR, 'whole_result')
START_COUNT = 0  # 300
STOP_COUNT = 0  # 600
LOG = logging.getLogger(__name__)

def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class HandInfoStruct(object):
    def __init__(self):
        skeleton = None
        skeleton_widths = None

class DataStruct(object):
    '''
    class to hold extracted rosbag single topic
    '''

    def __init__(self, name=''):
        self.name = name
        self.frames = []
        self.sync = []
        self.timestamps = []
        self.info = []


class DataProcess(object):

    def __init__(self, save=True):
        self.mog2 = moda.Mog2()
        self.skeleton = None
        self.hands_mask = None
        self.gmm_num = None
        self.bg_ratio = None
        self.var_thres = None
        self.gmm_num = co.CONST['gmm_num']
        self.bg_ratio = co.CONST['bg_ratio']
        self.var_thres = co.CONST['var_thres']
        self.history = co.CONST['history']
        self.reg_key = None
        self.farm_key = None
        self.save_path_exists = False
        self.save_path = co.CONST['whole_save_path']
        self.skeletons = {}
        self.str_len = 0
        self.save = save
        self.data = {}
        self.initial_im_set = []
        self.img_count = 0
        self.sync_count = -1
        self.nnz_img = None
        self.initial_background = None
        self.untrusty_pixels = None
        self.prev_topic = None
        self.kernel = np.ones((5, 5), np.uint8)
        self.folder = None
        self.angle_vec = None
        self.center_vec = None
        self.data_append = False
        # Gather first num(=30) frames and find an initial background
        self.init_num = 30
        if save:
            try:
                makedir(self.save_path)
            except:
                LOG.warning('whole_save_path is invalid, ' +
                                ROSBAG_WHOLE_RES_SAVE_PATH +
                                ' will be used instead')
                self.save_path = ROSBAG_WHOLE_RES_SAVE_PATH

    def set_mog2_parameters(self, gmm_num=co.CONST['gmm_num'],
                            bg_ratio=co.CONST['bg_ratio'],
                            var_thres=co.CONST['var_thres'],
                            history=co.CONST['history']):
        self.gmm_num = gmm_num
        self.bg_ratio = bg_ratio
        self.var_thres = var_thres
        self.history = history

    def run(self, data=co.CONST['train_bag_path'], filt=True,
            farm_key='depth', reg_key='hand', save=False, load=False,
            pathname='extracted_bag', low_ram=True,
            detect_hand=True, dialog=None,
            save_res=False, derotate=False, append=False):
        '''
        Save rosbag frames to a dictionary according to the topic. Using as reference the
        newest topic with the closest timestamp , a synchronization vector is made. Each
        dictionary contains:
        ->a sublist, containing every captured frame
        ->the corresponding synchronization vector
        ->the corresponding ROS timestamp vector if given
        if color_filt, any topic containing string convstr  is converted to
        convtype
        if save, dictionary is saved to file pathname
        if load, dictionary is loaded from file pathname
        If append, do not remove previous samples
        '''
        self.folder = ''
        self.data = {}
        self.skeletons = {}
        self.sync_count = -1
        if save_res:
            if isinstance(data, basestring):
                self.folder = os.path.splitext(
                  os.path.normpath(
                      data).split(os.sep)[-1])[0]
            else:
                self.folder = 'processed'
            makedir(os.path.join(self.save_path,self.folder))
        loaded = 0
        if load:
            try:
                with open(pathname, 'r') as inp:
                    data = pickle.load(inp)
                loaded = 1
            except (IOError, EOFError):
                print 'No file available, repeating process'
        if not loaded:
            self.img_count = 0
            data = self.process(data, farm_key,
                                reg_key, low_ram,
                                detect_hand,
                                dialog,
                                save_res=save_res,
                                save_path = os.path.join(
                                    self.save_path, self.folder),
                                derotate=derotate,
                                append=append)
        if save:
            with open(pathname, 'w') as out:
                pickle.dump(data, out)
        return data

    def reset(self):
        self.data = {}
        self.skeletons = {}
        self.prev_topic = ''
        self.sync_count = -1
        self.seg_count = 0
        self.fold_count = 0
        self.angle_vec = []
        self.center_vec = []
        self.mog2.reset()
        self.mog2.initialize(3,
                             self.bg_ratio,
                             self.var_thres,
                             30)
        self.data = {}

    def process(self, inp=co.CONST['train_bag_path'], farm_key='depth',
                reg_key='hand',
                low_ram=True,
                detect_hand=True, dialog=None, start=str(START_COUNT),
                stop=str(STOP_COUNT), save_res=False, save_path=None, timestamp=0,
                derotate=False, append=False):
        '''
        If <inp> is a string, then it is the path of a rosbag file.
        If <inp> is a numpy array, then it is the data to be processed.
        A <DataStruct> dictionary is constructed.
        <farm_key> refers to the key of the dictionary where original data will
        be saved
        <reg_key> refers to the key of the dictionary where the processed data
        will be added.
        <low_ram> is true if the original and processed data should have their
        size reduced, by downsampling them
        <detect_hand> is True if the special method for hand detection is to be
        used
        <dialog> can contain a <wx.ProcessDialog>, where the progress will be
        shown (in case the input is a rosbag path)
        <start> and <stop> are flags used when the inp is a rosbag path, to
        show where to begin and stop rosbag reading and processing
        <save_res> is True if one wants to save the processed data locally as
        png images
        '''
        if save_path is not None:
            self.save_path = save_path
        if self.farm_key is None:
            self.set_keys(farm_key, reg_key)
        if isinstance(inp, basestring):
            bridge = CvBridge()
            iterat = rosbag.Bag(inp).read_messages()
            info_dict = yaml.load(rosbag.Bag(inp)._get_yaml_info())
            self.str_len = 0
            for topic in info_dict['topics']:
                self.str_len = max(topic['messages'], self.str_len)
            self.str_len = len(str(self.str_len))
        else:
            iterat = [(farm_key, inp, timestamp)]
            self.str_len = 6  # ok, a million pictures seem enough
        if isinstance(start, basestring):
            if not isinstance(stop, basestring):
                raise Exception('start and stop should be both strings or not')
            start_inds = sorted([int(item)
                                 for item in start.rstrip().split(',')])
            stop_inds = sorted([int(item)
                                for item in stop.rstrip().split(',')])
            if len(start_inds) != len(stop_inds):
                raise Exception('start and stop should have equal length')

        else:
            start_inds = [start]
            stop_inds = [stop]
        self.fold_count = 0
        for topic, msg, timestamp in iterat:
            if self.sync_count == -1:
                self.prev_topic = ''
                self.seg_count = 0
                if append and save_res:
                    try:
                        try:
                            self.fold_count = max([int(filter(unicode.isdigit,fil)) for fil in
                                              os.listdir(self.save_path) if
                                                   unicode.isdigit(fil) and
                                                   os.path.isdir(os.path.join(
                                                       self.save_path,fil)) and
                                                  len(os.listdir(os.path.join(
                                                       self.save_path,fil)))!=0])+1
                        except TypeError:
                            self.fold_count = max([int(filter(str.isdigit,fil)) for fil in
                                              os.listdir(self.save_path) if
                                                   str.isdigit(fil) and
                                                   os.path.isdir(os.path.join(
                                                       self.save_path,fil)) and
                                                  len(os.listdir(os.path.join(
                                                       self.save_path,fil)))!=0])+1
                    except ValueError:
                        self.fold_count = 0
                else:
                    self.fold_count = 0
                self.mog2.reset()
                self.mog2.initialize(3,
                                     self.bg_ratio,
                                     self.var_thres,
                                     30)
            if farm_key in topic:
                if isinstance(inp, basestring):
                    try:
                        cv_image = bridge.imgmsg_to_cv2(
                            msg, 'passthrough')
                    except CvBridgeError as err:
                        print err
                else:
                    cv_image = inp
                self.sync_count+=1
                if low_ram:
                    cop = cv_image.copy()
                    if len(cop.shape) == 3:
                        cop = np.mean(cop, axis=2)
                    if not isinstance(cop[0, 0], np.uint8) or np.max(
                            cop) == 1:
                        if np.max(cop) > 256:
                            cop = cop % 256
                        else:
                            cop = (cop / float(np.max(cop))) * 255
                    cop = cop.astype(np.uint8)
                else:
                    cop = cv_image
                try:
                    self.data[topic].frames.append(cop)
                except (AttributeError, KeyError):
                    self.data[topic] = DataStruct(topic)
                    self.data[topic].frames.append(cop)
                self.data[topic].timestamps.append(timestamp)
                self.data[topic].sync.append(self.sync_count)
                self.data[topic].info.append(self.seg_count)
            else:
                continue
            if self.sync_count < self.init_num:
                self.initial_im_set.append(cv_image)
                if self.nnz_img is None:
                    self.nnz_img = np.zeros_like(cv_image)
                flag = cv_image > 0
                self.nnz_img[flag] = cv_image[flag]
            elif self.sync_count == self.init_num:
                self.initial_background = np.median(
                    np.array(self.initial_im_set), axis=0)
                self.untrusty_pixels = ((self.initial_background <=
                                         self.nnz_img - 10) + (self.initial_background >=
                                                               self.nnz_img +
                                                               10)).astype(bool)
                # DEBUGGING
                '''
                cv2.imwrite('untrusty_pixels.png', self.untrusty_pixels.astype(np.uint16))
                cv2.imwrite('background_datamining.png',
                            self.initial_background.astype(np.uint16))
                '''
            check = (self.sync_count >= start_inds[self.seg_count] and
                     self.sync_count <=
                     stop_inds[self.seg_count] if stop_inds[0] != 0 else 1)
            init_check = self.sync_count < self.init_num
            # init_check = True : force background initialization from
            # early frames
            if save_res and self.sync_count == 0 and not append:
                for root, directories,filenames in os.walk(self.save_path):
                    for filename in filenames:
                        if filename.endswith('.png') or filename.endswith('.txt'):
                            os.remove(os.path.join(root,filename))
            else:
                if append:
                    self.append_data = True
            if save_res:
                    makedir(os.path.join(self.save_path,
                                         co.CONST['mv_obj_fold_name'],
                                 str(self.fold_count)))
                    makedir(os.path.join(self.save_path,
                                         co.CONST['hnd_mk_fold_name'],
                                 str(self.fold_count)))

            if check or init_check:
                filtered_cv_image = cv_image.copy()
                for _ in range(2):
                    # Replacing most untrusty pixels with local information
                    median_filtered_cv_image = cv2.medianBlur(
                        filtered_cv_image, 5)
                    filtered_cv_image[self.untrusty_pixels] = median_filtered_cv_image[
                        self.untrusty_pixels]
                cv_image = filtered_cv_image
                if init_check:
                    self.mog2.fgbg.apply(cv_image.astype(np.float32))
                    continue
                if detect_hand:
                    self.register_hand(topic, single=True,
                                       frame=cv_image,
                                       frame_sync=self.data[
                                           topic].sync[-1],
                                       low_ram=low_ram, save_res=save_res,
                                       derotate=derotate)
                if dialog is not None:
                    wx.Yield()
                    keep_going, _ = dialog.Update(
                        self.sync_count - min(start_inds))
                    if not keep_going:
                        break
                if self.sync_count == stop_inds[
                        self.seg_count] and stop_inds[self.seg_count] != 0:
                    if self.seg_count == len(start_inds) - 1:
                        return self.data
                    else:
                        self.seg_count += 1
                        self.fold_count += 1
                        if save_res and os.path.isdir(os.path.join(
                            self.save_path,co.CONST['mv_obj_fold_name'],
                            str(self.fold_count))):
                            [os.remove(os.path.join(
                                self.save_path,co.CONST['mv_obj_fold_name'],
                                str(self.fold_count),
                                f)) for f in os.listdir(
                                    os.path.join(self.save_path,
                                                 co.CONST['mv_obj_fold_name'],
                                                 str(self.fold_count)))
                             if f.endswith('.png') or
                             f.endswith('.txt')]
                            [os.remove(os.path.join(
                                self.save_path,co.CONST['hnd_mk_fold_name'],
                                str(self.fold_count),
                                f)) for f in os.listdir(
                                    os.path.join(self.save_path,
                                                 co.CONST['mv_obj_fold_name'],
                                                 str(self.fold_count)))
                             if f.endswith('.png') or
                             f.endswith('.txt')]
        return self.data

    def set_keys(self, farm_key, reg_key):
        self.reg_key = reg_key
        self.farm_key = farm_key

    def register_hand(self, farm_key=None, reg_key=None, overwrite=False,
                      rename_key=True, single=False, frame=None,
                      frame_sync=None, dialog=None,
                      low_ram=False, save_res=False,
                      derotate=False):
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
            if not self.data:
                raise Exception('run first')
            topic_name = None
            for topic in self.data.keys():
                if farm_key in topic:
                    topic_name = topic
                    break
            if topic_name is None:
                raise Exception('Invalid farm_key given')
            else:
                farm_key = topic_name
            if reg_key in self.data.keys():
                if overwrite:
                    LOG.info('Overwriting previous hand data..')
                else:
                    count = 0
                    if rename_key:
                        LOG.info('Renaming given key..')
                        while True:
                            if reg_key + str(count) not in self.data.keys():
                                reg_key = reg_key + str(count)
                                break
                    else:
                        raise Exception('reg_key given already exists. \n' +
                                        self.register_hand.__doc__)
            frames = self.data[farm_key].frames
            frames_sync = self.data[farm_key].sync
            self.mog2.reset()
        else:
            frames = [frame]
            frames_sync = [frame_sync]
        str_len = len(str(len(frames)))
        for frame, frame_sync in zip(frames, frames_sync):
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
            mask1 = cv2.morphologyEx(mog2_res.copy(), cv2.MORPH_OPEN, self.kernel)
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
                    res =(mog2_res > 0) * frame
                    # provide invariance to rotation of link
                    last_link = (self.skeleton.skeleton[-1][1] -
                                 self.skeleton.skeleton[-1][0])
                    angle = np.arctan2(
                        last_link[0], last_link[1])
                    if self.angle_vec is None:
                        self.angle_vec = []
                        self.center_vec = []
                    if len(self.angle_vec) >= 5:
                        self.angle_vec = self.angle_vec[1:] + [angle]
                        self.center_vec = (self.center_vec[1:] +
                                           [self.skeleton.hand_start])
                    else:
                        self.angle_vec.append(angle)
                        self.center_vec.append(self.skeleton.skeleton[-1][0])
                    if low_ram and frame.max() > 256:
                        res = (res % 256).astype(np.uint8)
                    # res = (self.skeleton.draw_skeleton(
                    #    res, show=False)).astype(np.uint8)
                    derotate_angle = np.mean(self.angle_vec)
                    derotate_center = np.mean(self.center_vec, axis=0)
                    if derotate:
                        res = co.pol_oper.derotate(res,
                                                   derotate_angle,
                                                   derotate_center)

                    #DEBUGGING
                    #cv2.imshow('test', (res%255).astype(np.uint8))
                    #cv2.waitKey(10)

                    if not save_res:
                        try:
                            self.data[reg_key].frames.append(
                                (res))
                        except (AttributeError, KeyError):
                            self.data[reg_key] = DataStruct(reg_key)
                            self.data[reg_key].frames.append(res)
                        self.data[reg_key].info.append([derotate_angle,
                                                        derotate_center])
                        '''
                        self.data[reg_key].info.append(HandInfoStruct())
                        self.data[reg_key].info[
                            -1].skeleton = self.skeleton.skeleton
                        self.data[reg_key].info[
                            -1].skeleton_widths = self.skeleton.skeleton_widths
                        '''
                        self.data[reg_key].sync.append(frame_sync)
                    else:
                        cv2.imwrite(os.path.join(
                            self.save_path,co.CONST['mv_obj_fold_name'],
                            str(self.fold_count),
                            str(frame_sync).zfill(self.str_len)) + '.png',res)
                        cv2.imwrite(os.path.join(
                            self.save_path,co.CONST['hnd_mk_fold_name'],
                            str(self.fold_count),
                            str(frame_sync).zfill(self.str_len)) + '.png',mask)

                        with open(os.path.join(self.save_path,
                                               co.CONST['mv_obj_fold_name'],
                                               str(self.fold_count),
                                               'angles.txt'), 'a') as out:
                            out.write("%f\n" % derotate_angle)
                        with open(os.path.join(self.save_path,
                                               co.CONST['mv_obj_fold_name'],
                                               str(self.fold_count),
                                               'centers.txt'), 'a') as out:
                            out.write("%f %f\n" % (derotate_center[0],
                                        derotate_center[1]))
                        try:
                            self.skeletons[os.path.join(
                                self.save_path,co.CONST['mv_obj_fold_name'],
                                str(self.fold_count))].append(
                                    [self.skeleton.skeleton,
                                     self.skeleton.skeleton_widths])
                        except KeyError:
                            self.skeletons[os.path.join(
                                self.save_path,co.CONST['mv_obj_fold_name'],
                                str(self.fold_count))] = [
                                    [self.skeleton.skeleton,
                                     self.skeleton.skeleton_widths]]

            if dialog is not None:
                wx.Yield()
                keepGoing, _ = dialog.Update(self.img_count)
                if not keepGoing:
                    dialog.Destroy()
                    return wx.ID_CANCEL

