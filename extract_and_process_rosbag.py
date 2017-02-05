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
import yaml
from skimage import io, exposure, img_as_uint, img_as_float
CURR_DIR = os.getcwd()
ROSBAG_WHOLE_RES_SAVE_PATH = os.path.join(CURR_DIR, 'whole_result')
START_COUNT = 0  # 300
STOP_COUNT = 0  # 600
def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
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
        self.save_path_exists = False
        self.save_path = co.CONST['rosbag_res_save_path']
        self.str_len = 0
        try:
            makedir(self.save_path)
        except:
            logging.warning('rosbag_res_save_path is invalid, ' +
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

    def run(self, bag_path=co.CONST['train_bag_path'], filt=True,
            farm_key='depth', reg_key='hand', save=False, load=False,
            pathname='extracted_bag', low_ram=True,
            detect_hand=True, dialog=None,
            save_res=False):
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
            bagdata = self.process(bag_path, farm_key,
                                   reg_key, low_ram,
                                   detect_hand,
                                   dialog,
                                   save_res=save_res)
        if save:
            with open(pathname, 'w') as out:
                pickle.dump(bagdata, out)
        return bagdata

    def process(self, bag_path=co.CONST['train_bag_path'], farm_key='depth',
                reg_key='hand',
                low_ram=True,
                detect_hand=True, dialog=None, start=str(START_COUNT),
                stop=str(STOP_COUNT), save_res=False):
        bagdata = {}
        bridge = CvBridge()
        prev_topic = ''
        sync_count = 0
        count = 0
        # Gather first num(=30) frames and find an initial background
        num = 30
        initial_im_set = []
        nnz_img = None
        info_dict = yaml.load(rosbag.Bag(bag_path)._get_yaml_info())
        self.str_len = 0
        for topic in info_dict['topics']:
            self.str_len = max(topic['messages'], self.str_len)
        self.str_len = len(str(self.str_len))
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
        cv2.imwrite('untrusty_pixels.png', untrusty_pixels.astype(np.uint16))
        cv2.imwrite('background_datamining.png',
                    initial_background.astype(np.uint16))
        count = 0
        self.mog2.reset()
        if isinstance(start, basestring):
            if not isinstance(stop, basestring):
                raise Exception('start and stop should be both strings or not')
            start_inds = sorted([int(item) for item in start.rstrip().split(',')])
            stop_inds = sorted([int(item) for item in stop.rstrip().split(',')])
            if len(start_inds) != len(stop_inds):
                raise Exception('start and stop should have equal length')

        else:
            start_inds = [start]
            stop_inds = [stop]
        seg_count = 0

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
            # from here on cv_image coresponds to farm_key frame
            try:
                if topic == prev_topic or bagdata[
                        topic].sync[-1] == sync_count:
                    sync_count += 1
            except (KeyError, IndexError):
                if topic == prev_topic:
                    sync_count += 1
            prev_topic = topic
            if count < start_inds[seg_count]:
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
            check = ( count >= start_inds[seg_count] and count <=
                      stop_inds[seg_count] if stop_inds[0] !=0 else 1)
            if check:
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
                    if not isinstance(copy[0, 0], np.uint8) or np.max(
                            copy) == 1:
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
                bagdata[topic].timestamps.append(timestamp)
                bagdata[topic].sync.append(sync_count)
                bagdata[topic].info.append(seg_count)
                if detect_hand:
                    if self.farm_key in topic:
                        self.register_hand(bagdata, topic, single=True,
                                           frame=cv_image,
                                           frame_sync=bagdata[topic].sync[-1],
                                           low_ram=low_ram, save_res=save_res)
                if dialog is not None:
                    wx.Yield()
                    keepGoing, _ = dialog.Update(count - min(start_inds) )
                    if not keepGoing:
                        break
                if count == stop_inds[seg_count] and stop_inds[seg_count] != 0:
                    if seg_count == len(start_inds) - 1:
                        return bagdata
                    else:
                        seg_count += 1
            count += 1
        return bagdata

    def set_keys(self, farm_key, reg_key):
        self.reg_key = reg_key
        self.farm_key = farm_key

    def register_hand(self, bagdata, farm_key=None, reg_key=None, overwrite=False,
                      rename_key=True, single=False, frame=None,
                      frame_sync=None, dialog=None,
                      low_ram=False, save_res=False):
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
        str_len = len(str(len(frames)))
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
                               (frame % 256).astype(np.uint8))
                    else:
                        res = ((mask * mog2_res) > 0).astype(np.uint8)
                        res = res * frame
                        # res = (self.skeleton.draw_skeleton(
                        #    res, show=False)).astype(np.uint8)
                    if not save_res:
                        try:
                            bagdata[reg_key].frames.append(
                                (res))
                        except (AttributeError, KeyError):
                            bagdata[reg_key] = RosbagStruct(reg_key)
                            bagdata[reg_key].frames.append(res)
                        bagdata[reg_key].info.append(
                            self.skeleton.skeleton_widths)
                        bagdata[reg_key].sync.append(frame_sync)
                    else:
                        io.imsave(os.path.join(
                            self.save_path, str(frame_sync).zfill(self.str_len)) + '.png',
                            res.astype(np.uint16))
                        with open(os.path.join(self.save_path,
                                               str(frame_sync).zfill(
                                                   self.str_len)
                                               + '.txt'), 'w') as out:
                            for item in self.skeleton.skeleton_widths:
                                out.write("%s\n" %
                                          np.array_str(np.array(item)))

            if dialog is not None:
                wx.Yield()
                keepGoing, _ = dialog.Update(self.mog2.frame_count)
                if not keepGoing:
                    dialog.Destroy()
                    return wx.ID_CANCEL
