import numpy as np
import cv2
from math import pi
import class_objects as co

def find_nonzero(arr):
    return np.fliplr(cv2.findNonZero(arr).squeeze())

#Kinect Intrinsics
'''
PRIM_X = 479.75
PRIM_Y = 269.75
FLNT = 540.68603515625
'''
#Senz3d Intrinsics
PRIM_X =317.37514566554989
PRIM_Y=246.61273826510859
FLNT= 595.333159044648 
class SpaceHistogram(object):

    def __init__(self):
        self.binarized_space = []
        self.bin_size = 0

    def binarize_3d(self, constants):
        b_x = np.linspace(-1.0,1.0,self.bin_size)
        b_y = np.linspace(-1.0,1.0,self.bin_size)
        b_z = np.linspace(-1.0, 1.0,self.bin_size)
        self.binarized_space = [b_x, b_y, b_z]
    def binarize_1d(self, constants):
        self.binarized_space = [np.linspace(0.0, pi, self.bin_size)]
    def hist_data(self, sample):
        return np.histogramdd(sample, self.binarized_space, normed=True)[0].ravel()

class ActionRecognition(object):

    def __init__(self):
        self.prev_depth_im = np.zeros(0)
        self.curr_depth_im = np.zeros(0)
        self.unified_roi = np.zeros(0)
        self.prev_count = 0
        self.curr_count = 0
        self.features = FeatureExtraction()
        self.prev_patch = np.zeros(0)
        self.curr_patch = np.zeros(0)
        self.prev_patch_pos = np.zeros(0)
        self.curr_patch_pos = np.zeros(0)

    def extract_features(self, constants):
        self.features.find_roi(self.prev_patch, self.curr_patch,
                               self.prev_patch_pos, self.curr_patch_pos)
        roi=self.curr_depth_im[self.features.roi[0,0]:self.features.roi[0,1],
                               self.features.roi[1,0]:self.features.roi[1,1]]
        '''
        cv2.imshow('test2',roi/np.max(roi).astype(float))
        cv2.waitKey(0)
        '''
        hof_features = self.features.hof3d(
            self.prev_depth_im, self.curr_depth_im, constants)
        hog_features = self.features.ghog(self.curr_depth_im, constants)
        return hof_features,hog_features
    def update(self,hand_patch,hand_patch_pos):
        (self.prev_depth_im,
         self.curr_depth_im) = (self.curr_depth_im,
                                co.data.uint8_depth_im*co.meas.found_objects_mask)
        (self.curr_count,
         self.prev_count)=(co.counters.im_number,
                           self.curr_count)
        (self.prev_patch,
         self.curr_patch)=(self.curr_patch,
                                      hand_patch)
        (self.prev_patch_pos,
         self.curr_patch_pos)=(self.curr_patch_pos,
                                      hand_patch_pos)


class FeatureExtraction(object):

    def __init__(self):
        self.features = np.zeros(0)
        self.prev_projection = np.zeros(0)
        self.curr_projection = np.zeros(0)
        self.roi = np.zeros(0)

    def compute_scene_flow(self, prev_depth_im, curr_depth_im):
        prev_hand_patch = prev_depth_im[self.roi[0, 0]:self.roi[0, 1],
                                        self.roi[1, 0]:self.roi[1, 1]]
        curr_hand_patch = curr_depth_im[self.roi[0, 0]:self.roi[0, 1],
                                        self.roi[1, 0]:self.roi[1, 1]]
        nonzero_mask=prev_hand_patch+curr_hand_patch
        yx_coords = (find_nonzero(nonzero_mask.astype(np.uint8))-
                     np.array([[PRIM_Y,PRIM_X]]))
        prev_z_coords = prev_hand_patch[nonzero_mask > 0][:,None]/255.0
        curr_z_coords = curr_hand_patch[nonzero_mask > 0][:,None]/255.0
        dz_coords=curr_z_coords-prev_z_coords
        YX_coords=yx_coords * dz_coords / FLNT
        return np.concatenate((YX_coords,
                               dz_coords), axis=1)

    def find_roi(self, prev_patch, curr_patch, prev_patch_pos, curr_patch_pos):
        '''
        print 'roi'
        print '\t prev_patch shape',prev_patch.shape
        print '\t curr_patch.shape',curr_patch.shape
        '''
        self.roi = np.array([[
            min(prev_patch_pos[0], curr_patch_pos[0]),
            max((prev_patch.shape[0] + prev_patch_pos[0],
                    curr_patch.shape[0] + curr_patch_pos[0]))],
            [min(prev_patch_pos[1], curr_patch_pos[1]),
             max(prev_patch.shape[1] + prev_patch_pos[1],
                    curr_patch.shape[1] + curr_patch_pos[1])]])
        '''
        print '\t roi shape',self.roi[0,1]-self.roi[0,0],\
                self.roi[1,1]-self.roi[1,0]
        '''
    def hof3d(self, prev_depth_im, curr_depth_im, constants):
        if len(hofhist.binarized_space) == 0:
            hofhist.bin_size = 4
            hofhist.binarize_3d(constants)
        disp = self.compute_scene_flow(prev_depth_im,curr_depth_im)
        disp_norm = np.sqrt((disp[:, 0] * disp[:, 0] + disp[:, 1] *
                     disp[:, 1] + disp[:, 2] * disp[:, 2]))[:,None]
        disp_norm[disp_norm == 0] = 1
        disp /= disp_norm
        return hofhist.hist_data(disp)

    def grad_angles(self, patch):
        gradx, grady = np.gradient(patch)
        return np.arctan(grady, gradx)  # returns values 0 to pi

    def ghog(self, depth_im, constants):
        im_patch = depth_im[self.roi[0, 0]:self.roi[0, 1],
                            self.roi[1, 0]:self.roi[1, 1]]
        if len(hoghist.binarized_space) == 0:
            hoghist.bin_size = 9
            hoghist.binarize_1d(constants)
        return hoghist.hist_data(self.grad_angles(im_patch).ravel())



# pylint: disable=C0103
hofhist = SpaceHistogram()
hoghist = SpaceHistogram()

action_recog = ActionRecognition()
