import numpy as np
import cv2
from math import pi
import time
from cv_bridge import CvBridge, CvBridgeError
import yaml
from __init__ import *
import logging


def find_nonzero(arr):
    return np.fliplr(cv2.findNonZero(arr).squeeze())


class CircularOperations(object):
    '''
    Holds operations on positions in circular lists
    '''

    def diff(self, pos1, pos2, leng, no_intersections=False):
        '''
        Returns the smallest distances between
        two positions vectors inside a circular 1d list
        of length leng
        '''
        if no_intersections and pos2.size == 2:
            # some strolls are forbidden
            argmindiffs = np.zeros_like(pos1)
            mindiffs = np.zeros_like(pos1)
            pos_i = np.min(pos2)
            pos_j = np.max(pos2)
            check = pos1 >= pos_j
            tmp = np.concatenate(
                ((leng - pos1[check] + pos_i)[:, None],
                 (-pos_j + pos1[check])[:, None]), axis=1)

            argmindiffs[check > 0] = np.argmin(tmp, axis=1)
            mindiffs[check > 0] = np.min(tmp, axis=1)
            check = pos1 <= pos_i
            tmp = np.concatenate(
                ((pos_i - pos1[check])[:, None],
                 (leng - pos_j + pos1[check])[:, None]), axis=1)
            argmindiffs[check > 0] = np.argmin(tmp, axis=1)
            check = (pos1 <= pos_j) * (pos1 >= pos_i)
            tmp = np.concatenate(
                ((-pos_i + pos1[check])[:, None],
                 (pos_j - pos1[check])[:, None]), axis=1)
            argmindiffs[check > 0] = np.argmin(tmp, axis=1)
            return argmindiffs[mindiffs.argmin()]
        pos1 = pos1[:, None]
        pos2 = pos2[None, :]
        return np.min(np.concatenate((
            np.abs(pos1 - pos2)[:, None],
            np.abs((leng - pos2) + pos1)[:, None],
            np.abs((leng - pos1) + pos2)[:, None]),
            axis=2), axis=2)

    def find_min_dist_direction(self, pos1, pos2, leng, filter_len=None):
        '''
        find pos1 to pos2 minimum distance direction
        '''
        to_count = -1
        if filter_len is not None:
            mask = np.zeros(leng)
            mask[min(pos1, pos2):max(pos1, pos2) + 1] = 1
            if np.sum(mask * leng) > 0:
                to_count = 0
            else:
                to_count = 1
        dists = np.array([np.abs(pos1 - pos2),
                          np.abs(leng - pos2 + pos1),
                          np.abs(leng - pos1 + pos2)])
        if to_count == 1:
            res = np.sign(pos2 - pos1)
            return res, dists[0]
        if to_count == 0:
            dists[0] = 1000000000
        choose = np.argmin(dists)
        if choose == 0:
            res = np.sign(pos2 - pos1)
            if res == 0:
                res = 1
        elif choose == 1:
            res = -1
        else:
            res = 1
        return res, np.min(dists)

    def filter(self, pos1, pos2, length, direction):
        '''
        output: filter_mask of size length with filter_mask[pos1--direction-->pos2]=1, elsewhere 0
        '''
        filter_mask = np.zeros(length)
        if pos2 >= pos1 and direction == 1:
            filter_mask[pos1: pos2 + 1] = 1
        elif pos2 >= pos1 and direction == -1:
            filter_mask[: pos1 + 1] = 1
            filter_mask[pos2:] = 1
        elif pos1 > pos2 and direction == 1:
            filter_mask[: pos2 + 1] = 1
            filter_mask[pos1:] = 1
        elif pos1 > pos2 and direction == -1:
            filter_mask[pos2: pos1 + 1] = 1
        return filter_mask

    def find_single_direction_dist(self, pos1, pos2, length, direction):
        if pos2 >= pos1 and direction == 1:
            return pos2 - pos1
        elif pos2 >= pos1 and direction == -1:
            return length - pos2 + pos1
        elif pos1 > pos2 and direction == 1:
            return length - pos1 + pos2
        elif pos1 > pos2 and direction == -1:
            return pos1 - pos2


class ConvexityDefect(object):
    '''Convexity_Defects holder'''

    def __init__(self):
        self.hand = None


class Contour(object):
    '''Keeping all contours variables'''

    def __init__(self):
        self.arm_contour = np.zeros(1)
        self.hand = np.zeros(1)
        self.cropped_hand = np.zeros(1)
        self.hand_centered_contour = np.zeros(1)
        self.edges_inds = np.zeros(1)
        self.edges = np.zeros(1)
        self.interpolated = Interp()


class Counter(object):
    '''variables needed for counting'''

    def __init__(self):
        self.aver_count = 0
        self.im_number = 0
        self.save_im_num = 0
        self.outlier_time = 0


class CountHandHitMisses(object):
    '''
    class to hold hand hit-misses statistics
    '''

    def __init__(self):
        self.no_obj = 0
        self.found = 0
        self.no_entry = 0
        self.in_im_corn = 0
        self.rchd_mlims = 0
        self.no_cocirc = 0
        self.no_abnorm = 0
        self.rchd_abnorm = 0

    def reset(self):
        self.no_obj = 0
        self.found = 0
        self.no_entry = 0
        self.in_im_corn = 0
        self.rchd_mlims = 0
        self.no_cocirc = 0
        self.no_abnorm = 0
        self.rchd_abnorm = 0

    def print_stats(self):
        members = [attr for attr in dir(self) if not
                   callable(getattr(self, attr))
                   and not attr.startswith("__")]
        for member in members:
            print member, ':', getattr(self, member)


class Data(object):
    '''variables necessary for input and output'''

    def __init__(self):
        self.depth3d = np.zeros(1)
        self.uint8_depth_im = np.zeros(1)
        self.depth = []
        self.color = []
        self.depth_im = np.zeros(1)
        self.color_im = np.zeros(1)
        self.hand_shape = np.zeros(1)
        self.hand_im = np.zeros(1)
        self.initial_im_set = np.zeros(1)
        self.depth_mem = []
        self.reference_uint8_depth_im = np.zeros(1)
        self.depth_raw = None


class Edges(object):

    def __init__(self):
        self.calib_edges = None
        self.calib_frame = None
        self.exists_lim_calib_image = False
        self.edges_positions_indices = None
        self.edges_positions = None
        self.nonconvex_edges_lims = None
        self.exist = False

    def construct_calib_edges(self, im_set=None, convex=0,
                              frame_path=None, edges_path=None,
                              whole_im=False,
                              img=None,
                              write=False):
        if not whole_im:
            im_set = np.array(im_set)
            tmp = np.zeros(im_set.shape[1:], dtype=np.uint8)
            tmp[np.sum(im_set, axis=0) > 0] = 255
        else:
            if img is None:
                raise Exception('img argument must be given')
            tmp = np.ones(img.shape, dtype=np.uint8)
        _, frame_candidates, _ = cv2.findContours(
            tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        frame_index = np.argmax(
            np.array([cv2.arcLength(contour, 1) for contour in
                      frame_candidates]))
        self.calib_edges = np.zeros(
            tmp.shape, dtype=np.uint8)
        self.calib_frame = np.zeros(
            tmp.shape, dtype=np.uint8)
        cv2.drawContours(
            self.calib_edges, frame_candidates,
            frame_index, 255, 1)
        cv2.drawContours(
            self.calib_frame, frame_candidates,
            frame_index, 255, -1)
        if edges_path is None:
            edges_path = CONST['cal_edges_path']
        if frame_path is None:
            frame_path = CONST['cal_edges_path']
        if write:
            cv2.imwrite(CONST['cal_edges_path'], self.calib_edges)
            cv2.imwrite(CONST['cal_frame_path'], self.calib_frame)

    def load_calib_data(self, convex=0, edges_path=None, frame_path=None,
                        whole_im=False, img=None):
        if whole_im:
            if img is None:
                raise Exception('img argument must be given')
            self.construct_calib_edges(
                img=img, whole_im=True, write=False)
        else:
            if (frame_path is None) ^ (edges_path is None):
                if frame_path is None:
                    logging.error('Missing frame_path input, but edges_path is' +
                                  ' given')
                else:
                    logging.error('Missing edges_path input, but frame_path is' +
                                  ' given')
            if edges_path is None:
                edges_path = CONST['cal_edges_path']
                frame_path = CONST['cal_frame_path']
            if not os.path.isfile(edges_path):
                self.exists_lim_calib_image = 0
            else:
                self.exists_lim_calib_image = 1
                self.calib_frame = cv2.imread(frame_path, 0)
                self.calib_edges = cv2.imread(frame_path, 0)
                self.calib_frame[
                    self.calib_frame < 0.9 * np.max(self.calib_frame)] = 0
                self.calib_edges[
                    self.calib_edges < 0.9 * np.max(self.calib_edges)] = 0
        if not convex:
            self.find_non_convex_edges_lims()
        else:
            self.find_convex_edges_lims()
        self.exist = True
        return edges_path, frame_path

    def find_non_convex_edges_lims(self, edge_tolerance=10):
        '''
        Find non convex symmetrical edges minimum orthogonal lims with some tolerance
        Inputs: positions,edges mask[,edges tolerance=10]
        '''
        self.edges_positions_indices = np.nonzero(cv2.dilate(
            self.calib_edges, np.ones((3, 3), np.uint8), cv2.CV_8U) > 0)
        self.edges_positions = np.transpose(
            np.array(self.edges_positions_indices))
        lr_positions = self.edges_positions[
            np.abs(self.edges_positions[:, 0] - self.calib_edges.shape[0] / 2.0) < 1, :]
        tb_positions = self.edges_positions[
            np.abs(self.edges_positions[:, 1] - self.calib_edges.shape[1] / 2.0) < 1, :]
        self.nonconvex_edges_lims = np.array(
            [np.min(lr_positions[:, 1]) + edge_tolerance,
             np.min(tb_positions[:, 0]) + edge_tolerance,
             np.max(lr_positions[:, 1]) - edge_tolerance,
             np.max(tb_positions[:, 0]) - edge_tolerance])

    def find_convex_edges_lims(self, positions=None):
        '''
        Find convex edges minimum orthogonal lims
        '''
        def calculate_cart_dists(cart_points, cart_point=[]):
            '''
            Input either numpy array either 2*2 list
            Second optional argument is a point
            '''
            if cart_point == []:

                try:
                    return np.sqrt(
                        (cart_points[1:, 0] - cart_points[: -1, 0]) *
                        (cart_points[1:, 0] - cart_points[: -1, 0]) +
                        (cart_points[1:, 1] - cart_points[: -1, 1]) *
                        (cart_points[1:, 1] - cart_points[: -1, 1]))
                except (TypeError, AttributeError):
                    return np.sqrt((cart_points[0][0] - cart_points[1][0])**2 +
                                   (cart_points[0][1] - cart_points[1][1])**2)

            else:
                return np.sqrt(
                    (cart_points[:, 0] - cart_point[0]) *
                    (cart_points[:, 0] - cart_point[0]) +
                    (cart_points[:, 1] - cart_point[1]) *
                    (cart_points[:, 1] - cart_point[1]))
        calib_positions = positions[self.calib_edges > 0, :]
        calib_dists = calculate_cart_dists(
            calib_positions,
            np.array([0, 0]))

        upper_left = calib_positions[np.argmin(calib_dists), :]
        calib_dists2 = calculate_cart_dists(
            calib_positions,
            np.array([self.calib_edges.shape[0],
                      self.calib_edges.shape[1]]))
        lower_right = calib_positions[np.argmin(calib_dists2), :]
        # Needs filling
        self.convex_edges_lims = []


class ExistenceProbability(object):
    '''
    Class to find activated cells
    '''

    def __init__(self):
        self.init_val = 0
        self.distance = np.zeros(0)
        self.distance_mask = np.zeros(0)
        self.objects_mask = np.zeros(0)
        self.can_exist = np.zeros(0)
        self.max_distancex = 50
        self.max_distancey = 25
        self.framex1 = 0
        self.framex2 = 0
        self.framey1 = 0
        self.framey2 = 0
        self.always_checked = []
        self.wearing_par = 8
        self.wearing_mat = np.zeros(0)

    def calculate(self):
        '''
        calculate activated cells
        '''
        if self.init_val == 0:
            self.wearing_mat = np.zeros(segclass.total_obj_num)
            self.init_val = 1
            # self.distance_mask = np.pad(
            # np.ones(tuple(np.array(data.depth_im.shape) - 2)), 1, 'constant')
            sums = np.sum(edges.calib_frame[
                          :, 1: edges.calib_frame.shape[1] / 2], axis=0) > 0
            self.framex1 = np.where(np.diff(sums))[0] + self.max_distancex
            self.framex2 = meas.imx - self.framex1
            self.framey1 = self.max_distancey
            self.framey2 = meas.imy - self.max_distancey
            for count, center in enumerate(segclass.nz_objects.initial_center):
                if center[0] < self.framey1 or center[0] > self.framey2 or\
                        center[1] < self.framex1 or center[1] > self.framex2:
                    self.always_checked.append(count)

        new_arrivals = []
        for neighborhood in segclass.filled_neighborhoods:
            new_arrivals += neighborhood
        self.wearing_mat[new_arrivals + self.always_checked] = self.wearing_par
        self.can_exist = np.where(
            self.wearing_mat[: segclass.nz_objects.count] > 0)[0]

        self.wearing_mat -= 1
        '''
        im_res = np.zeros_like(data.depth_im, dtype=np.uint8)
        for val in self.can_exist:
            im_res[segclass.objects.image == val] = 255
        im_results.images.append(im_res)
        '''


class Hull(object):
    '''Convex Hulls of contours'''

    def __init__(self):
        self.hand = None


class Interp(object):
    '''Interpolated Contour variables'''

    def __init__(self):
        self.vecs_starting_ind = None
        self.vecs_ending_ind = None
        self.points = None
        self.final_segments = None


class KalmanFilter:

    def __init__(self):
        self.prev_objects_mask = np.zeros(0)
        self.cur_objects_mask = np.zeros(0)
        self.can_exist = np.zeros(0)


class Lim(object):
    '''limits for for-loops'''

    def __init__(self):
        self.max_im_num_to_save = 0
        self.init_n = 0


class Mask(object):
    '''binary masks'''

    def __init__(self):
        self.rgb_final = None
        self.background = None
        self.final_mask = None
        self.color_mask = None
        self.memory = None
        self.anamneses = None
        self.bounds = None


class Memory(object):
    '''
    class to keep memory
    '''

    def __init__(self):
        self.memory = None
        self.image = None
        self.bounds = None
        self.bounds_mask = None

    def live(self, anamnesis, importance=1):
        '''
        add anamnesis to memory
        '''
        if not type(anamnesis[0, 0]) == np.bool_:
            anamnesis = anamnesis > 0
        if self.image is None:
            self.image = np.zeros(anamnesis.shape)
        else:
            self.image *= (1 - CONST['memory_fade_exp'])
        self.image += (CONST['memory_power'] * anamnesis * importance -
                       CONST['memory_fade_const'])
        self.image[self.image < 0] = 0

    def erase(self):
        '''
        erase memory
        '''
        self.memory = None
        self.image = None

    def remember(self, find_bounds=False):
        '''
        remember as many anamneses as possible
        '''
        if self.image is None:
            return None, None, None, None
        self.memory = self.image > CONST['memory_strength']
        if np.sum(self.memory) == 0:
            return None, None, None, None
        if find_bounds:
            self.anamneses = np.atleast_2d(cv2.findNonZero(self.memory.
                                                           astype(np.uint8)).squeeze())
            x, y, w, h = cv2.boundingRect(self.anamneses)
            self.bounds_mask = np.zeros_like(self.image, np.uint8)
            cv2.rectangle(self.bounds_mask, (x, y), (x + w, y + h), 1, -1)
            self.bounds = (x, y, w, h)
            return self.memory, self.anamneses, self.bounds, self.bounds_mask
        return self.memory, None, None, None

# class BackProjectionFilter(object):


class CamShift(object):
    '''
    Implementation of camshift algorithm for depth images
    '''

    def __init__(self):
        self.track_window = None
        self.rect = None

    def calculate(self, frame, mask1, mask2=None):
        '''
        frame and mask are 2d arrays
        '''
        _min = np.min(frame[frame != 0])
        inp = ((frame.copy() - _min) /
               float(np.max(frame) - _min)).astype(np.float32)
        inp[inp < 0] = 0
        hist = cv2.calcHist([inp], [0],
                            (mask1 > 0).astype(np.uint8),
                            [1000],
                            [0, 1])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hist = hist.reshape(-1)
        if self.track_window and self.track_window[2] > 0 and self.track_window[3] > 0:
            if mask2 is None:
                mask2 = np.ones(frame.shape)
            prob_tmp = cv2.calcBackProject([inp[mask2 > 0]], [0], hist, [0, 1],
                                           1).squeeze()
            prob = np.zeros(frame.shape)
            prob[mask2 > 0] = prob_tmp
            '''
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         CONST['max_count'],
                         CONST['par_eps'])
            self.rect, self.track_window = cv2.CamShift(prob,
                                                             self.track_window,
                                                             term_crit)
            '''
        return prob

    def reset(self):
        '''
        reset algorithm
        '''
        self.track_window = None
        self.rect = None


class Measure(object):
    '''variables from measurements'''

    def __init__(self):
        self.w = 0
        self.h = 0
        self.w_hand = 0
        self.h_hand = 0
        self.imx = 0
        self.imy = 0
        self.min1 = 0
        self.min2 = 0
        self.least_resolution = 0
        self.aver_depth = 0
        self.len = None
        self.lam = None
        self.interpolated_contour_angles = 0
        self.segment_angle = None
        self.segment_points_num = None
        self.contours_areas = None
        self.im_count = 0
        self.nprange = np.zeros(0)
        self.erode_size = 5
        # lims: l->t->r->b
        self.nonconvex_edges_lims = []
        self.convex_edges_lims = []
        self.edges_positions_indices = []
        self.edges_positions = []
        # trusty_pixels is 1 for the pixels that remained nonzero during
        # initialisation
        self.trusty_pixels = np.zeros(1)
        # valid_values hold the last seen nonzero value of an image pixel
        # during initialisation
        self.valid_values = np.zeros(1)
        # changed all_positions to cart_positions
        self.cart_positions = None
        self.polar_positions = None
        self.background = np.zeros(1)
        self.found_objects_mask = np.zeros(0)
        self.hand_patch = None
        self.hand_patch_pos = None

    def construct_positions(self, img, polar=False):
        self.cart_positions = np.transpose(np.nonzero(np.ones_like(
            img))).reshape(img.shape + (2,))
        if polar:
            cmplx_positions = (self.cart_positions[:, :, 0] * 1j +
                               self.cart_positions[:, :, 1])
            ang = np.angle(cmplx_positions)
            ang[ang < -pi] += 2 * pi
            ang[ang > pi] -= 2 * pi
            self.polar_positions = np.concatenate(
                (np.absolute(cmplx_positions)[..., None], ang[..., None]),
                axis=2)
            return self.cart_positions, self. polar_positions
        else:
            return self.cart_positions


class Model(object):
    '''variables for modelling nonlinearities'''

    def __init__(self):
        self.noise_model = (0, 0)
        self.med = None
        self.var = None


class NoiseRemoval(object):

    def remove_noise(self, thresh=None, norm=True, img=None, in_place=False):
        if img is None:
            img = data.depth_im
        elif id(img) == id(data.depth_im):
            in_place = True
        if thresh is None:
            thresh = CONST['noise_thres']
        mask = img < thresh
        img = img * mask
        if norm:
            img = img / float(thresh)
        if in_place:
            data.depth_im = img
        return img


class Path(object):
    '''necessary paths for loading and saving'''

    def __init__(self):
        self.depth = ''
        self.color = ''


class Point(object):
    '''variables addressing to image coordinates'''

    def __init__(self):
        self.x = 0
        self.y = 0
        self.x_hand = 0
        self.y_hand = 0
        self.wristpoints = None

# pylint:disable=no-self-use


class PolarOperations(object):
    '''
    Class to hold all used  operations on polar coordinates
    '''

    def find_cocircular_points(self, polar_points, radius, resolution=np.sqrt(2) / 2.0):
        '''
        Find cocircular points given radius and suggested resolution
        '''
        return polar_points[np.abs(polar_points[:, 0] - radius) <= resolution, :]

    def change_origin(self, old_polar, old_ref_angle, old_ref_point, new_ref_point):
        '''Caution: old_polar is changed'''
        old_polar[:, 1] += old_ref_angle
        complex_diff = (new_ref_point[0] - old_ref_point[0]) * \
            1j + new_ref_point[1] - old_ref_point[1]
        polar_diff = [np.absolute(complex_diff), np.angle(complex_diff)]
        _radius = np.sqrt(polar_diff[0]**2 + old_polar[:, 0] * old_polar[:, 0] - 2 *
                          polar_diff[0] * old_polar[:, 0] * np.cos(old_polar[:, 1] - polar_diff[1]))
        _sin = old_polar[:, 0] * np.sin(old_polar[:, 1]) - \
            polar_diff[0] * np.sin(polar_diff[1])
        _cos = old_polar[:, 0] * np.cos(old_polar[:, 1]) - \
            polar_diff[0] * np.cos(polar_diff[1])
        _angle = np.arctan2(_sin, _cos)
        return np.concatenate((_radius[:, None], _angle[:, None]), axis=1)

    def polar_to_cart(self, polar, center, ref_angle):
        '''
        Polar coordinates to cartesians,
        given center and reference angle
        '''
        return np.concatenate(
            ((polar[:, 0] * np.sin(ref_angle + polar[:, 1]) +
              center[0])[:, None].astype(int),
             (polar[:, 0] * np.cos(ref_angle + polar[:, 1]) +
              center[1])[:, None].astype(int)), axis=1)

    def mod_correct(self, polar):
        '''
        Correct polar points, subjects to
        modular (-pi,pi). polar is changed.
        '''
        polar[polar[:, 1] > pi, 1] -= 2 * pi
        polar[polar[:, 1] < -pi, 1] += 2 * pi

    def fix_angle(self, angle):
        '''
        Same with mod_correct, for single angles
        '''
        if angle < -pi:
            angle += 2 * pi
        elif angle > pi:
            angle -= 2 * pi
        return angle

    def mod_between_vals(self, angles, min_bound, max_bound):
        '''
        Find angles between bounds, using modular (-pi,pi) logic
        '''
        if max_bound == min_bound:
            return np.zeros((0))
        res = self.mod_diff(max_bound, min_bound, 1)[1]
        if res == 0:
            return (angles <= max_bound) * (angles >= min_bound)
        else:
            return ((angles >= max_bound) * (angles <= pi)
                    + (angles >= -pi) * (angles <= min_bound))

    def mod_diff(self, angles1, angles2, ret_argmin=0):
        '''
        Angle substraction using modulo in (-pi,pi)
        '''
        sgn = -1 + 2 * (angles1 > angles2)
        if len(angles1.shape) == 0:

            diff1 = np.abs(angles1 - angles2)
            diff2 = 2 * pi - diff1
            if ret_argmin:
                return sgn * min([diff1, diff2]), np.argmin([diff1, diff2])
            else:
                return sgn * min([diff1, diff2])
        diff = np.empty((2, angles1.shape[0]))
        diff[0, :] = np.abs(angles1 - angles2)
        diff[1, :] = 2 * pi - diff[0, :]
        if ret_argmin:
            return sgn * np.min(diff, axis=0), np.argmin(diff, axis=0)
        else:
            return sgn * np.min(diff, axis=0)
# pylint:enable=no-self-use


class Result(object):
    '''class to keep results'''

    def __init__(self):
        self.images = []
        self.data = []
        self.name = 'Results'
        self.im_name = ' '
        self.maxdim = 3
        self.images = []
        self.data = []
        self.name = 'Results'

    def show_results(self, var1, var2):
        '''class to save or display images like montage'''
        if len(self.images) == 0:
            return 1
        shapes = [(im_shape[0], im_shape[1], c, len(im_shape))
                  for (c, im_shape) in enumerate([im.shape for im in self.images])]
        isrgb = 1 if sum(zip(*shapes)[3]) != 2 * len(shapes) else 0
        sorted_images = [self.images[i] for i in list(
            zip(*sorted(shapes, key=lambda x: x[0], reverse=True))[2])]
        imy, imx, _, _ = tuple([max(coord) for
                                coord in zip(*shapes)])
        yaxis = (len(self.images) - 1) / self.maxdim + 1
        xaxis = min(self.maxdim, len(self.images))
        if not isrgb:
            montage = 255 *\
                np.ones((imy * yaxis, imx * xaxis), dtype=np.uint8)
        elif isrgb:
            montage = 255 *\
                np.ones((imy * yaxis, imx * xaxis, 3), dtype=np.uint8)
        x_init = 0
        for count, image in enumerate(sorted_images):
            _max = np.max(image)
            _min = np.min(image)
            if _max != _min:
                image = ((image - _min) / float(_max - _min) *
                         255).astype(np.uint8)
            if isrgb:
                if len(image.shape) == 2:
                    image = np.tile(image[:, :, None], (1, 1, 3))
            if not isrgb:
                montage[(count / self.maxdim) * imy: (count / self.maxdim + 1)
                        * imy, x_init: x_init + image.shape[1]] = image
            else:
                montage[(count / self.maxdim) * imy: (count / self.maxdim + 1)
                        * imy, x_init: x_init + image.shape[1], :] = image
            if (count + 1) % self.maxdim == 0:
                x_init = 0
            else:
                x_init += image.shape[1]
        if var1 == 'stdout':
            cv2.imshow('results', montage)
        elif var1 == 'ros':
            if not isrgb:
                montage = np.tile(montage[:, :, None], (1, 1, 3))
            try:
                var2[0].publish(var2[1].cv2_to_imgmsg(montage, 'bgr8'))
            except CvBridgeError as e:
                raise(e)
        else:
            cv2.imwrite(var2, montage)
        return

    def print_results(self, filename):
        '''class to print data to file'''
        with open(filename, 'w') as fil:
            for line in self.data:
                fil.write(line + '\n')


class SceneObjects():
    '''Class to process segmented objects'''

    def __init__(self):
        self.pixsize = []
        self.xsize = []
        self.ysize = []
        self.center = np.zeros(0)
        self.center_displacement = np.zeros(0)
        self.centers_to_calculate = np.zeros(0)
        self.center_displ_angle = np.zeros(0)
        self.count = -1
        self.initial_center = np.zeros(0)
        self.initial_vals = []
        self.locs = np.zeros(0)
        self.masses = np.zeros(0)
        self.vals = np.zeros(0)
        self.count = np.zeros(0)
        self.is_unreliable = np.zeros(0)
        self.image = np.zeros(0)
        self.pixel_dim = 0
        self.untrusty = []

    def find_partitions(self, points, dim):
        '''Separate big segments'''
        center = np.mean(points, axis=1)
        objs = []
        points = np.array(points)

        compare = points <= center[:, None]
        if dim == 'all':
            objs.append(np.reshape(points[np.tile(np.all(compare, axis=0)[None, :],
                                                  (2, 1))], (2, -1)))

            objs.append(np.reshape(points[np.tile(
                np.all((compare[0, :],
                        np.logical_not(compare[1, :])), axis=0)[None, :], (2,
                                                                           1))], (2, -1)))
            objs.append(np.reshape(points[np.tile(
                np.all((np.logical_not(compare[0, :]),
                        compare[1, :]), axis=0)[None, :], (2, 1))], (2, -1)))
            objs.append(np.reshape(
                points[np.tile(np.all(np.logical_not(compare), axis=0)[None, :], (2,
                                                                                  1))], (2, -1)))
        elif dim == 'x':
            objs.append(np.reshape(points[np.tile(compare[1, :][None, :], (2,
                                                                           1))], (2, -1)))
            objs.append(np.reshape(
                points[np.tile(np.logical_not(compare[1, :])[None, :], (2,
                                                                        1))], (2, -1)))
        else:
            objs.append(np.reshape(points[np.tile(compare[0, :][None, :], (2,
                                                                           1))], (2, -1)))
            objs.append(np.reshape(
                points[np.tile(np.logical_not(compare[0, :])[None, :], (2,
                                                                        1))], (2, -1)))
        return objs

    def register_object(self, points, pixsize, xsize, ysize):
        '''Register object to objects structure'''
        minsize = 5
        if xsize > minsize or ysize > minsize:
            self.count += 1
            self.image[tuple(points)] = self.count
            self.pixsize.append(pixsize)
            self.xsize.append(xsize)
            self.ysize.append(ysize)
        else:
            self.untrusty.append((points, pixsize, xsize, ysize))

    def check_object_dims(self, points):
        '''Check if segments are big'''
        maxratio = 10
        if len(points) <= 1:
            return ['ok', 1, 1, 1]
        xymax = np.max(points, axis=1)
        xymin = np.min(points, axis=1)
        ysize = xymax[1] - xymin[1] + 1
        xsize = xymax[0] - xymin[0] + 1
        ans = ''
        if ysize > 2 * meas.imx / maxratio and xsize > 2 * meas.imx / maxratio:
            ans = 'all'
        elif ysize > 2 * meas.imx / maxratio:
            ans = 'x'
        elif xsize > 2 * meas.imx / maxratio:
            ans = 'y'
        else:
            ans = 'ok'
        return [ans, len(points[0]), xsize, ysize]

    def object_partition(self, points, check):
        '''Recursively check and register objects to objects structure'''
        if check[0] == 'ok':
            if np.any(points):
                self.register_object(points, check[1], check[2], check[3])
            return
        objs = self.find_partitions(points, check[0])
        for obj in objs:
            self.object_partition(obj, self.check_object_dims(points))

    def process(self, val, pos):
        '''Process segments'''
        points = np.unravel_index(pos, data.depth_im.shape)
        self.object_partition(points, self.check_object_dims(points))
        return self.count

    def find_centers_displacement(self):
        self.center_displacement = self.center - self.initial_center
        self.center_displ_angle = np.arctan2(
            self.center_displacement[:, 0], self.center_displacement[:, 1])

    def find_object_center(self, refer_to_nz):
        '''Find scene objects centers of mass'''
        first_time = self.locs.size == 0
        self.pixel_dim = np.max(self.pixsize)
        if first_time:
            data.uint8_depth_im = data.reference_uint8_depth_im
            self.center = np.zeros((self.count + 1, 2), dtype=int)
            self.centers_to_calculate = np.ones(
                (self.count + 1), dtype=bool)
            self.locs = np.zeros((self.pixel_dim, self.count + 1),
                                 dtype=complex)

            for count in range(self.count + 1):
                vals_mask = (self.image == count)
                try:
                    locs = find_nonzero(vals_mask.astype(np.uint8))
                    self.locs[:locs.shape[0], count] = locs[
                        :, 0] + locs[:, 1] * 1j
                except:
                    pass
                self.pixsize[count] = locs.shape[0]
            self.vals = np.empty((self.count + 1,
                                  self.pixel_dim))
            self.initial_vals = np.zeros(
                (self.count + 1, self.pixel_dim))
            self.masses = np.ones(self.count + 1, dtype=int)

        else:
            data.uint8_depth_im = data.uint8_depth_im
            self.center = self.initial_center.copy()
            existence.calculate()
            self.centers_to_calculate = np.zeros(
                (self.count + 1), dtype=bool)

            self.centers_to_calculate[np.array(existence.can_exist)] = 1
        '''cv2.imshow('test',self.uint8_depth_im)
        cv2.waitKey(0)
        '''
        for count in np.arange(self.count + 1)[self.centers_to_calculate]:
            xcoords = self.locs[
                :self.pixsize[count], count].real.astype(int)
            ycoords = self.locs[
                :self.pixsize[count], count].imag.astype(int)
            self.masses[count] = np.sum(data.uint8_depth_im[xcoords, ycoords])
            if refer_to_nz:
                if self.masses[count] > 0:
                    self.vals[count, :self.pixsize[count]] = (data.uint8_depth_im
                                                              )[xcoords, ycoords
                                                                ][None, :]
                    complex_res = np.dot(
                        self.vals[count, :self.pixsize[count]],
                        self.locs[:self.pixsize[count], count]) / self.masses[count]
                    self.center[count, :] = np.array(
                        [int(complex_res.real), int(complex_res.imag)])
                else:
                    if first_time:
                        complex_res = np.mean(self.locs[
                            :self.pixsize[count], count])
                        self.center[count, :] = np.array(
                            [complex_res.real.astype(int), complex_res.imag.astype(int)])
            else:
                complex_res = np.mean(self.locs[
                    :self.pixsize[count], count])
                self.center[count, :] = np.array(
                    [complex_res.real.astype(int), complex_res.imag.astype(int)])

            if first_time:

                self.initial_vals[
                    count, :self.pixsize[count]] = data.uint8_depth_im[xcoords, ycoords]
        if first_time:
            self.initial_center = self.center.copy()


class Segmentation(object):
    ''' Objects used for background segmentation '''

    def __init__(self):
        self.bounding_box = []
        self.needs_segmentation = 1
        self.check_if_segmented_1 = 0
        self.check_if_segmented_2 = 0
        self.prev_im = np.zeros(0)
        self.exists_previous_segmentation = 0
        self.initialised_centers = 0
        self.z_objects = SceneObjects()
        self.nz_objects = SceneObjects()
        self.neighborhood_existence = np.zeros(0)
        self.proximity_table = np.zeros(0)
        self.filled_neighborhoods = []
        self.found_objects = np.zeros(0)
        self.total_obj_num = 0
        self.fgbg = None

    def flush_previous_segmentation(self):
        self.bounding_box = []
        self.z_objects = SceneObjects()
        self.nz_objects = SceneObjects()
        self.proximity_table = np.zeros(0)
        self.filled_neighborhoods = []
        self.found_objects = np.zeros(0)
        self.total_obj_num = 0

    def initialise_neighborhoods(self):
        center = np.array(list(self.nz_objects.center) +
                          list(self.z_objects.center))
        zcenters = (center[:, 0] + center[:, 1] * 1j)[:, None]
        distances = abs(zcenters - np.transpose(zcenters))
        sorted_indices = np.argsort(distances, axis=0)
        self.proximity_table = sorted_indices[:18, :]
        self.total_obj_num = self.nz_objects.count + self.z_objects.count + 2
        self.neighborhood_existence = np.zeros((self.total_obj_num,
                                                self.total_obj_num), dtype=int)

        for count in range(self.total_obj_num):
            self.neighborhood_existence[count, :] = np.sum(self.proximity_table
                                                           == count, axis=0)

    def find_objects(self):
        time1 = time.clock()
        check_atoms = []
        # nz_objects.center at the beginning of center list so following is
        # valid
        for count1, vec in\
                enumerate(self.nz_objects.center_displacement):
            if (abs(vec[0]) > CONST['min_displacement'] or abs(vec[1]) >
                    CONST['min_displacement']):
                if np.linalg.norm(vec) > 0:
                    check_atoms.append(count1)
        sliced_proximity = list(self.proximity_table[:, check_atoms].T)
        neighborhoods = []
        self.filled_neighborhoods = []
        for atom, neighbors in enumerate(sliced_proximity):

            neighborhood_id = -1
            for n_id, neighborhood in enumerate(neighborhoods):
                if check_atoms[atom] in neighborhood:
                    neighborhood_id = n_id
                    break
            if neighborhood_id == -1:
                neighborhoods.append([check_atoms[atom]])
                self.filled_neighborhoods.append([check_atoms[atom]])
            for neighbor in neighbors:
                if self.neighborhood_existence[neighbor, check_atoms[atom]]:
                    if neighbor in check_atoms:
                        if neighbor not in neighborhoods[neighborhood_id]:
                            neighborhoods[neighborhood_id].append(neighbor)
                    if neighbor not in self.filled_neighborhoods[neighborhood_id]:
                        self.filled_neighborhoods[
                            neighborhood_id].append(neighbor)

        time2 = time.clock()
        for neighborhood in self.filled_neighborhoods:
            if 0 in neighborhood:
                self.filled_neighborhoods.remove(neighborhood)

        time1 = time.clock()
        self.found_objects = np.zeros(data.depth_im.shape)

        self.bounding_box = []
        for neighborhood in self.filled_neighborhoods:
            for neighbor in neighborhood:
                if neighbor > self.nz_objects.count:  # z_objects here
                    locs = self.z_objects.locs[
                        :self.z_objects.pixsize[neighbor - (self.nz_objects.count + 1)],
                        neighbor - (self.nz_objects.count + 1)]
                    neighborhood_xs = np.real(locs).astype(int)
                    neighborhood_ys = np.imag(locs).astype(int)
                    vals = data.uint8_depth_im[
                        neighborhood_xs, neighborhood_ys]
                    valid_values = meas.valid_values[
                        neighborhood_xs, neighborhood_ys]
                    vals = ((np.abs(valid_values.astype(float)
                                    - vals.astype(float))).astype(np.uint8) >
                            CONST['depth_tolerance']) * (vals > 0)
                else:  # nz_objects here
                    locs = self.nz_objects.locs[
                        :self.nz_objects.pixsize[neighbor],
                        neighbor]
                    init_vals = self.nz_objects.initial_vals[
                        neighbor, :self.nz_objects.pixsize[neighbor]]
                    last_vals = self.nz_objects.vals[
                        neighbor,
                        :self.nz_objects.pixsize[neighbor]]
                    vals = (np.abs(last_vals -
                                   init_vals) >
                            CONST['depth_tolerance']) * (last_vals > 0)
                    neighborhood_xs = np.real(locs).astype(int)
                    neighborhood_ys = np.imag(locs).astype(int)

                self.found_objects[neighborhood_xs,
                                   neighborhood_ys] = vals

                ''''
                if np.min(neighborhood_xs)<self.bounding_box[count][0]:
                    self.bounding_box[count][0]=np.min(neighborhood_xs)
                if np.min(neighborhood_ys)<self.bounding_box[count][1]:
                    self.bounding_box[count][1]=np.min(neighborhood_ys)
                if np.max(neighborhood_xs)>self.bounding_box[count][0]:
                    self.bounding_box[count][2]=np.max(neighborhood_xs)
                if np.max(neighborhood_ys)>self.bounding_box[count][1]:
                    self.bounding_box[count][3]=np.max(neighborhood_ys)
                '''

        # self.found_objects=data.depth_im*((self.found_objects>20))
        im_results.images.append(self.found_objects)
        # im_results.images.append(np.abs(((self.found_objects+(meas.trusty_pixels==0))>0)
        #                        -0.5*(self.z_objects.image>0)))
        time2 = time.clock()
        '''
        print 'Found', len(self.filled_neighborhoods), 'objects in', time2 - time1, 's'
        for count1 in range(len(self.size)):
            for count2 in range(count1, len(self.size)):
                count += 1
                vec2 = self.center_displacement[count2]
                vec1 = self.center_displacement[count1]
                if np.sqrt(vec1[0]**2 + vec1[1]**2) > 7 and\
                   np.sqrt(vec2[0]**2 + vec2[1]**2) > 7:
                    pnt1 = self.initial_center[count1]
                    pnt2 = self.initial_center[count2]
                    v1v2cross = np.cross(vec1, vec2)
                    if np.any(v1v2cross > 0):
                         lam = np.sqrt(np.sum(np.cross(pnt2 - pnt1, vec2) ** 2)) /\
                            float(np.sqrt(np.sum(v1v2cross**2)))
                         pnt = pnt1 + lam * vec1
                         if pnt[0] > 0 and pnt[0] < meas.imy\
                           and pnt[1] > 0 and pnt[1] < meas.imx and\
                           masks.calib_frame[int(pnt[0]), int(pnt[1])] > 0:
                           # and
                           # existence.can_exist[int(pnt[0]),int(pnt[1])]>1:
                             inters.append(pnt.astype(int))

        complex_inters = np.array(
            [[complex(point[0], point[1]) for point in inters]])
        # euclid_dists = np.abs(np.transpose(complex_inters) - complex_inters)
        if inters:
            desired_center = np.median(np.array(inters), axis=0)
            real_center_ind = np.argmin(
                np.abs(complex_inters -
                       np.array([[complex(desired_center[0], desired_center[1])]])))
            return inters, inters[real_center_ind]
        else:
            return 'Not found intersection points', []

        '''
        return self.found_objects


class Threshold(object):
    '''necessary threshold variables'''

    def __init__(self):
        self.lap_thres = 0
        self.depth_thres = 0

with open(CONST_LOC + "/config.yaml", 'r') as stream:
    try:
        CONST = yaml.load(stream)
    except yaml.YAMLError as exc:
        print exc
# pylint: disable=C0103
circ_oper = CircularOperations()
contours = Contour()
counters = Counter()
chhm = CountHandHitMisses()
data = Data()
edges = Edges()
lims = Lim()
masks = Mask()
meas = Measure()
models = Model()
paths = Path()
points = Point()
pol_oper = PolarOperations()  # depends on Measure class
thres = Threshold()
im_results = Result()
segclass = Segmentation()
existence = ExistenceProbability()
interpolated = contours.interpolated
noise_proc = NoiseRemoval()
