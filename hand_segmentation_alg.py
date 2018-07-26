'''
Following algorithm refers to the obsolete part:
Input: Hand Mask nonzero xy_points
Use laplacian of binary mask
0.Find entry segment
1.Calculate segment middle point
2.Transform nonzero points  across the right semiplane* to new_polar coordinates ,
with reference center the point found and reference angle the one of the normal
to the semiplane*.
3.Start augmenting a semicircle from the reference center in the
semiplane* ,calculate zero crossings on its perimeter, with low step,
and append the nonzero points  to a 2d list.

4.While 4 zerocrossings (2 edge points), calculate distance of white points and mean angle:
    i. If mean angle changes, assume joint existence. Record segment as joint and go to 1.
    ii.If distance increases steadily  inside a checked window, assume reach of
    wrist (still disputed, so dont do anything serious here).Record mean segment as special
    one.
    iii.If distance severely decreases, assume wrist bypassed. Begin from the
    segment in list where everything was ok. Go to 6
5. If more than 4 zerocrossings, with max distance approximately same as the
distance with 4 zerocrossings, assume reach of fingers.Record mean segment as
special one. Go back to where 4 zerocrossings were observed:
    If 4ii happened:
        find mean segment between the two special ones. Go to 6
    else:
        go to 6.

6.
        i. do 1 and 2
        ii. find 4 closest points with almost the same radius, whose angle sum
        is closer to zero.
        iii. Palm is found. Record palm circle

*Semiplane is considered the one with limits defined by the  normal to the reference
segment. Reference segment is found by finding the normal to the axis of link.
The direction is at 0-th joint towards the center of the image and
at n-th joint defined by previous directions.
'''
import logging
from math import pi
import os
# import time
import numpy as np
import cv2
import class_objects as co
if __name__ == '__main__':
    import urllib
    import yaml
    import timeit
    import cProfile as profile
from __init__ import get_real_coordinate

def with_laplacian(binarm):
    '''
    Find edges using Laplacian
    '''
    return cv2.Laplacian(binarm, cv2.CV_8U)


def usingcontours(imag):
    '''
    Find edges using cv2.findContours
    '''
    points = np.transpose(cv2.findContours(
        imag.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1][0])
    tmp = np.zeros_like(imag)
    tmp[tuple(points[::-1, :])] = 255
    return tmp


def find_rows_in_array(arr, rows):
    '''
    find indices of rows in array if they exist
    '''
    tmp = np.prod(np.swapaxes(
        arr[:, :, None], 1, 2) == rows, axis=2)
    return np.sum(np.cumsum(tmp, axis=0) * tmp == 1,
                  axis=1) > 0


def array_row_intersection(arr1, arr2):
    '''
    Returns rows, which exist in both arr1 and arr2
    '''
    return arr1[find_rows_in_array(arr1, arr2)]

def find_trues_segments(inp, iscircular):
    if iscircular:
        shift = np.argmin(inp)
        inp = np.roll(inp, -shift)
        a = np.concatenate(([0], inp, [0]))
        dif = np.diff(a)[1:]
    else:
        a = np.concatenate(([0], inp, [0]))
        dif = np.diff(a)
    ind_start = cv2.findNonZero(
        np.roll(dif == 1, 1)[:, None].astype(np.uint8))[:, 0, 1]
    ind_end = cv2.findNonZero(
        (dif == -1)[:, None].astype(np.uint8))[:, 0, 1]
    if iscircular:
        ind_start = (ind_start + shift)%len(inp)
        ind_end = (ind_end + shift) %len(inp)
    return np.concatenate((ind_start[:,None],ind_end[:,None]),axis=-1)

def find_largest_trues_segment(inp, iscircular=True, ret_filtered=True):
    if inp.sum() == inp.size:
        res = np.array([0, inp.size - 1])
        filtered = inp.copy()
    else:
        if iscircular:
            shift = np.argmin(inp)
            inp = np.roll(inp, -shift)
        a = np.concatenate(([0], inp, [0]))
        dif = np.diff(a)[1:]
        ind_start = cv2.findNonZero(
            np.roll(dif == 1, 1)[:, None].astype(np.uint8))[:, 0, 1]
        ind_end = cv2.findNonZero(
            (dif == -1)[:, None].astype(np.uint8))[:, 0, 1]
        tmp = np.cumsum(inp)[ind_end]
        mass = np.diff(np.concatenate(([0], tmp)))
        ind_largest = np.argmax(mass)
        res = np.array([ind_start[ind_largest], ind_end[ind_largest]])

        if ret_filtered:
            filtered = inp.copy()
            filtered[:res[0]] = 0
            filtered[res[1] + 1:] = 0
            filtered = np.roll(filtered, shift)
        if iscircular:
            res = res + shift
            res = res % len(inp)
    if ret_filtered:
        return res, filtered.astype(bool)
    else:
        return res

def find_corrected_point(polar_points, ref_angle,
                         ref_point, ref_radius,
                         entry_angles=None, entry_points=None,
                         width=None):
    '''
    Find matching cocircular point.
    if entry_angles are given, the best point is found to be the one with the
        least angle difference with ref_angle. polar points should
    else if entry_points are given, the best point is found to be the one with
        distance from the entry_points similar to the distance defined by
        entry_points. entry_points should be in cartesian coordinates
    '''
    resolution = np.sqrt(2) / 2.0
    cocircular_points = co.pol_oper.find_cocircular_points(
        polar_points, ref_radius, resolution)
    # cocircular_points[np.abs(cocircular_points[:, 1] + pi) < 0.001, 1] *= -1
    if entry_angles is not None:
        check1 = np.abs(entry_angles[0] -
                        (cocircular_points[:, 1] + ref_angle))
        check2 = np.abs(entry_angles[1] -
                        (cocircular_points[:, 1] + ref_angle))
    else:
        if width is None:
            width = calculate_cart_dists(entry_points).squeeze()
        cocirc_cart_points = co.pol_oper.polar_to_cart(cocircular_points, ref_point,
                                                       ref_angle)
        check1 = np.abs(calculate_cart_dists(cocirc_cart_points,
                                             entry_points[0, :]) - width)
        check2 = np.abs(calculate_cart_dists(cocirc_cart_points,
                                             entry_points[1, :]) - width)
    _argmin1 = np.argmin(check1)
    _min1 = check1[_argmin1]
    _argmin2 = np.argmin(check2)
    _min2 = check2[_argmin2]
    if np.abs(_min1) < np.abs(_min2):
        corrected_pol_point = cocircular_points[_argmin1].ravel()[0:2]
    else:
        corrected_pol_point = cocircular_points[_argmin2].ravel()[0:2]
    corrected_cart_point = co.pol_oper.polar_to_cart(np.array([corrected_pol_point]), ref_point,
                                                     ref_angle).squeeze()
    return corrected_pol_point, corrected_cart_point, cocircular_points


def find_segment_to_point_box(positions, entry_segment, point):
    '''
    Returns all positions belonging to an orthogonal, which has one side equal
    to the entry segment and all the other sides are defined by the other
    point, for which it is considered that it will belong to the opposite side
    of the side that is the entry_segment.
    '''
    entry_segment = np.array(entry_segment)
    e_diff = (entry_segment[1, :] - entry_segment[0, :])
    segment_mag2 = float(np.dot(e_diff, e_diff))
    if segment_mag2 == 0:
        _lambda0 = 0
    else:
        _lambda0 = np.dot(entry_segment[1, :] -
                          entry_segment[0, :],
                          point - entry_segment[0, :]
                          ) / segment_mag2
    # perp_to_unit_e has direction from point to segment
    # p_to_seg_vec has direction from point to segment ->mi ends up negative
    # when moving positively...
    p_to_seg_vec = _lambda0 * \
        (entry_segment[1, :] - entry_segment[0, :]) + \
        entry_segment[0, :] - point
    p_to_seg_dist = np.sqrt(_lambda0 * _lambda0 * segment_mag2 +
                            np.dot(entry_segment[0, :] - point,
                                   entry_segment[0, :] - point) +
                            2 * _lambda0 * np.dot(e_diff, entry_segment[0, :] - point))
    if p_to_seg_dist <= 5:
        return np.array([[]])
    pos_diff = positions - entry_segment[0, :]
    perp_to_unit_e = p_to_seg_vec / p_to_seg_dist
    _lambda = (pos_diff[:, 0] * e_diff[0] / segment_mag2 +
               pos_diff[:, 1] * e_diff[1] / segment_mag2)
    _mi = pos_diff[:, 0] * perp_to_unit_e[0] + \
        pos_diff[:, 1] * perp_to_unit_e[1]
    tmp = -(p_to_seg_vec[0] * perp_to_unit_e[0] +
            p_to_seg_vec[1] * perp_to_unit_e[1])
    _mi_max = max([tmp, 0])
    _mi_min = min([tmp, 0])
    return (positions[(_mi < _mi_max) * (_mi > _mi_min)
                      * (_lambda < 1) * (_lambda > 0), :], perp_to_unit_e)


def calculate_chords_lengths(polar_points):
    '''
    Assuming polar points are sorted by angle
    '''
    return ((polar_points[:-1, 0] + polar_points[1:, 0]) / 2.0) * np.sqrt(2 * (
        1 - np.cos(co.pol_oper.mod_diff(polar_points[:-1, 1], polar_points[1:, 1]))))


def calculate_cart_dists(cart_points, cart_point=None):
    '''
    Input either numpy array either 2*2 list
    Second optional argument is a point
    '''
    if cart_point is None:

        try:
            return np.sqrt(
                (cart_points[1:, 0] - cart_points[:-1, 0]) *
                (cart_points[1:, 0] - cart_points[:-1, 0]) +
                (cart_points[1:, 1] - cart_points[:-1, 1]) *
                (cart_points[1:, 1] - cart_points[:-1, 1]))
        except (TypeError, AttributeError):
            return np.sqrt((cart_points[0][0] - cart_points[1][0])**2 +
                           (cart_points[0][1] - cart_points[1][1])**2)

    else:
        return np.sqrt(
            (cart_points[:, 0] - cart_point[0]) *
            (cart_points[:, 0] - cart_point[0]) +
            (cart_points[:, 1] - cart_point[1]) *
            (cart_points[:, 1] - cart_point[1]))


def find_nonzero(arr):
    '''
    Returns nonzero indices of 2D array
    '''
    return np.fliplr(cv2.findNonZero(arr).squeeze())


class LHOGE(object):

    def __init__(self, frame=None):
        self.hist_bin_size = co.CONST['HOG_bin_size']
        self.hist_range = [[0, pi]]
        self.win_overlap_ratio = 0.3
        self.frame = frame

    def grad_angles(self, img):
        '''
        Compute gradient angles on image patch for GHOG
        '''
        gradx, grady = np.gradient(img.astype(float))
        ang = np.arctan2(grady, gradx)
        ang[ang < 0] = pi + ang[ang < 0]

        return ang  # returns values 0 to pi

    def hist_data(self, sample, weights=None):
        '''
        Compute normalized N-D histograms
        '''
        if weights is None:
            weights = np.ones(sample.shape[0]) / float(sample.shape[0])
        hist, edges = np.histogramdd(sample, self.hist_bin_size,
                                     range=self.hist_range,
                                     weights=weights)
        return hist, edges

    def calculate_single(self, link):
        '''
        Compute LHOGE features for single link.
        link : points belonging to link contour
        '''
        segments = self.segment_link(link)
        img = np.zeros(self.frame.shape)
        res = np.zeros(self.frame.shape)
        count = 1
        inp = self.grad_angles(self.frame)
        segment_ent = np.zeros(segments.shape[0])
        for segment in segments:
            cv2.drawContours(img, [np.int0(segment)], 0, count, -1)
            segment_hog, _ = self.hist_data(inp[img == count])
            segment_ent[
                count - 1] = -np.sum(np.log2(segment_hog[segment_hog != 0]) * segment_hog[segment_hog != 0])
            res[img == count] = np.maximum(
                res[img == count], segment_ent[count - 1])
            count += 1
        return res

    def segment_link(self, link):
        box = cv2.boxPoints(cv2.minAreaRect(link))
        l1 = np.linalg.norm(box[0, :] - box[1, :])
        l2 = np.linalg.norm(box[1, :] - box[2, :])
        if l1 > l2:
            length = l1
            width = l2
            i11 = 0
            i21 = 1
            i12 = 3
            i22 = 2
        else:
            length = l2
            width = l1
            i11 = 0
            i21 = 3
            i12 = 1
            i22 = 2
        hog_win_len = width
        hog_overlap = width * self.win_overlap_ratio
        st_rat = np.arange(0,
                           1,
                           hog_overlap / float(length))[:, None]
        en_rat = np.arange(hog_win_len / float(length),
                           1 + hog_overlap / float(length),
                           hog_overlap / float(length))[:, None]
        en_rat = np.minimum(en_rat, 1)
        st_rat = st_rat[:en_rat.size]
        st1 = st_rat * (box[i21, :]) + (1 - st_rat) * box[i11, :]
        en1 = en_rat * (box[i21, :]) + (1 - en_rat) * box[i11, :]
        st2 = st_rat * (box[i22, :]) + (1 - st_rat) * box[i12, :]
        en2 = en_rat * (box[i22, :]) + (1 - en_rat) * box[i12, :]

        segments = np.concatenate((st1[:, None, :], en1[:, None, :], en2[
                                  :, None, :], st2[:, None, :]), axis=1)
        segments = np.maximum(np.int0(segments), 0)
        return segments

    def add_frame(self, frame):
        self.frame = frame


class HandExtractor(object):


    def get_point_depth(self, depth, x, y, tol=5):
        part = depth[max(0, y - tol): min(y + tol, depth.shape[0] - 1),
                        max(0, x - tol): min(x + tol, depth.shape[1] - 1)]
        return np.median(part[np.logical_and(part>0, np.isfinite(part))])


    def calculate_visible_arm_length(self, depth, skeleton):
        length = 0
        for link in skeleton:
            length += np.linalg.norm(
                get_real_coordinate(link[0][0], link[0][1],
                                    self.get_point_depth(depth, *link[0])) -
                get_real_coordinate(link[1][0], link[1][1],
                                    self.get_point_depth(depth, *link[1])))
        return length

    def get_point_vertical_vector(self, depth, point, patch_area=1):
        x_vec = (get_real_coordinate(point[0] + patch_area, point[1],
                                     depth[point[1], point[0] + patch_area]) -
                 get_real_coordinate(point[0] - patch_area, point[1],
                                     depth[point[1], point[0] - patch_area]))
        y_vec = (get_real_coordinate(point[0], point[1] + patch_area,
                                     depth[point[1] + patch_area, point[0]]) -
                 get_real_coordinate(point[0], point[1] - patch_area,
                                     depth[point[1] - patch_area, point[0]]))
        point_vec = np.cross(x_vec, y_vec)
        return point_vec

    def refine_skeleton(self, depth, skeleton):
        new_skeleton = []
        for link in skeleton:
            vec10 = link[0]
            vec11 = self.get_point_vertical_vector(depth, link[0])
            vec20 = link[1]
            vec21 = self.get_point_vertical_vector(depth, link[1])
            vec30 = (link[0] + link[1]) / 2
            vec31 = self.get_point_vertical_vector(depth, (link[0] + link[1]) / 2)






    def predict_joints_displacement(self, depth, skeleton):
        skeleton_joints = np.vstack((skeleton[0], skeleton[1][:-1]))
        if len(skeleton) == len(self.prev_skeleton):
            prev_skeleton_joints = np.vstack((skeleton[0], skeleton[1][:-1]))
            skeleton - prev_skeleton
        corrected_skeleton = (skeleton - prev_skeleton)
        skeleton - prev_skeleton


    def approx_hand(self, arm, skeleton, surrounding_skel):
        '''
        run longest_ray method to use this function
        '''
        arm_contour = cv2.findContours(
            arm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1][0]
        box = cv2.boxPoints(cv2.minAreaRect(arm_contour.squeeze()[
            surrounding_skel[-1], :]))[:, ::-1].astype(int)
        # find box corners closest to skeleton end
        hand_corners_inds = np.argsort(
            calculate_cart_dists(box, skeleton[-1][1]))[:2]
        hand_corners_inds = np.sort(hand_corners_inds)
        without_corn_inds = np.concatenate((box[:hand_corners_inds[0], :],
                                            box[hand_corners_inds[0] +
                                                1:hand_corners_inds[1], :],
                                            box[hand_corners_inds[1] + 1:, :]))
        far_from_hand_corn0_ind = np.argmin(calculate_cart_dists(
            without_corn_inds, box[hand_corners_inds[0], :]))
        far_from_hand_corn1_ind = np.argmin(calculate_cart_dists(
            without_corn_inds, box[hand_corners_inds[1], :]))
        width = calculate_cart_dists(box[hand_corners_inds])
        length = calculate_cart_dists(box[hand_corners_inds[0], :][None, :],
                                      without_corn_inds[far_from_hand_corn0_ind, :])
        rate = 1.7 * width / float(length)
        point_close_to_corn0 = ((1 - rate) * box[hand_corners_inds[0], :]
                                + rate * without_corn_inds[far_from_hand_corn0_ind, :]).astype(int)
        point_close_to_corn1 = ((1 - rate) * box[hand_corners_inds[1], :]
                                + rate * without_corn_inds[far_from_hand_corn1_ind, :]).astype(int)
        new_box = np.concatenate((box[hand_corners_inds, :],
                                  point_close_to_corn0[None, :],
                                  point_close_to_corn1[None, :]), axis=0)
        self.hand_mask = np.zeros(self.frame.shape, np.uint8)
        self.hand_start =  np.array([(point_close_to_corn0[0] +
                                      point_close_to_corn1[0]) / 2.0,
                                     (point_close_to_corn0[1] +
                                      point_close_to_corn1[1]) / 2.0])
        cv2.drawContours(self.hand_mask, [cv2.convexHull(
            new_box).squeeze()[:, ::-1]], 0, 1, -1)

    def run(self, arm, skeleton, surrounding_skel):
        self.approx_hand(arm, skeleton, surrounding_skel)



class FindArmSkeleton(object):
    '''
    Class to find arm skeleton and segment arm contour relatively to joints
    '''

    def __init__(self, frame=None, angle_bin_num=500, min_coords_num=50,
                 max_links=8, link_width_thres=50, draw=False, focus='speed'):
        # skeleton holds the cartesian skeleton joints
        self.skeleton = []
        # skeleton_widths is populated only with rectangle_approx method and holds the
        # segments that define each joint
        self.skeleton_widths = []
        # surrounding_skel holds the contour points that refer to each link
        self.surrounding_skel = []
        # hand_start holds the starting point of the hand, relatively to rest of the arm
        self.hand_start = None
        self.init_point = None
        self.entry = None
        self.entry_inds = None
        self.contour = None
        self.filter_mask = None
        self.frame = None
        self.img = None
        self.armpoints = None
        self.hand_mask = None
        self.polar = None
        self.draw = draw
        self.angle_bin_num = angle_bin_num
        self.min_coords_num = min_coords_num
        self.max_links = max_links
        self.link_width_thres = link_width_thres
        self.positions_initiated = True
        self.focus = focus
        if frame is not None:
            if not co.edges.exist:
                co.edges.load_calib_data(whole_im=True, img=frame)
        if co.meas.polar_positions is None:
            if frame is None:
                self.positions_initiated = False
            else:
                (self.all_cart_positions,
                 self.all_polar_positions) = co.meas.construct_positions(frame,
                                                                         polar=True)
        else:
            self.all_cart_positions = co.meas.cart_positions
            self.all_polar_positions = co.meas.polar_positions
        self.car_res = np.sqrt(2) / 2.0


    def detect_entry_upgraded(self, hull, hull_inds=None):
        '''
        Use convex hull to detect the entry of arm in the image
        '''
        if co.edges.nonconvex_edges_lims is None:
            if self.frame is None:
                raise Exception('Run reset first')
            else:
                co.edges.load_calib_data(whole_im=True, img=self.frame)

        points_in_lims_mask = ((hull[:, 0] <= co.edges.nonconvex_edges_lims[0]) +
                               (hull[:, 1] <= co.edges.nonconvex_edges_lims[1]) +
                               (hull[:, 0] >= co.edges.nonconvex_edges_lims[2]) +
                               (hull[:, 1] >= co.edges.nonconvex_edges_lims[3])).ravel()
        points = np.fliplr(hull[points_in_lims_mask, :])
        if np.size(points) == 0:
            return None
        cmplx = (points[:, 0] * 1j + points[:, 1])[:, None]
        dist_mat = np.abs(cmplx.T - cmplx)
        ind1, ind2 = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
        entry = np.array([points[ind1, :], points[ind2, :]])
        if hull_inds is not None:
            entry_inds = np.array([hull_inds[points_in_lims_mask][ind1],
                                   hull_inds[points_in_lims_mask][ind2]])
            return entry, entry_inds
        return entry

    def run(self, frame=None, contour=None, method='longest_ray'):
        '''
        method = longest_ray or rectangle_approx, default is longest_ray
        '''
        if not self.reset(frame, contour):
            return 0
        if method == 'rectangle_approx':
            self.run_rectangle_approx()
        elif method == 'longest_ray':
            return self.run_longest_ray()
        else:
            raise Exception(self.run.__doc__)
        return 1

    def reset(self, frame=None, contour=None):
        '''
        necessary function to be used before skeletonizing
        '''
        if frame is None:
            if self.frame is not None:
                frame = self.frame
            else:
                raise Exception('frame is needed')
        else:
            self.frame = frame
        if contour is None:
            raise Exception('contour is needed')
        else:
            self.contour = contour
        if self.draw:
            self.img = np.tile(frame[..., None] / float(np.max(frame)), (1, 1, 3))
        if not self.positions_initiated:
            (self.all_cart_positions,
             self.all_polar_positions) = co.meas.construct_positions(frame,
                                                                     polar=True)
        hull_inds = cv2.convexHull(contour, returnPoints=False).squeeze()
        hull_points = contour.squeeze()[hull_inds, :]
        try:
            self.entry, self.entry_inds = self.detect_entry_upgraded(
                hull_points, hull_inds)
        except:
            return 0
        self.armpoints = self.all_cart_positions[frame > 0]
        pos = self.contour.squeeze()
        self.polar = self.all_polar_positions[pos[:, 1], pos[:, 0], :]
        self.surrounding_skel = []
        self.skeleton_widths = []
        self.init_point = np.mean(self.entry, axis=0)
        self.skeleton = []
        self.filter_mask = None
        return 1

    def run_longest_ray(self, contour=None, new_init_point=None):
        '''
        function to be used to compute longest_ray method
        run self.reset first
        '''
        if contour is not None:
            self.contour = contour
        if new_init_point is not None:
            self.init_point = new_init_point
        else:
            self.init_point = np.mean(self.entry, axis=0)
        polar = self.polar.copy()
        polar = co.pol_oper.change_origin(polar, 0, [0, 0], self.init_point)
        count = 0
        while polar.shape[0] > self.min_coords_num and count < self.max_links:
            if not self.detect_longest_ray_inside_contour(
                    polar):
                break
            polar = self.polar[self.filter_mask, :]
            polar = co.pol_oper.change_origin(polar, 0,
                                              [0, 0],
                                              self.skeleton[-1][1])
            count += 1
        if not self.surrounding_skel:
            return False
        self.surrounding_skel[-1][self.filter_mask] = True
        return True

    def detect_longest_ray_inside_contour(self, polar=None,
                                          detect_width_end=True):
        '''
        polar has origin init_point
        detect_width_end is True if a rectangle approximation
        is needed, when computing the ray ends
        Run run_longest_ray to use this function
        '''
        if polar is None:
            polar = self.polar
        if self.skeleton:
            new_cart_ref_point = self.skeleton[-1][1]
        else:
            new_cart_ref_point = self.init_point
        # digitize initial polar coordinates
        r_bins_edges = np.arange(
            np.min(polar[:, 0]), np.max(polar[:, 0]) + 2, 2)
        p_bins_edges = np.linspace(np.min(polar[:, 1]), np.max(
            polar[:, 1]) + 0.01, self.angle_bin_num)
        r_d_ind = np.minimum(np.digitize(polar[:, 0],
                                  r_bins_edges),len(r_bins_edges)-1)
        p_d_ind = np.minimum(np.digitize(polar[:, 1],
                                  p_bins_edges),len(p_bins_edges)-1)

        r_d = r_bins_edges[r_d_ind]
        p_d = p_bins_edges[p_d_ind]
        # :digitized coordinates
        polar_d = np.concatenate((r_d[:, None], p_d[:, None]), axis=1)
        bins = (r_bins_edges, p_bins_edges)
        # put the digitized polar coordinates into a 2d histogram
        H, _, _ = np.histogram2d(r_d, p_d, bins=bins)
        # following might work in a future update of OpenCV (~-1ms)
        # H = cv2.calcHist([r_d.astype(np.float32),p_d.astype(np.float32)], [0,1], None, (len(r_bins_edges)-1,angle_bin_num-1),
        #                 (r_bins_edges, p_bins_edges))
        # the frequency in each bin doesn't matter (optimization might exist)
        H = H > 0
        # find the angles where a line with unique intersection does not exist
        a = np.sum(H, axis=0) > 2
        # create a matrix with zeroed all the columns refering to above angles
        s = H.copy()
        s[:, a] = 0
        if self.focus == 'speed':
            # find where the intersection happens for each non zeroed angle
            r_indices = np.argmax(s[:, :], axis=0)
            # find which intersection happens farthest. This is the result.
            y_ind = np.argmax(r_indices)
            x_ind = r_indices[y_ind]
            # find the winning point
            new_pol_ref_point = np.array(
                [[r_bins_edges[x_ind], p_bins_edges[y_ind]]])
        elif self.focus == 'accuracy':
            polpoints = polar.copy()
            #polpoints[:, 1] *= 10
            #polpoints = np.round(polpoints).astype(int)
            p_uni, polpoints[:,1] = np.unique(polpoints[:,1],
                                              return_inverse=True)
            r_uni, polpoints[:,0] = np.unique(polpoints[:,0],
                                              return_inverse=True)
            polpoints = (polpoints).astype(int)
            dists = calculate_cart_dists(polpoints)
            segm_thres = polpoints.max()/float(20)
            segments_to_cut = dists < segm_thres
            segm_inds = find_trues_segments(segments_to_cut,iscircular=False)
            segments = [np.int32(polpoints[:,::-1][start+1:end]) for [start, end] in
                        segm_inds]
            s_new = np.zeros(tuple((polpoints.max(axis=0)+1).tolist()))
            cv2.polylines(s_new, segments ,False, 1)
            s_new[0,:] = 0
            #ref_p = np.min(polpoints, axis=0)
            #polpoints = polpoints
            #s_new[polpoints[:,0], polpoints[:,1]] = 1
            '''
            check = s>0
            h_points = find_nonzero(check.astype(np.uint8))
            d = np.abs(h_points - np.median(h_points,axis=0))
            mdev = np.median(d,axis=0)
            mdev[mdev==0] = 0.01
            rat = d / mdev.astype(float) <= 2
            h_points = h_points[np.prod(rat,axis=1).astype(bool)]
            print h_points
            s_new = np.zeros_like(s)
            s_new[h_points[:,0],h_points[:,1]] = s[h_points[:,0],h_points[:,1]]
            '''
            s = s_new
            # find where the intersection happens for each non zeroed angle
            r_indices = np.argmax(s[:, :], axis=0)
            # find which intersection happens farthest. This is the result.
            y_ind = np.argmax(r_indices)
            x_ind = r_indices[y_ind]
            dists = calculate_cart_dists(polpoints,
                                         np.hstack((x_ind,y_ind)))
            new_pol_ref_point = polar[np.argmin(dists), :][None, :]
        if self.draw:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            axes = fig.add_subplot(111)
            #im = (H[...,None]*np.array([[[1,0,0]]])+s[...,None]*np.array([[[0,1,0]]]))
            im = (s[...,None]*np.array([[[0,1,0]]]))
            im = im / np.max(im).astype(float)
            cv2.circle(im, (y_ind, x_ind), 3, [1,1,1], -1)
            axes.imshow(im)
            #plt.hist2d(p_d,r_d, bins=(p_bins_edges, r_bins_edges))
            plt.show()
        old_cart_ref_point = new_cart_ref_point.copy()
        new_cart_ref_point = co.pol_oper.polar_to_cart(
            new_pol_ref_point, old_cart_ref_point, 0).squeeze()
        if detect_width_end:
            # detecting the second point to define segment end
            mask1 = (np.abs(new_pol_ref_point[0, 0] - polar[:, 0])
                     <= 2)
            mask2 = (np.abs(new_pol_ref_point[0, 1] - polar[:, 1])
                     <= 2 * pi / float(self.angle_bin_num))
            link_end_1st_ind = (mask1 * mask2).argmax()
            if self.skeleton:
                ref_point = self.skeleton[-1][1]
            else:
                ref_point = self.init_point
            self.detect_link_end_by_distance(ref_point, 0, link_end_1st_ind,
                                             polar)
            if calculate_cart_dists(np.array(self.skeleton_widths[-1][1])) < 10:
                return 0
        else:
            # no second windth end point is detected, link points are considered the ones
            # having already been rejected during ray computation
            # those are:
            # surr_points_r_ind = np.argmax(H[:x_ind,:],axis=0)
            # However, this result can not be considered safe choice as the
            # contour is not smooth and the angle bin size should get smaller,
            # thus the procedure will be lower. So detect_width_end is by
            # default true.
            H_part_argsort = np.argpartition(H[:x_ind, :], -2, axis=0)
            surr_points_r_ind = H_part_argsort[-2:, :]
            surr_points_p_ind = np.arange(H.shape[1])
            r_mask1 = r_d_ind[:, None] == surr_points_r_ind[0, :][None, :]
            p_mask = p_d_ind[:, None] == surr_points_p_ind[None, :]
            r_mask2 = r_d_ind[:, None] == surr_points_r_ind[1, :][None, :]
            link_points_mask = np.sum(np.logical_and(np.logical_or(
                r_mask1, r_mask2), p_mask), axis=1).astype(bool)
            closing_size = co.CONST['longest_ray_closing_size']
            # We perform a closing operation in 1d space, that might fix some
            # irregularities, although the results are again disputable
            # dilate mask to gather outliers
            for count in xrange(closing_size):
                link_points_mask += (np.roll(link_points_mask, 1) +
                                     np.roll(link_points_mask, -1))
            # erode mask to return it to the previous condition
            for count in xrange(closing_size):
                link_points_mask *= (np.roll(link_points_mask, 1) *
                                     np.roll(link_points_mask, -1))
            _, link_points_mask = find_largest_trues_segment(link_points_mask)

            if self.filter_mask is not None:
                surr_mask = self.filter_mask.copy()
                surr_mask[self.filter_mask > 0] = link_points_mask
                self.surrounding_skel.append(surr_mask)
                self.filter_mask[
                    self.filter_mask > 0] = np.logical_not(link_points_mask)
            else:
                self.surrounding_skel.append(link_points_mask)
                self.filter_mask = np.logical_not(link_points_mask)
            self.skeleton.append([np.int0(old_cart_ref_point),
                                  np.int0(new_cart_ref_point)])
        return 1

    def run_rectangle_approx(self):
        '''
        function to run rectangle_approx method
        run self.reset first
        '''
        used_polar_size = self.polar.shape[0]
        while (len(self.skeleton) <= self.max_links and
               used_polar_size > 10):
            if not self.rectangle_approx_single_link(self.polar):
                break
            used_polar_size = self.filter_mask.sum()

    def find_used_polar(self):
        if self.filter_mask is None:
            used_polar = self.polar.copy()
        else:
            used_polar = self.polar[self.filter_mask, :]
        return used_polar

    def detect_link_end_by_distance(self, ref_point, ref_angle,
                                    link_end_1st_ind,
                                    used_polar,
                                    entry_segment=None,
                                    entry_inds=None):
        # finding the direction (1 or -1) of the circular vector with the closest distance
        # from the width_segment point to the link_end_1st_ind
        # (=w_to_crit_direction)
        if entry_inds is None:
            if not self.skeleton:
                if self.entry is not None:
                    entry_segment = self.entry
                    entry_inds = self.entry_inds
                else:
                    raise Exception('entry_segment and entry_inds are needed')
            else:
                entry_inds = np.array([0, self.filter_mask.sum()])

        if entry_segment is None:
            if self.skeleton_widths:
                entry_segment = self.skeleton_widths[-1][1]
            else:
                raise Exception('entry_segment is needed')
        tmp_closest_to_1st_end_width_ind = co.circ_oper.diff(np.array([link_end_1st_ind]),
                                                             entry_inds,
                                                             used_polar.shape[
                                                                 0],
                                                             no_intersections=True)
        closest_to_1st_end_width_ind = entry_inds[
            tmp_closest_to_1st_end_width_ind]
        closest_to_2nd_end_width_ind = entry_inds[
            1 - tmp_closest_to_1st_end_width_ind]

        w_to_crit_direction, _ = co.circ_oper.find_min_dist_direction(
            closest_to_1st_end_width_ind,
            link_end_1st_ind,
            used_polar.shape[0])
        # from here are the changes
        link_end_1st = used_polar[link_end_1st_ind, :]
        # The other point that sets the end of the link is
        # assumed to be cocircular (rectangle approximation) with the
        # link_end_1st.
        length_radius = link_end_1st[0]
        cocircular_with_crit_mask = np.abs(
            used_polar[:, 0] - length_radius) < self.car_res
        if self.draw:
            cv2.circle(self.img, tuple(ref_point.astype(int)[::-1]), int(length_radius),
                       [0.5,0.0,0.0])
        try:
            cocircular_with_crit_inds = cv2.findNonZero(cocircular_with_crit_mask[:, None]
                                                        .astype(np.uint8))[:, 0, 1]
        except TypeError:
            return 0
        # from the cocircular points we keep the one, which is closest to the
        # unused width point index ( not the one closer to link_end_1st_ind),
        # from the opposite direction of w_to_crit_direction

        #####folowing for should be optimized######
        dists = np.zeros_like(cocircular_with_crit_inds)
        for count in xrange(cocircular_with_crit_inds.size):
            dists[count] = co.circ_oper.find_single_direction_dist(
                closest_to_2nd_end_width_ind,
                cocircular_with_crit_inds[count],
                used_polar.shape[0], -w_to_crit_direction)
        link_end_2nd_ind = cocircular_with_crit_inds[dists.argmin()]
        link_end_2nd = used_polar[link_end_2nd_ind, :]
        link_end_width_segment = np.zeros((2, 2))
        link_end_width_segment[0, :] = co.pol_oper.polar_to_cart(
            np.array([link_end_1st]), ref_point, ref_angle).squeeze()
        link_end_width_segment[1, :] = co.pol_oper.polar_to_cart(
            np.array([link_end_2nd]), ref_point, ref_angle).squeeze()
        skel_end = np.mean(link_end_width_segment, axis=0)
        if self.skeleton:
            link_length = calculate_cart_dists(
                np.atleast_2d(self.skeleton[-1][1]), skel_end)
        if self.skeleton:
            if link_length < self.link_width_thres:
                self.skeleton[-1][1] = skel_end.astype(int)
                self.skeleton_widths[-1][
                    1] = link_end_width_segment.astype(int)
            else:
                self.skeleton.append([self.skeleton[-1][1],
                                      np.mean(link_end_width_segment, axis=0).astype(int)])
                self.skeleton_widths.append([self.skeleton_widths[-1][1],
                                             link_end_width_segment.astype(int)])
        else:
            self.skeleton.append([np.mean(entry_segment, axis=0).astype(int),
                                  np.mean(link_end_width_segment, axis=0).astype(int)])
            self.skeleton_widths.append([entry_segment.astype(int),
                                         link_end_width_segment.astype(int)])
        surr_points_mask = np.ones(used_polar.shape[0])
        # changes from here
        filter1 = co.circ_oper.filter(closest_to_1st_end_width_ind,
                                      link_end_1st_ind,
                                      used_polar.shape[0],
                                      w_to_crit_direction)
        filter2 = co.circ_oper.filter(closest_to_2nd_end_width_ind,
                                      link_end_2nd_ind,
                                      used_polar.shape[0],
                                      -w_to_crit_direction)
        surr_points_mask = np.logical_xor(filter1, filter2).astype(bool)
        if self.filter_mask is None:
            # entry_segment must be also taken care of
            dir3, _ = co.circ_oper.find_min_dist_direction(entry_inds[0],
                                                           entry_inds[1],
                                                           used_polar.shape[0])
            filter3 = co.circ_oper.filter(entry_inds[0], entry_inds[1],
                                          used_polar.shape[0], dir3)
            surr_points_mask = np.logical_or(surr_points_mask, filter3)
            self.surrounding_skel.append(surr_points_mask)
            self.filter_mask = np.logical_not(surr_points_mask)
        else:
            surr_mask = self.filter_mask.copy()
            surr_mask[self.filter_mask > 0] = surr_points_mask
            if link_length < self.link_width_thres:
                self.surrounding_skel[-1] += surr_mask
                self.surrounding_skel[
                    -1] = self.surrounding_skel[-1].astype(bool)
            else:
                self.surrounding_skel.append(surr_mask.astype(bool))
            self.filter_mask[self.filter_mask >
                             0] = np.logical_not(surr_points_mask)
        self.filter_mask = self.filter_mask.astype(bool)

        return 1

    def rectangle_approx_single_link(self, polar, entry_inds=None, entry_segment=None,
                                     polar_ref_point=[0, 0], polar_ref_angle=0):
        '''
        Use rectangle approximation of the links to find the skeleton
        '''

        if entry_inds is None:
            if not self.skeleton:
                if self.entry is not None:
                    entry_segment = self.entry
                    entry_inds = self.entry_inds
                else:
                    raise Exception('entry_segment and entry_inds are needed')
            else:
                entry_inds = np.array([0, self.filter_mask.sum()])

        if entry_segment is None:
            if self.skeleton_widths:
                entry_segment = self.skeleton_widths[-1][1]
            else:
                raise Exception('entry_segment is needed')
        if self.filter_mask is None:
            used_polar = polar.copy()
        else:
            used_polar = polar[self.filter_mask, :]
        ##############################################
        #Making a guess of the rectangle orientation,#
        #enclosing the first link, by examining the  #
        #points near the entry_segment               #
        ##############################################
        new_ref_point = [(entry_segment[0][0] + entry_segment[1][0]) /
                         2.0, (entry_segment[0][1] + entry_segment[1][1]) / 2.0]
        used_polar = co.pol_oper.change_origin(
            used_polar, polar_ref_angle, polar_ref_point, new_ref_point)
        new_ref_radius = calculate_cart_dists(entry_segment) / 2.0
        width_vec = entry_segment[0, :] - entry_segment[1, :]
        width_vec_orient = np.arctan2(width_vec[0], width_vec[1])

        angle1 = co.pol_oper.fix_angle(width_vec_orient + pi / 2)
        angle2 = co.pol_oper.fix_angle(width_vec_orient - pi / 2)
        try:
            (_, corrected_cart_point, _) = find_corrected_point(
                used_polar, polar_ref_angle, new_ref_point, new_ref_radius, [angle1, angle2])
            box, perp_to_segment_unit = find_segment_to_point_box(
                self.armpoints, entry_segment, np.array(corrected_cart_point))
            new_ref_angle, corrected_entry_segment = find_link_direction(
                box, entry_segment, perp_to_segment_unit, np.array(corrected_cart_point))
        except ValueError:
            seg_len = calculate_cart_dists(entry_segment)[0]
            seg_angle = np.arctan2(width_vec[0], width_vec[1])
            angle1 = co.pol_oper.fix_angle(seg_angle - pi / 2)
            angle2 = co.pol_oper.fix_angle(seg_angle + pi / 2)
            comp_angle = used_polar[
                np.argmin(used_polar[(used_polar[:, 0] - seg_len / 2.0) > 2, 0]), 1]

            angdiff1 = co.pol_oper.mod_diff(comp_angle, angle1)
            angdiff2 = co.pol_oper.mod_diff(comp_angle, angle2)
            if np.abs(angdiff1) < np.abs(angdiff2):
                new_ref_angle = angle1
            else:
                new_ref_angle = angle2

        used_polar[:, 1] -= (new_ref_angle)
        co.pol_oper.mod_correct(used_polar)
        intersection_mask = (np.abs(used_polar[:, 1]) <
                             co.CONST['angle_resolution'])
        if np.sum(intersection_mask) == 0:
            # No intersection found, must have reached the end on hand
            self.surrounding_skel[-1][self.filter_mask] = True
            return 0
        # finding closest to entry_segment intersection points
        cand_crit_points_inds = cv2.findNonZero(intersection_mask[:, None]
                                                .astype(np.uint8))[:, 0, 1]

        comp = co.circ_oper.diff(cand_crit_points_inds,
                                 entry_inds,
                                 intersection_mask.size)
        cand_ind, closest_to_cand_width_point_ind = np.unravel_index(
            np.argmin(comp), comp.shape)
        # saving closest one's position (=link_end_1st_ind)
        link_end_1st_ind = cand_crit_points_inds[cand_ind]
        # finding the direction (1 or -1) of the circular vector with the closest distance
        # from the width_segment point to the link_end_1st_ind
        # (=w_to_crit_direction)
        closest_to_1st_end_width_ind = entry_inds[
            closest_to_cand_width_point_ind]
        w_to_crit_direction, _ = co.circ_oper.find_min_dist_direction(
            closest_to_1st_end_width_ind,
            link_end_1st_ind,
            intersection_mask.size)
        link_end_1st = used_polar[link_end_1st_ind, :]
        # The other point that sets the end of the link is
        # assumed to be cocircular (rectangle approximation) with the
        # link_end_1st.
        length_radius = link_end_1st[0]
        cocircular_with_crit_mask = np.abs(
            used_polar[:, 0] - length_radius) < self.car_res
        cocircular_with_crit_inds = cv2.findNonZero(cocircular_with_crit_mask[:, None]
                                                    .astype(np.uint8))[:, 0, 1]
        # from the cocircular points we keep the ones , whose vectors of minimum distance from
        # the width_segment point to the points have different direction from the direction
        # of the corresponding one from the width_segment_point to the
        # link_end_1st_ind
        crit_inds_rev_dir_mask = np.zeros_like(
            cocircular_with_crit_inds)

        #####folowing for should be optimized######
        dists = np.zeros_like(cocircular_with_crit_inds)
        for count in xrange(cocircular_with_crit_inds.size):
            (crit_inds_rev_dir_mask[count],
             dists[count]) = co.circ_oper.find_min_dist_direction(
                 entry_inds[1 - closest_to_cand_width_point_ind],
                 cocircular_with_crit_inds[count],
                 intersection_mask.size)
        crit_inds_rev_dir_mask = (
            crit_inds_rev_dir_mask != w_to_crit_direction)
        # if there are no such cocircular points then we keep the cocircular point,
        # whose minimum distance is furthest from the width_segment_points
        if np.sum(crit_inds_rev_dir_mask) == 0:
            crit_inds_rev_dir_mask[dists.argmax()] = True
        # if there are more than one passing points, then keep the one closest to the
        # width_segment_points
        comp = co.circ_oper.diff(cocircular_with_crit_inds[
            crit_inds_rev_dir_mask],
            entry_inds,
            intersection_mask.size)
        cand_ind, closest_to_cand_width_point_ind = np.unravel_index(
            np.argmin(comp), comp.shape)
        closest_to_2nd_end_width_ind = entry_inds[
            closest_to_cand_width_point_ind]
        # link_end_2nd_ind is the resulting second point
        link_end_2nd_ind = cocircular_with_crit_inds[
            crit_inds_rev_dir_mask][cand_ind]
        link_end_2nd = used_polar[link_end_2nd_ind, :]
        link_end_width_segment = np.zeros((2, 2))
        link_end_width_segment[0, :] = co.pol_oper.polar_to_cart(
            np.array([link_end_1st]), new_ref_point, new_ref_angle).squeeze()
        link_end_width_segment[1, :] = co.pol_oper.polar_to_cart(
            np.array([link_end_2nd]), new_ref_point, new_ref_angle).squeeze()
        if self.skeleton:
            self.skeleton.append([self.skeleton[-1][1],
                                  np.mean(link_end_width_segment, axis=0).astype(int)])
            self.skeleton_widths.append([self.skeleton_widths[-1][1],
                                         link_end_width_segment.astype(int)])
        else:
            self.skeleton.append([np.mean(entry_segment, axis=0).astype(int),
                                  np.mean(link_end_width_segment, axis=0).astype(int)])
            self.skeleton_widths.append([entry_segment.astype(int),
                                         link_end_width_segment.astype(int)])
        surr_points_mask = np.ones(used_polar.shape[0])
        dir_1st, _ = co.circ_oper.find_min_dist_direction(
            closest_to_1st_end_width_ind, link_end_1st_ind, used_polar.shape[0])
        dir_2nd, _ = co.circ_oper.find_min_dist_direction(
            closest_to_2nd_end_width_ind, link_end_2nd_ind, used_polar.shape[0])
        filter1 = co.circ_oper.filter(closest_to_1st_end_width_ind, link_end_1st_ind,
                                      used_polar.shape[0], dir_1st)
        filter2 = co.circ_oper.filter(closest_to_2nd_end_width_ind, link_end_2nd_ind,
                                      used_polar.shape[0], dir_2nd)
        surr_points_mask = np.logical_or(filter1, filter2).astype(bool)
        if self.filter_mask is None:
            self.surrounding_skel.append(surr_points_mask)
            self.filter_mask = np.logical_not(surr_points_mask)
        else:
            surr_mask = self.filter_mask.copy()
            surr_mask[self.filter_mask > 0] = surr_points_mask
            self.surrounding_skel.append(surr_mask.astype(bool))
            self.filter_mask[self.filter_mask >
                             0] = np.logical_not(surr_points_mask)
        self.filter_mask = self.filter_mask.astype(bool)

        return 1

    def draw_skeleton(self, frame, show=True):
        '''
        draws links and skeleton on frame
        '''
        if show:
            from matplotlib import pyplot as plt
        if not self.draw:
            self.img = np.tile(frame[..., None] / float(np.max(frame)), (1, 1, 3))
        c_copy = self.contour.squeeze()
        for surr_points_mask in self.surrounding_skel:
            colr = np.random.random(3)
            surr_points = c_copy[surr_points_mask, :]
            for count in range(surr_points.shape[0]):
                cv2.circle(self.img, tuple(surr_points[count, :]),
                           1, colr.astype(tuple), -1)
        for link in self.skeleton:
            cv2.arrowedLine(self.img, tuple(
                link[0][::-1]), tuple(link[1][::-1]), [0, 0, 1], 2)
        if show:
            plt.figure()
            plt.imshow(self.img)
            plt.show()
        return (self.img * np.max(frame))


def main():
    '''Main Caller Function'''
    if not os.path.exists('arm_example.png'):
        urllib.urlretrieve("https://www.packtpub.com/\
                           sites/default/files/Article-Images/B04521_02_04.png",
                           "arm_example.png")
    # binarm3d = cv2.imread('random.png')
    # binarm3d = cv2.imread('arm_example.png')
    #binarm = cv2.imread('random.png', -1)
    binarm = cv2.imread('arm_example.png', -1)
    _, cnts, _ = cv2.findContours(
        binarm.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    co.edges.load_calib_data(whole_im=True, img=binarm)
    #skel = FindArmSkeleton(binarm)
    # binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
    #    binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
    # co.edges.calib_edges = np.pad(np.zeros((binarm3d.shape[
    #    0] - 2, binarm3d.shape[1] - 2), np.uint8), ((1, 1), (1, 1)), 'constant', constant_values=1)
    # co.edges.find_non_convex_edges_lims(edge_tolerance=1)
    # print\
    #    timeit.timeit(lambda: main_process_upgraded(binarm3d.copy(),
    #                                                binarm3d_positions, 0), number=100) / 100
    # print\
    #    timeit.timeit(lambda: skel.run(binarm,cnts[0]), number=100) / 100

    # profile.runctx('main_process_upgraded(binarm3d,binarm3d_positions,0)',
    #               globals(), locals())
    #profile.runctx('skel.run(binarm, cnts[0])', globals(), locals())
    for _ in range(4):
        # rows, cols, _ = binarm3d.shape
        rows, cols = binarm.shape
        rot_mat = cv2.getRotationMatrix2D(
            (np.floor(cols / 2.0), np.floor(rows / 2.0)), 90, 1)
        rot_mat[0, 2] += np.floor(rows / 2.0 - cols / 2.0)
        rot_mat[1, 2] += np.floor(cols / 2.0 - rows / 2.0)
        # binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
        #    binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
        skel = FindArmSkeleton(binarm, link_width_thres=0, draw=True,
                               focus='accuracy')
        _, cnts, _ = cv2.findContours(
            binarm.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        skel.run(binarm, cnts[0], 'longest_ray')
        skel.draw_skeleton(binarm)
        #_, _, res = main_process_upgraded(
        #    binarm3d.copy(), binarm3d_positions, 1)
        # if res is not None:
        #    cv2.imshow('res', res[0])
        #    cv2.waitKey(0)
        # binarm3d = 255 * \
        #    ((cv2.warpAffine(binarm3d, rot_mat, (rows, cols))) > 0).astype(np.uint8)
        binarm = 255 *\
            ((cv2.warpAffine(binarm, rot_mat, (rows, cols))) > 0).astype(np.uint8)
        co.meas.construct_positions(binarm,
                                    polar=True)
        # co.edges.calib_edges = np.pad(np.zeros((
        #    binarm3d.shape[0] - 2, binarm3d.shape[1] - 2), np.uint8), (
        #        (1, 1), (1, 1)), 'constant', constant_values=1)
        co.edges.load_calib_data(whole_im=True, img=binarm)
        # co.edges.find_non_convex_edges_lims(edge_tolerance=1)
    # cv2.destroyAllWindows

LOG = logging.getLogger('__name__')
CH = logging.StreamHandler()
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel('INFO')
if __name__ == '__main__':
    main()
#####################################################################
#####################################################################
############################################################
###OBSOLETE:###
# define obs_skeleton for backwards compatibility
obs_skeleton = FindArmSkeleton()


def detect_entry(bin_mask):
    '''
    Function to detect  intersection limits of mask with calibration edges
    Assuming non_convex calib edges..
    '''
    # entry_segments=positions[bin_mask*co.masks.calib_edges>0]
    entry_segments = co.edges.edges_positions[
        bin_mask[co.edges.edges_positions_indices] > 0]
    if entry_segments.shape[0] == 0:
        return None
    approx_entry_segments = entry_segments.copy()
    approx_entry_segments[
        entry_segments[:, 1] <
        co.edges.nonconvex_edges_lims[0], 1] = co.edges.nonconvex_edges_lims[0]
    approx_entry_segments[
        entry_segments[:, 0] <
        co.edges.nonconvex_edges_lims[1], 0] = co.edges.nonconvex_edges_lims[1]
    approx_entry_segments[
        entry_segments[:, 1] >
        co.edges.nonconvex_edges_lims[2], 1] = co.edges.nonconvex_edges_lims[2]
    approx_entry_segments[
        entry_segments[:, 0] >
        co.edges.nonconvex_edges_lims[3], 0] = co.edges.nonconvex_edges_lims[3]
    approx_entry_points = cv2.convexHull(approx_entry_segments).squeeze()
    not_approx = 0
    if approx_entry_points.size == 2:
        not_approx = 1
        approx_entry_points = cv2.convexHull(entry_segments).squeeze()
    if approx_entry_points.shape[0] == 2:
        try:
            if calculate_cart_dists(approx_entry_points) > np.min(bin_mask.shape) / 10.0:
                return entry_segments[find_rows_in_array(
                    approx_entry_segments, approx_entry_points)]
            else:
                return np.array([])
        except:
            print('lims', co.edges.nonconvex_edges_lims)
            print('hull', approx_entry_points)
            print('aprox segments', approx_entry_segments)
            print('segments', entry_segments)
            raise
    approx_entry_orient = np.diff(approx_entry_points, axis=0)
    try:
        approx_entry_orient = (approx_entry_orient /
                               calculate_cart_dists(
                                   approx_entry_points)[:, None])
    except:
        print(approx_entry_points)
        raise
    approx_entry_vert_orient = np.dot(
        approx_entry_orient, np.array([[0, -1], [1, 0]]))
    num = []
    for count, orient in enumerate(approx_entry_vert_orient):
        if not_approx:
            pos = find_segment_to_point_box(entry_segments,
                                            np.array([approx_entry_points[count, :] + orient * 10,
                                                      approx_entry_points[count, :] - orient * 10]),
                                            approx_entry_points[count + 1, :])[0]

        else:
            pos = find_segment_to_point_box(approx_entry_segments,
                                            np.array([approx_entry_points[count, :] + orient * 10,
                                                      approx_entry_points[count, :] - orient * 10]),
                                            approx_entry_points[count + 1, :])[0]
        num.append(pos.shape[0])
    _argmax = np.argmax(num)
    if not_approx:
        entry_points = entry_segments[find_rows_in_array(
            entry_segments, approx_entry_points[_argmax:_argmax + 2, :])]
    else:
        entry_points = entry_segments[find_rows_in_array(
            approx_entry_segments, approx_entry_points[_argmax:_argmax + 2, :])]
    return entry_points


def main_process_upgraded(binarm3d, positions=None, display=0):
    if len(binarm3d.shape) == 3:
        binarm = binarm3d[:, :, 0].copy()
        if np.max(binarm3d) != 255:
            binarm3d = (binarm3d / float(np.max(binarm3d))) * 255
        binarm3d = binarm3d.astype(np.uint8)
    else:
        binarm = binarm3d.copy()
        if display == 1:
            binarm3d = np.tile(binarm[:, :, None], (1, 1, 3))
            if np.issubdtype(binarm3d[0, 0, 0], np.uint8):
                if np.max(binarm3d) == 1:
                    binarm3d *= 255
            else:
                binarm3d = (255 * binarm3d).astype(np.uint8)
    if positions is None:
        if co.meas.cart_positions is None:
            co.meas.construct_positions(binarm, polar=True)
            obs_skeleton.positions_initiated = True
        positions = co.meas.cart_positions
    _, cnts, _ = cv2.findContours(
        binarm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hand_patches, hand_patches_pos, masks = (None, None, None)
    cnt_count = 0
    for cnt in cnts:
        if cv2.contourArea(cnt) < co.CONST['min_area']:
            continue
        else:
            cnt_count += 1
        if cnt_count > co.CONST['max_hsa_contours_num']:
            break
        hull_inds = cv2.convexHull(cnt, returnPoints=False)
        hull_pnts = cnt[hull_inds.squeeze()].squeeze()
        ref_col, ref_row, col_num, row_num = cv2.boundingRect(cnt)
        img = np.zeros((row_num, col_num), np.uint8)
        cnt -= np.array([[ref_col, ref_row]])
        cv2.drawContours(img, [cnt], 0, 255, -1)
        obs_skeleton.detect_entry_upgraded(hull_pnts)
        entry = obs_skeleton.entry
        if entry is None:
            LOG.debug('No entry found')
            co.chhm.no_entry += 1
            if display == 1:
                tag_im(binarm3d, 'No entry found')
                co.im_results.images.append(binarm3d)
            return None, None, None
        entry -= np.array([[ref_row, ref_col]])
        hand_patch, hand_patch_pos, mask = main_process(img, entry=entry,
                                                        positions=positions[ref_row:ref_row + row_num,
                                                                            ref_col:ref_col + col_num, :] -
                                                        np.array([[ref_row, ref_col]]), display=display)
        img = None
        if hand_patch is not None:
            hand_patch_pos += np.array([ref_row, ref_col])
            img = np.zeros(binarm.shape)
            img[hand_patch_pos[0]:hand_patch_pos[0] + hand_patch.shape[0],
                hand_patch_pos[1]:hand_patch_pos[1] + hand_patch.shape[1]] = hand_patch
            try:
                hand_patches.append(hand_patch)
            except:
                hand_patches, hand_patches_pos, masks = ([], [], [])
                hand_patches.append(hand_patch)
            hand_patches_pos.append(hand_patch_pos)
            masks.append(img)
    return hand_patches, hand_patches_pos, masks


def main_process(binarm3d, positions=None, display=0, entry=None):
    '''Main processing function'''
    LOG.setLevel('INFO')
    if len(binarm3d.shape) == 3:
        binarm = binarm3d[:, :, 0].copy()
        if np.max(binarm3d) != 255:
            binarm3d = (binarm3d / float(np.max(binarm3d))) * 255
        binarm3d = binarm3d.astype(np.uint8)
    else:
        binarm = binarm3d.copy()
        if display == 1:
            binarm3d = np.tile(binarm[:, :, None], (1, 1, 3))
            if np.issubdtype(binarm3d[0, 0, 0], np.uint8):
                if np.max(binarm3d) == 1:
                    binarm3d *= 255
            else:
                binarm3d = (255 * binarm3d).astype(np.uint8)
    if positions is None:
        if co.meas.cart_positions is None:
            co.meas.construct_positions(binarm)
        positions = co.meas.cart_positions
    try:
        armpoints = find_nonzero(binarm)
    except AttributeError:  # binarm is []
        LOG.debug('No objects found')
        co.chhm.no_obj += 1
        if display == 1:
            tag_im(binarm3d, 'No object found')
            co.im_results.images.append(binarm3d)
        return None, None, None

    try:
        points = find_nonzero(with_laplacian(binarm))
    except AttributeError:
        return None, None, None
    points = points[:, 0] * 1j + points[:, 1]
    tmp = np.angle(points)
    tmp[tmp < -pi] += 2 * pi
    tmp[tmp > pi] -= 2 * pi
    new_polar = np.concatenate(
        (np.absolute(points)[:, None], tmp[:, None]), axis=1)
    if entry is None:
        entry = detect_entry(binarm)
    '''
    except IndexError:
        LOG.debug('No entry found')
        if display == 1:
            tag_im(binarm3d, 'No entry found')
            co.im_results.images.append(binarm3d)
        return None, None, None
    '''
    if entry is None:
        LOG.debug('No entry found')
        co.chhm.no_entry += 1
        if display == 1:
            tag_im(binarm3d, 'No entry found')
            co.im_results.images.append(binarm3d)
        return None, None, None
    if entry.shape[0] <= 1:
        LOG.debug('Arm in image corners or its entry is occluded' +
                      ', hand segmentation algorithm cannot function')
        co.chhm.in_im_corn += 1
        if display == 1:
            tag_im(binarm3d, 'Arm in image corners or its entry is occluded' +
                   ', hand segmentation algorithm cannot function')
            co.im_results.images.append(binarm3d)
        return None, None, None

    link_end_radius = 1 / 2.0 * calculate_cart_dists(entry)
    link_end_segment = entry
    new_ref_point = [0, 0]
    new_ref_angle = 0
    new_crit_ind = 0
    link_end_2nd = []
    resolution = np.sqrt(2) / 2.0
    new_corrected_segment = entry[:]
    # for _count in range(3):
    link_count = 0
    while True:
        link_count += 1
        prev_ref_point = new_ref_point[:]
        prev_polar = new_polar[:]
        prev_ref_angle = new_ref_angle
        prev_ref_point = new_ref_point[:]
        if (new_crit_ind > new_polar.shape[0] - 10 or
                link_count > co.CONST['max_link_number']):
            LOG.debug('Reached Mask Limits')
            co.chhm.rchd_mlims += 1
            if display == 1:
                tag_im(binarm3d, 'Reached Mask Limits')
                co.im_results.images.append(binarm3d)
            return None, None, None

        new_ref_point = [(link_end_segment[0][0] + link_end_segment[1][0]) /
                         2.0, (link_end_segment[0][1] + link_end_segment[1][1]) / 2.0]
        new_polar = co.pol_oper.change_origin(
            prev_polar.copy(), prev_ref_angle, prev_ref_point, new_ref_point)
        new_ref_radius = calculate_cart_dists(link_end_segment) / 2.0
        link_end_diff = link_end_segment[0, :] - link_end_segment[1, :]
        tmpangle = np.arctan2(link_end_diff[0], link_end_diff[1])
        angle1 = co.pol_oper.fix_angle(tmpangle + pi / 2)
        angle2 = co.pol_oper.fix_angle(tmpangle - pi / 2)
        try:
            (_, corrected_cart_point, _) = find_corrected_point(
                new_polar, 0, new_ref_point, new_ref_radius, [angle1, angle2])
            box, perp_to_segment_unit = find_segment_to_point_box(
                armpoints, link_end_segment, np.array(corrected_cart_point))
            if display == 1:
                binarm3d = picture_box(binarm3d, box)
            new_ref_angle, new_corrected_segment = find_link_direction(
                box, link_end_segment, perp_to_segment_unit, np.array(corrected_cart_point))
        except ValueError:
            seg_len = calculate_cart_dists(link_end_segment)[0]
            seg_angle = np.arctan2(link_end_diff[
                0], link_end_diff[1])
            angle1 = co.pol_oper.fix_angle(seg_angle - pi / 2)
            angle2 = co.pol_oper.fix_angle(seg_angle + pi / 2)
            try:
                comp_angle = new_polar[
                    np.argmin(new_polar[(new_polar[:, 0] - seg_len / 2.0) > 2, 0]), 1]
            except ValueError:
                return None, None, None

            angdiff1 = co.pol_oper.mod_diff(comp_angle, angle1)
            angdiff2 = co.pol_oper.mod_diff(comp_angle, angle2)
            if np.abs(angdiff1) < np.abs(angdiff2):
                new_ref_angle = angle1
            else:
                new_ref_angle = angle2
        new_polar[:, 1] -= (new_ref_angle)
        co.pol_oper.mod_correct(new_polar)
        new_polar = new_polar[new_polar[:, 0] >= new_ref_radius, :]
        new_polar = new_polar[new_polar[:, 0].argsort(), :]
        if display == 1:
            tmp = co.pol_oper.polar_to_cart(
                new_polar, new_ref_point, new_ref_angle)
            '''
            binarm3d[tuple(tmp[np.abs(np.sqrt((tmp[:, 0] - new_ref_point[0])**2
                                              + (tmp[:, 1] - new_ref_point[1])**2)
                                      - new_polar[new_crit_ind, 0]) <=
                               resolution].T)] = [255, 0, 0]
            '''
            binarm3d[link_end_segment[0][0],
                     link_end_segment[0][1]] = [0, 0, 255]
            binarm3d[link_end_segment[1][0],
                     link_end_segment[1][1]] = [0, 0, 255]
            cv2.line(binarm3d, (new_corrected_segment[0][1], new_corrected_segment[0][0]),
                     (new_corrected_segment[1][1], new_corrected_segment[1][0]), [0, 0, 255])
        cand_crit_points = new_polar[np.abs(new_polar[:, 1]) < co.CONST[
            'angle_resolution'], :]
        if len(cand_crit_points) == 0:
            LOG.debug('No cocircular points found,' +
                          ' reached end of hand')
            co.chhm.no_cocirc += 1
            if display == 1:
                tag_im(binarm3d, 'No cocircular points found,' +
                       ' reached end of hand')
                co.im_results.images.append(binarm3d)
            return None, None, None

        _min = cand_crit_points[0, :]
        new_crit_ind = np.where(new_polar == _min)[0][0]

        cocircular_crit = co.pol_oper.find_cocircular_points(new_polar,
                                                             new_polar[
                                                                 new_crit_ind, 0],
                                                             resolution)
        cocircular_crit = cocircular_crit[cocircular_crit[:, 1].argsort(), :]
        crit_chords = calculate_chords_lengths(cocircular_crit)
        if display == 1:
            tmp = co.pol_oper.polar_to_cart(
                new_polar, new_ref_point, new_ref_angle)
            # binarm3d[tuple(tmp.T)] = [255, 255, 0]
            '''
            binarm3d[tuple(co.pol_oper.polar_to_cart(cocircular_crit, new_ref_point, new_ref_angle).T)] = [
                255, 255, 0]
            '''
            binarm3d[tuple(tmp[np.abs(np.sqrt((tmp[:, 0] - new_ref_point[0])**2
                                              + (tmp[:, 1] - new_ref_point[1])**2)
                                      - new_polar[new_crit_ind, 0]) <=
                               resolution].T)] = [255, 255, 0]
            binarm3d[tuple(co.pol_oper.polar_to_cart(new_polar[np.abs(new_polar[new_crit_ind, 0] -
                                                                      new_polar[:, 0]) < 0.1, :],
                                                     new_ref_point, new_ref_angle).T)] = [255, 0, 255]
            binarm3d[np.abs(np.sqrt((positions[:, :, 0] - new_ref_point[0])**2 + (positions[
                :, :, 1] - new_ref_point[1])**2) - new_ref_radius) <= resolution] = [255, 255, 0]
            binarm3d[int(new_ref_point[0]), int(
                new_ref_point[1])] = [255, 0, 0]
            cv2.arrowedLine(binarm3d, (int(new_ref_point[1]),
                                       int(new_ref_point[0])),
                            (int(new_ref_point[1] +
                                 new_polar[new_crit_ind, 0] *
                                 np.cos(new_ref_angle)),
                             int(new_ref_point[0] +
                                 new_polar[new_crit_ind, 0] *
                                 np.sin(new_ref_angle))), [0, 0, 255], 2, 1)
        if cocircular_crit == []:
            LOG.debug('Reached end of hand without finding abnormality')
            co.chhm.no_abnorm += 1
            if display == 1:
                tag_im(binarm3d, 'Reached end of hand without finding ' +
                       'abnormality')
                co.im_results.images.append(binarm3d)
            return None, None, None

            '''
            cv2.arrowedLine(binarm3d, (int(new_ref_point[1]), int(new_ref_point[0])), (
                int(new_ref_point[1] + new_ref_radius * np.cos(new_ref_angle)),
                int(new_ref_point[0] + new_ref_radius * np.sin(new_ref_angle))),
                [0, 0, 255], 2, 1)
            '''
        width_lo_thres = new_ref_radius * co.CONST['abnormality_tol']
        check_abnormality = ((crit_chords < width_lo_thres) *
                             (crit_chords > 1))
        reached_abnormality = np.sum(check_abnormality)
        if display == 1:
            interesting_points_ind = np.nonzero(reached_abnormality)[0]
            for ind in interesting_points_ind:
                cv2.line(binarm3d, tuple(co.pol_oper.polar_to_cart(
                    np.array([cocircular_crit[ind, :]]),
                    new_ref_point, new_ref_angle)[:, ::-1].flatten()),
                    tuple(co.pol_oper.polar_to_cart(
                        np.array([cocircular_crit[ind + 1, :]]),
                        new_ref_point, new_ref_angle)
                    [:, ::-1].flatten()), [0, 255, 0], 3)
        if reached_abnormality:
            hand_patch, hand_patch_pos, full_res_mask = find_hand(binarm, binarm3d, armpoints, display,
                                                                  new_polar,
                                                                  new_corrected_segment, new_ref_angle,
                                                                  new_crit_ind, new_ref_point, resolution)
        if display == 1:
            try:
                binarm3d[int(corrected_cart_point[0]),
                         int(corrected_cart_point[1])] = [255, 0, 255]
            except UnboundLocalError:
                pass
            if __name__ == '__main__':
                cv2.imshow('test', binarm3d)
                cv2.waitKey(0)
        if reached_abnormality:
            if display == 1:
                co.im_results.images.append(binarm3d)
            return hand_patch, hand_patch_pos, full_res_mask

        link_end_1st = new_polar[new_crit_ind, :]
        link_end_radius = link_end_1st[0]
        tmp = new_polar[
            np.abs(new_polar[:, 0] - link_end_radius) < resolution, :]
        link_end_2nd = tmp[np.argmax(np.abs(tmp[:, 1])), :]
        link_end_segment[0] = co.pol_oper.polar_to_cart(
            np.array([link_end_1st]), new_ref_point, new_ref_angle)
        link_end_segment[1] = co.pol_oper.polar_to_cart(
            np.array([link_end_2nd]), new_ref_point, new_ref_angle)
        new_polar = new_polar[new_crit_ind:, :]



def find_hand(*args):
    '''
    Find hand when abnormality reached
    '''
    # binarm,polar,ref_angle,ref_point,crit_ind,corrected_segment,resolution,display,binarm3d
    [binarm, binarm3d, armpoints, display, polar, corrected_segment,
     ref_angle, crit_ind, ref_point, resolution] = args[0:10]
    separate_hand = 0
    ref_dist = calculate_cart_dists(corrected_segment)
    bins = np.arange(resolution, np.max(
        polar[:crit_ind + 1, 0]) + 2 * resolution, resolution)
    dig_rad = np.digitize(polar[crit_ind::-1, 0], bins)
    angles_bound = np.abs(np.arctan((ref_dist / 2.0) / (bins))) +\
        co.CONST['angle_tol']
    try:
        angles_bound = angles_bound[dig_rad]
    except IndexError as e:
        print('dig_rad', dig_rad)
        print('bins', bins)
        print('angles_bound', angles_bound)
        raise(e)
    angles_thres = np.abs(polar[crit_ind::-1, 1]) < angles_bound
    # Compmat holds in every column elements of the same radius bin
    compmat = dig_rad[:, None] == np.arange(bins.shape[0])
    # angles_thres-1 gives -1 for elements outside bounds and 0 for elements
    # inside bounds
    #(angles_thres-1)+compmat == 1 is in every column True for elements of
    # the column radius bin that reside inside angles bounds
    # sameline is true if there are such elements in a column
    # and false when there are no such elements
    sameline = np.sum(
        (((angles_thres[:, None] - 1) + compmat) == 1), axis=0) == 1
    # compmat[:,sameline] holds the columns for which there is at least one
    # such element inside.
    # sum over axis=1  returns a column that holds all elements with this
    # criterion

    dig_rad_thres = np.sum(compmat[:, sameline], axis=1) > 0
    dig_rad_thres = dig_rad_thres.astype(int)
    # I make TRUE some solo FALSES, so that to create compact TRUEs segments
    # 'dilate'
    dig_rad_thres[1:-1] += (np.roll(dig_rad_thres, 1) +
                            np.roll(dig_rad_thres, -1))[1:-1]
    # 'erode'
    dig_rad_thres[1:-1] *= (np.roll(dig_rad_thres, 1)
                            * np.roll(dig_rad_thres, -1))[1:-1]

    if np.sum(dig_rad_thres) == 0:
        LOG.debug('Hand not found but reached abnormality')
        if display == 1:
            tag_im(binarm3d, 'Hand not found but reached abnormality')
        co.chhm.rchd_abnorm += 1
        return None, None, None

    _, dig_rad_thres = find_largest_trues_segment(dig_rad_thres > 0)
    used_polar = polar[crit_ind::-1, :][dig_rad_thres]
    if display == 1:
        binarm3d[tuple(co.pol_oper.polar_to_cart(
            used_polar,
            ref_point, ref_angle).T)] = [255, 0, 0]
    # sort_inds holds the indices that sort the used_polar, first by
    # corresponding bin radius and afterwards by angles
    sort_inds = np.lexsort(
        (used_polar[:, 1], bins[dig_rad[dig_rad_thres]]))[::-1]
    # used_polar now holds values at the above order
    used_polar = used_polar[sort_inds, :]
    # chords length between consecutive rows of used_polar is
    # calculated. In interchanging bins radii the distance is either too small,
    # so it does no bad, or big enough to get a working length that can help, or
    # too big, that is thrown away
    same_rad_dists = calculate_chords_lengths(used_polar)
    dist_threshold = ((np.abs(same_rad_dists)
                       <= co.CONST['dist_tol'] * np.abs(ref_dist)) *
                      (np.abs(same_rad_dists) >
                       2 * np.abs(ref_dist) / 3.0))
    if display == 1:
        binarm3d[tuple(co.pol_oper.polar_to_cart(
            used_polar[np.concatenate(
                (dist_threshold, [0]), axis=0).astype(bool)],
            ref_point, ref_angle).T)] = [0, 255, 0]
    same_rad_dists[np.logical_not(dist_threshold)] = 1000
    if display == 3:
        flag = 1
        tmp = co.pol_oper.polar_to_cart(
            used_polar, ref_point, ref_angle)[:, ::-1]
        for count1, (row1, row2) in enumerate(zip(tmp[:-1, :], tmp[1:, :])):
            if dist_threshold[count1] == 0:
                cv2.line(binarm3d, tuple(row1), tuple(row2), [255, 0, 0], 1)
            elif flag == 1:
                cv2.line(binarm3d, tuple(row1), tuple(row2), [0, 0, 255], 1)
                flag = 0
    try:
        ind = np.argmin(same_rad_dists)
        chosen = used_polar[ind:ind + 2, :]
        if display == 2:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.scatter(bins[dig_rad[dig_rad_thres]][
                sort_inds], used_polar[:, 1])
            plt.draw()
            plt.pause(0.1)
            plt.waitforbuttonpress(timeout=-1)
            plt.close(fig)
        wristpoints = co.pol_oper.polar_to_cart(
            chosen, ref_point, ref_angle)
        wrist_radius = calculate_cart_dists(
            wristpoints) / 2.0

        hand_edges = co.pol_oper.polar_to_cart(
            polar[polar[:, 0] > np.min(chosen[:, 0])], ref_point, ref_angle)
        if separate_hand:
            hand_box = binarm[np.min(hand_edges[:, 0]):
                              np.max(hand_edges[:, 0]),
                              np.min(hand_edges[:, 1]):
                              np.max(hand_edges[:, 1])]
            # perform top hat transform, with suggested
            # finger radius smaller than wrist radius
            struct_el = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, tuple(2 * [int(0.9 * wrist_radius)]))
            palm = cv2.morphologyEx(
                hand_box, cv2.MORPH_OPEN, struct_el)
            fingers = hand_box - palm
            fingers = cv2.morphologyEx(fingers, cv2.MORPH_OPEN, np.ones(
                tuple(2 * [int(struct_el.shape[0] / 4)]), np.uint8))
        if display == 1:
            convexhand = cv2.convexHull(hand_edges).squeeze()
            cv2.drawContours(
                binarm3d, [np.fliplr(convexhand)], 0, [0, 0, 255], 2)
            binarm3d[tuple(co.pol_oper.polar_to_cart(np.array(
                [np.mean(polar[ind:, :],
                         axis=0)]), ref_point, ref_angle).
                astype(int).T)] = [255, 0, 255]
        LOG.debug('Hand found')
        if display == 1:
            tag_im(binarm3d, 'Hand found')
        co.chhm.found += 1
        if display == 2:
            if separate_hand:
                cv2.imshow('Fingers', fingers)
                cv2.imshow('Palm', palm)
                binarm3d[tuple(hand_edges.T)] = [255, 0, 0]
                binarm3d[tuple(wristpoints.T)] = [0, 0, 255]
        # binarm3d[tuple(co.pol_oper.polar_to_cart(polar[polar[:,1]>0],ref_point,ref_angle).T)]=[255,0,0]
        hand_patch = binarm[np.min(hand_edges[:, 0]):
                            np.max(hand_edges[:, 0]),
                            np.min(hand_edges[:, 1]):
                            np.max(hand_edges[:, 1])]
        full_res_mask = np.zeros(binarm.shape)
        full_res_mask[np.min(hand_edges[:, 0]):
                      np.max(hand_edges[:, 0]),
                      np.min(hand_edges[:, 1]):
                      np.max(hand_edges[:, 1])] = hand_patch
        hand_patch_pos = np.array(
            [hand_edges[:, 0].min(), hand_edges[:, 1].min()])
        # hand_patch has same values as input data.
        # hand_patch_pos denotes the hand_patch upper left corner absolute
        #   location
        # full_res_mask is hand_patch in place
        return hand_patch, hand_patch_pos, full_res_mask
    except IndexError:
        LOG.debug('Hand not found but reached abnormality')
        if display == 1:
            tag_im(binarm3d, 'Hand not found but reached abnormality')
        co.chhm.rchd_abnorm += 1
        return None, None, None


def find_link_direction(
        xy_points, entry_segment, perp_to_segment_unit, point):
    '''
    function call:
    find_link_direction(xy_points, entry_segment,
                        perp_to_segment_unit, point)
    '''
    vecs = np.array(point) - np.array(entry_segment)
    incl = np.arctan2(vecs[:, 0], vecs[:, 1])
    init_incl_bounds = [np.arctan2(
        perp_to_segment_unit[0], perp_to_segment_unit[1])]
    if init_incl_bounds[0] + pi > pi:
        init_incl_bounds.append(init_incl_bounds[0] - pi)
    else:
        init_incl_bounds.append(init_incl_bounds[0] + pi)

    num = [0, 0]
    for count in range(2):
        complex_points = (xy_points[:, 0] -
                          entry_segment[count][0]) * 1j + (xy_points[:, 1] -
                                                           entry_segment[count][1])
        angles = np.angle(complex_points)
        incl_bound = init_incl_bounds[np.argmin(np.abs(co.pol_oper.mod_diff(
            np.array(init_incl_bounds), np.array([incl[count], incl[count]]))))]
        _max = max([incl[count], incl_bound])
        _min = min([incl[count], incl_bound])
        num[count] = np.sum(co.pol_oper.mod_between_vals(angles, _min, _max))
    # Thales theorem:
    tmp = np.array(point) - \
        np.array(entry_segment[(np.argmax(num) + 1) % 2][:])

    return np.arctan2(tmp[0], tmp[1]), [entry_segment[np.argmax(num)][:], point]


def picture_box(binarm3d, points):
    '''Draw box on image'''
    tmp = np.zeros_like(binarm3d[:, :, 0])
    tmp[tuple(points.T)] = 255
    binarm3d[tmp[:, :] > 0] = [0, 255, 0]
    return binarm3d


def tag_im(img, text):
    '''
    Tag top right of img with description in red
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (0, 20), font, 0.5, (0, 0, 255), 2)
