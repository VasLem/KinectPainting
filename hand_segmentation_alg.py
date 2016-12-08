'''
Input: Hand Mask nonzero xy_points
Use laplacian of binary mask
0.Find entry segment
1.Calculate segment middle point
2.Transform nonzero points  across the right semiplane* to new_polar coordinates ,
with reference center the point found and reference angle the one of the normal
to the semiplane*.
3.Start augmenting a semicircle from the reference center in the right
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


def with_laplacian(binarm):
    '''
    Find edges using Laplacian
    '''
    return cv2.Laplacian(binarm, cv2.CV_8U)


def polar_to_cart(points, center, ref_angle):
    '''
    Polar coordinates to cartesians,
    given center and reference angle
    '''
    return np.concatenate(
        ((points[:, 0] * np.sin(ref_angle + points[:, 1]) +
          center[0])[:, None].astype(int),
         (points[:, 0] * np.cos(ref_angle + points[:, 1]) +
          center[1])[:, None].astype(int)), axis=1)


def usingcontours(imag):
    '''
    Find edges using cv2.findContours
    '''
    points = np.transpose(cv2.findContours(
        imag.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1][0])
    tmp = np.zeros_like(imag)
    tmp[tuple(points[::-1, :])] = 255
    return tmp


def polar_change_origin(old_polar, old_ref_angle, old_ref_point, new_ref_point):
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
    #_angle[(_sin<0) * (_cos<0)]+=pi #haven't checked yet
    #_angle[(_sin>0) * (_cos<0)]-=pi
    #_angle[_angle < -pi] += 2 * pi
    #_angle[_angle > pi] -= 2 * pi
    return np.concatenate((_radius[:, None], _angle[:, None]), axis=1)


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


def detect_entry(bin_mask):
    '''
    Function to detect  intersection limits of mask with calibration edges
    Assuming non_convex calib edges..
    '''
    # entry_segments=positions[bin_mask*co.masks.calib_edges>0]
    entry_segments = co.meas.edges_positions[
        bin_mask[co.meas.edges_positions_indices] > 0]
    if entry_segments.shape[0] == 0:
        return np.array([[]])
    approx_entry_segments = entry_segments.copy()
    approx_entry_segments[
        entry_segments[:, 1] <
        co.meas.nonconvex_edges_lims[0], 1] = co.meas.nonconvex_edges_lims[0]
    approx_entry_segments[
        entry_segments[:, 0] <
        co.meas.nonconvex_edges_lims[1], 0] = co.meas.nonconvex_edges_lims[1]
    approx_entry_segments[
        entry_segments[:, 1] >
        co.meas.nonconvex_edges_lims[2], 1] = co.meas.nonconvex_edges_lims[2]
    approx_entry_segments[
        entry_segments[:, 0] >
        co.meas.nonconvex_edges_lims[3], 0] = co.meas.nonconvex_edges_lims[3]
    approx_entry_points = cv2.convexHull(approx_entry_segments).squeeze()
    if approx_entry_points.shape[0] == 2:
        if calculate_cart_dists(approx_entry_points) > np.min(bin_mask.shape) / 10.0:
            return\
                [entry_segments[find_rows_in_array(
                    approx_entry_segments, approx_entry_points)]]
        else:
            return np.array([[]])
    approx_entry_orient = np.diff(approx_entry_points, axis=0)
    approx_entry_orient = (approx_entry_orient /
                           calculate_cart_dists(
                               approx_entry_points)[:, None])
    approx_entry_vert_orient = np.dot(
        approx_entry_orient, np.array([[0, -1], [1, 0]]))
    num = []
    for count, orient in enumerate(approx_entry_vert_orient):
        pos = find_segment_to_point_box(approx_entry_segments,
                                        np.array([approx_entry_points[count, :] + orient * 10,
                                                  approx_entry_points[count, :] - orient * 10]),
                                        approx_entry_points[count + 1, :])[0]
        num.append(pos.shape[0])
    _argmax = np.argmax(num)
    entry_points = entry_segments[find_rows_in_array(
        approx_entry_segments, approx_entry_points[_argmax:_argmax + 2, :])]
    return [entry_points]


def find_cocircular_points(polar_points, radius, resolution):
    '''
    Find cocircular points given radius and suggested resolution
    '''
    return polar_points[np.abs(polar_points[:, 0] - radius) <= resolution, :]


def find_cocircular_farthest_point(new_polar, ref_angle, ref_point, ref_radius,
                                   entry_angles):
    '''
    Find farthest cocircular point from entry segment
    '''
    resolution = np.sqrt(2) / 2.0
    cocircular_points = find_cocircular_points(
        new_polar, ref_radius, resolution)
    # cocircular_points[np.abs(cocircular_points[:, 1] + pi) < 0.001, 1] *= -1
    check1 = np.abs(entry_angles[0] - (cocircular_points[:, 1] + ref_angle))
    check2 = np.abs(entry_angles[1] - (cocircular_points[:, 1] + ref_angle))
    _min1 = np.min(check1)
    _min2 = np.min(check2)
    if np.abs(_min1) < np.abs(_min2):
        farthest_cocirc_point = cocircular_points[
            check1 == _min1].ravel()[0:2]
    else:
        farthest_cocirc_point = cocircular_points[
            check2 == _min2].ravel()[0:2]
    far_cocirc_point_xy = [int(farthest_cocirc_point[0] *
                               (np.sin(farthest_cocirc_point[1] + ref_angle)) + ref_point[0]),
                           int(farthest_cocirc_point[0] *
                               (np.cos(farthest_cocirc_point[1] + ref_angle)) + ref_point[1])]
    return farthest_cocirc_point, far_cocirc_point_xy, cocircular_points


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


def mod_correct(points):
    '''
    Correct polar points, subjects to
    modular (-pi,pi)
    '''
    points[points[:, 1] > pi, 1] -= 2 * pi
    points[points[:, 1] < -pi, 1] += 2 * pi


def calculate_chords_lengths(polar_points):
    '''
    Assuming polar points are sorted by angle
    '''
    return ((polar_points[:-1, 0] + polar_points[1:, 0]) / 2.0) * np.sqrt(2 * (
        1 - np.cos(mod_diff(polar_points[:-1, 1], polar_points[1:, 1]))))


def fix_angle(angle):
    '''
    Same with mod_correct, for single angles
    '''
    if angle < -pi:
        angle += 2 * pi
    elif angle > pi:
        angle -= 2 * pi
    return angle


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


def mod_between_vals(angles, min_bound, max_bound):
    '''
    Find angles between bounds, using modular (-pi,pi) logic
    '''
    if max_bound == min_bound:
        return np.zeros((0))
    res = mod_diff(max_bound, min_bound, 1)[1]
    if res == 0:
        return (angles <= max_bound) * (angles >= min_bound)
    else:
        return ((angles >= max_bound) * (angles <= pi)
                + (angles >= -pi) * (angles <= min_bound))


def mod_diff(angles1, angles2, ret_argmin=0):
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


def main_process(binarm3d, positions=None, display=0):
    '''Main processing function'''
    if positions is None:
        if co.meas.all_positions is None:

            co.meas.all_positions = np.transpose(np.nonzero(np.ones_like(
                binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
        positions=co.meas.all_positions
    if len(binarm3d.shape) == 3:
        binarm = binarm3d[:, :, 0].copy()
        if np.issubdtype(binarm3d[0, 0, 0], np.uint8):
            if np.max(binarm3d) == 1:
                binarm3d *= 255
        else:
            binarm3d = (255 * binarm3d).astype(np.uint8)
    else:
        binarm = binarm3d.copy()
        if display == 1:
            binarm3d = np.tile(binarm[:, :, None], (1, 1, 3))
            if np.issubdtype(binarm3d[0, 0, 0], np.uint8):
                if np.max(binarm3d) == 1:
                    binarm3d *= 255
            else:
                binarm3d = (255 * binarm3d).astype(np.uint8)
    try:
        armpoints = find_nonzero(binarm)
    except AttributeError:  # binarm is []
        if display == 1:
            tag_im(binarm3d, 'No object found')
            co.im_results.images.append(binarm3d)
        return np.array([[]]),np.array([[]])

    '''
    tmp=np.concatenate((np.array(co.meas.edges_positions_indices),
                    np.zeros((1,co.meas.edges_positions.shape[0]),int)),axis=0)
    print tmp.shape
    binarm3d[tuple(tmp)]=255
    '''
    points = find_nonzero(with_laplacian(binarm))
    points_size = points.shape[0]
    points = points[:, 0] * 1j + points[:, 1]
    tmp = np.angle(points)
    tmp[tmp < -pi] += 2 * pi
    tmp[tmp > pi] -= 2 * pi
    new_polar = np.concatenate(
        (np.absolute(points)[:, None], tmp[:, None]), axis=1)
    try:
        entry = detect_entry(binarm)[0]
    except IndexError:
        if display == 1:
            tag_im(binarm3d, 'No entry found')
            co.im_results.images.append(binarm3d)
        return np.array([[]]),np.array([[]])

    if entry.shape[0] <= 1:
        if display == 1:
            tag_im(binarm3d, 'Arm in image corners or its entry is occluded' +
                   ', hand segmentation algorithm cannot function')
            co.im_results.images.append(binarm3d)
        return np.array([[]]),np.array([[]])

    link_end_radius = 1 / 2.0 * calculate_cart_dists(entry)
    link_end_segment = entry[:]
    new_ref_point = [0, 0]
    new_ref_angle = 0
    new_crit_ind = 0
    link_end_2nd = []
    resolution = np.sqrt(2) / 2.0
    new_corrected_segment = entry[:]
    # for _count in range(3):
    while True:
        prev_ref_point = new_ref_point[:]
        prev_polar = new_polar[:]
        prev_ref_angle = new_ref_angle
        prev_ref_point = new_ref_point[:]
        if new_crit_ind > points_size - 10:
            if display == 1:
                tag_im(binarm3d, 'Reached Mask Limits')
                co.im_results.images.append(binarm3d)
            return np.array([[]]),np.array([[]])

        new_ref_point = [(link_end_segment[0][0] + link_end_segment[1][0]) /
                         2.0, (link_end_segment[0][1] + link_end_segment[1][1]) / 2.0]
        new_polar = polar_change_origin(
            prev_polar.copy(), prev_ref_angle, prev_ref_point, new_ref_point)
        new_ref_radius = calculate_cart_dists(link_end_segment) / 2.0
        link_end_diff = link_end_segment[0, :] - link_end_segment[1, :]
        tmpangle = np.arctan2(link_end_diff[0], link_end_diff[1])
        angle1 = fix_angle(tmpangle + pi / 2)
        angle2 = fix_angle(tmpangle - pi / 2)
        try:
            (_, far_cocirc_point_xy, _) = find_cocircular_farthest_point(
                new_polar, 0, new_ref_point, new_ref_radius, [angle1, angle2])
            box, perp_to_segment_unit = find_segment_to_point_box(
                armpoints, link_end_segment, np.array(far_cocirc_point_xy))
            if display == 1:
                binarm3d = picture_box(binarm3d, box)
            new_ref_angle, new_corrected_segment = find_link_direction(
                box, link_end_segment, perp_to_segment_unit, np.array(far_cocirc_point_xy))
        except ValueError:
            seg_len = calculate_cart_dists(link_end_segment)[0]
            par_to_segment_unit = link_end_diff / seg_len
            seg_angle = np.arctan2(par_to_segment_unit[
                0], par_to_segment_unit[1])
            angle1 = fix_angle(seg_angle - pi / 2)
            angle2 = fix_angle(seg_angle + pi / 2)
            try:
                comp_angle = new_polar[
                    np.argmin(new_polar[(new_polar[:, 0] - seg_len / 2.0) > 2, 0]), 1]
            except ValueError:
                return np.array([[]]),np.array([[]])

            angdiff1 = mod_diff(comp_angle, angle1)
            angdiff2 = mod_diff(comp_angle, angle2)
            if np.abs(angdiff1) < np.abs(angdiff2):
                new_ref_angle = angle1
            else:
                new_ref_angle = angle2
        new_polar[:, 1] -= (new_ref_angle)
        mod_correct(new_polar)
        new_polar = new_polar[new_polar[:, 0] >= new_ref_radius, :]
        new_polar = new_polar[new_polar[:, 0].argsort(), :]
        if display == 1:
            tmp = polar_to_cart(new_polar, new_ref_point, new_ref_angle)
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
            if display == 1:
                tag_im(binarm3d, 'No cocircular points found' +
                       ' reached end of hand')
                co.im_results.images.append(binarm3d)
            return np.array([[]]),np.array([[]])

        _min = cand_crit_points[0, :]
        new_crit_ind = np.where(new_polar == _min)[0][0]

        cocircular_crit = find_cocircular_points(new_polar,
                                                 new_polar[new_crit_ind, 0],
                                                 resolution)
        cocircular_crit = cocircular_crit[cocircular_crit[:, 1].argsort(), :]
        crit_chords = calculate_chords_lengths(cocircular_crit)
        if display == 1:
            tmp = polar_to_cart(new_polar, new_ref_point, new_ref_angle)
            # binarm3d[tuple(tmp.T)] = [255, 255, 0]
            '''
            binarm3d[tuple(polar_to_cart(cocircular_crit, new_ref_point, new_ref_angle).T)] = [
                255, 255, 0]
            '''
            binarm3d[tuple(tmp[np.abs(np.sqrt((tmp[:, 0] - new_ref_point[0])**2
                                              + (tmp[:, 1] - new_ref_point[1])**2)
                                      - new_polar[new_crit_ind, 0]) <=
                               resolution].T)] = [255, 255, 0]
            binarm3d[tuple(polar_to_cart(new_polar[np.abs(new_polar[new_crit_ind, 0] -
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
            if display == 1:
                tag_im(binarm3d, 'Reached end of hand without result')
            co.im_results.images.append(binarm3d)
            return np.array([[]]),np.array([[]])

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
                cv2.line(binarm3d, tuple(polar_to_cart(
                    np.array([cocircular_crit[ind, :]]),
                    new_ref_point, new_ref_angle)[:, ::-1].flatten()),
                    tuple(polar_to_cart(
                        np.array([cocircular_crit[ind + 1, :]]),
                        new_ref_point, new_ref_angle)
                    [:, ::-1].flatten()), [0, 255, 0], 3)
        if reached_abnormality:
            hand_patch, hand_patch_pos = find_hand(binarm, binarm3d, armpoints, display,
                                    new_polar,
                                    new_corrected_segment, new_ref_angle,
                                    new_crit_ind, new_ref_point, resolution)
        if display == 1:
            try:
                binarm3d[int(far_cocirc_point_xy[0]),
                         int(far_cocirc_point_xy[1])] = [255, 0, 255]
            except UnboundLocalError:
                pass
            if __name__ == '__main__':
                cv2.imshow('test', binarm3d)
                cv2.waitKey(0)
        if reached_abnormality:
            co.im_results.images.append(binarm3d)
            return hand_patch,hand_patch_pos

        link_end_1st = new_polar[new_crit_ind, :]
        link_end_radius = link_end_1st[0]
        tmp = new_polar[
            np.abs(new_polar[:, 0] - link_end_radius) < resolution, :]
        link_end_2nd = tmp[np.argmax(np.abs(tmp[:, 1])), :]
        link_end_segment[0] = polar_to_cart(
            np.array([link_end_1st]), new_ref_point, new_ref_angle)
        link_end_segment[1] = polar_to_cart(
            np.array([link_end_2nd]), new_ref_point, new_ref_angle)
        new_polar = new_polar[new_crit_ind:, :]


def find_hand(*args):
    '''
    Find hand when abnormality reached
    '''
    # binarm,polar,ref_angle,ref_point,crit_ind,corrected_segment,resolution,display,binarm3d
    [binarm, binarm3d, armpoints, display, polar, corrected_segment,
     ref_angle, crit_ind, ref_point, resolution]=args[0:10]
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
        print 'dig_rad', dig_rad
        print 'bins', bins
        print 'angles_bound', angles_bound
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

    dig_rad_thres = (dig_rad_thres > 0).astype(np.uint8)
    # I have to keep the biggest TRUEs segment and zero all the others
    dig_rad_thres_diff = np.diff(np.pad(dig_rad_thres.astype(
        int), (1, 1), 'constant', constant_values=0))[:-1]
    dig_rad_thres_diff_cum = np.cumsum(dig_rad_thres)
    tmp = dig_rad_thres_diff_cum[dig_rad_thres_diff != 0]
    dig_rad_segments_mass = np.diff(tmp)
    try:
        _argmax = np.argmax(np.abs(dig_rad_segments_mass))
    except ValueError as err:
        if display == 1:
            tag_im(binarm3d, 'Hand not found but reached abnormality')
        return np.array([[]]),np.array([[]])

    tmp = np.zeros((dig_rad_segments_mass.shape[0] + 1))
    tmp[_argmax] = 2
    tmp[_argmax + 1] = 3
    dig_rad_thres[(dig_rad_thres_diff != 0)] = tmp
    dig_rad_thres[:np.nonzero(dig_rad_thres == 2)[0][0]] = 0
    dig_rad_thres[np.nonzero(dig_rad_thres == 3)[0][0] + 1:] = 0
    dig_rad_thres = dig_rad_thres > 0
    # used_polar has every element of polar in reverse order, which satisfies
    # the criterion
    used_polar = polar[crit_ind::-1, :][dig_rad_thres]
    if display == 1:
        binarm3d[tuple(polar_to_cart(
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
                       <= co.CONST['dist_tol'] + np.abs(ref_dist)) *
                      (np.abs(same_rad_dists) >
                       2 * np.abs(ref_dist) / 3.0))
    if display == 1:
        binarm3d[tuple(polar_to_cart(
            used_polar[np.concatenate(
                (dist_threshold, [0]), axis=0).astype(bool)],
            ref_point, ref_angle).T)] = [0, 255, 0]
    same_rad_dists[np.logical_not(dist_threshold)] = 1000
    if display == 3:
        flag = 1
        tmp = polar_to_cart(used_polar, ref_point, ref_angle)[:, ::-1]
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
        wristpoints= polar_to_cart(
            chosen, ref_point, ref_angle)
        wrist_radius = calculate_cart_dists(
            wristpoints) / 2.0

        hand_edges = polar_to_cart(
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
            binarm3d[tuple(polar_to_cart(np.array(
                [np.mean(polar[ind:, :],
                         axis=0)]), ref_point, ref_angle).
                astype(int).T)] = [255, 0, 255]
            if display == 1:
                tag_im(binarm3d, 'Hand found')
        if display == 2:
            if separate_hand:
                cv2.imshow('Fingers', fingers)
                cv2.imshow('Palm', palm)
                binarm3d[tuple(hand_edges.T)] = [255, 0, 0]
                binarm3d[tuple(wristpoints.T)] = [0, 0, 255]
        # binarm3d[tuple(polar_to_cart(polar[polar[:,1]>0],ref_point,ref_angle).T)]=[255,0,0]
        hand_patch = binarm[np.min(hand_edges[:, 0]):
                          np.max(hand_edges[:, 0]),
                          np.min(hand_edges[:, 1]):
                          np.max(hand_edges[:, 1])]
        hand_patch_pos=np.array([hand_edges[:,0].min(),hand_edges[:,1].min()])
        return hand_patch, hand_patch_pos
    except IndexError:
        if display == 1:
            tag_im(binarm3d, 'Hand not found but reached abnormality')
        return np.array([[]]), np.array([[]])


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
        incl_bound = init_incl_bounds[np.argmin(np.abs(mod_diff(
            np.array(init_incl_bounds), np.array([incl[count], incl[count]]))))]
        _max = max([incl[count], incl_bound])
        _min = min([incl[count], incl_bound])
        num[count] = np.sum(mod_between_vals(angles, _min, _max))
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
    Tag top of img with description in red
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (0, 20), font, 0.5, (0, 0, 255), 2)


def main():
    '''Main Caller Function'''
    if not os.path.exists('arm_example.png'):
        urllib.urlretrieve("https://www.packtpub.com/\
                           sites/default/files/Article-Images/B04521_02_04.png",
                           "arm_example.png")
    # binarm3d = cv2.imread('random.png')
    binarm3d = cv2.imread('arm_example.png')
    binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
        binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
    co.masks.calib_edges = np.pad(np.zeros((binarm3d.shape[
        0] - 2, binarm3d.shape[1] - 2), np.uint8), ((1, 1), (1, 1)), 'constant', constant_values=1)
    co.meas.find_non_convex_edges_lims(co.masks.calib_edges)
    print\
        timeit.timeit(lambda: main_process(binarm3d.copy(),
                                           binarm3d_positions, 0), number=100) / 100
    profile.runctx('main_process(binarm3d,binarm3d_positions,0)',
                   globals(), locals())
    # main_process(binarm3d.copy(), binarm3d_positions)

    for _ in range(4):
        rows, cols, _ = binarm3d.shape
        rot_mat = cv2.getRotationMatrix2D(
            (np.floor(cols / 2.0), np.floor(rows / 2.0)), 90, 1)
        rot_mat[0, 2] += np.floor(rows / 2.0 - cols / 2.0)
        rot_mat[1, 2] += np.floor(cols / 2.0 - rows / 2.0)
        binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
            binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
        main_process(binarm3d.copy(), binarm3d_positions, 1)
        binarm3d = 255 * \
            ((cv2.warpAffine(binarm3d, rot_mat, (rows, cols))) > 0).astype(np.uint8)
        co.masks.calib_edges = np.pad(np.zeros((
            binarm3d.shape[0] - 2, binarm3d.shape[1] - 2), np.uint8), (
                (1, 1), (1, 1)), 'constant', constant_values=1)
        co.meas.find_non_convex_edges_lims(co.masks.calib_edges)

if __name__ == '__main__':
    main()
