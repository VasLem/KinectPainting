'''Input: Hand Mask nonzero points'''
'''
Use laplacian of binary mask
0.Find entry segment
1.Calculate segment middle point
2.Transform nonzero points  across the right semiplane* to new_polar coordinates ,
with reference center the point found and reference angle the one of the normal
to the semiplane*.
3.Start augmenting a semicircle from the reference center in the right
semiplane* ,calculate zero crossings on its perimeter, with low step,
and append the nonzero points  to a 2d list.
4.While 4 zerocrossings, calculate distance of white points and mean angle:
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
import os
import numpy as np
import cv2
import class_objects as co
import palm_detection_alg as pda
import urllib
import time
import timeit
import cProfile as profile
from math import pi


def withLaplacian(binarm):
    a = cv2.Laplacian(binarm, cv2.CV_8U)
    return a


def polar_to_cart(points, prev_ref_point, ref_angle):
    return np.concatenate(
        ((points[:, 0] * np.sin(ref_angle + points[:, 1]) + prev_ref_point[0])[:, None].astype(int),
         (points[:, 0] * np.cos(ref_angle + points[:, 1]) + prev_ref_point[1])[:, None].astype(int)), axis=1)


def cart_to_polar(points, desired_ref_point, desired_ref_angle):
    tmp = (desired_ref_point[0] + points[:, 0]) * \
        1j + (desired_ref_point[1] + points[:, 1])
    # return np.concatenate(tmp[0,:


def usingcontours(im):
    points = np.transpose(cv2.findContours(
        im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1][0])
    tmp = np.zeros_like(im)
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


def detect_entry(bin_mask, positions):
    '''function to detect  intersection limits of mask with calib_edges'''
    #_,entries,_=cv2.findContours(cv2.dilate(co.masks.calib_edges*bin_mask,np.ones((3,3),np.uint8)),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #entries_length=[cv2.arcLength(entry,1) for entry in entries]
    #(entries, entries_length) = zip(*sorted(zip(entries, entries_length),
    #                                               key=lambda b:b[1],
    #                                        reverse=1))
    # entry_points=cv2.convexHull(entries[0],0)
    entry_points = cv2.convexHull(
        positions[co.masks.calib_edges * bin_mask > 0], 0).squeeze()
    if entry_points.shape[1] == 2:
        # top_or_bot_entry==1 iff entry is from top or bottom
        top_or_bot_entry = entry_points[0][0] - entry_points[1][0] < 10
    else:
        complex_entry = entry_points[:][0] + entry_points[:][1] * 1j
        dists = complex_entry * complex_entry.T
        udclose = np.zeros(entry_points.shape[0], entry_points.shape[0])
        lrclose = np.zeros(entry_points.shape[0], entry_points.shape[0])
        for count1, entry_point1 in enumerate(entry_points):
            for count2, entry_point2 in enumerate(entry_points[count1, :]):
                if np.abs(entry_point1[0] - entry_point2[0]) < 10:
                    udclose[count1, count2] = dists[count1, count2]
                elif np.abs(entry_point1[1] - entry_point2[1]) < 10:
                    lrclose[count1, count2] = dists[count1, count2]
        # The following does not take into account that there can be entry from
        # the corners of the image. This will be a future update
        if np.sum(lrclose > 0) > (np.sum(udclose > 0)):
            # must come from left or right
            # chosen points which have the most distance
            entry_points = entry_points[list(np.unravel_index(lrclose.argmax(),
                                                              lrclose.shape)), :]
            top_or_bot_entry = 0
        else:
            # must come from top or bottom
            entry_points = entry_points[list(np.unravel_index(udclose.argmax(),
                                                              udclose.shape)), :]
            top_or_bot_entry = 1
    # dir_sign==1 iff entry is bottom or right

    if top_or_bot_entry == 0:
        dir_sign = (1 if
                    (bin_mask.shape[1] - entry_points[0]
                     [1]) > entry_points[0][1]
                    else -1)
    else:
        dir_sign = (1 if
                    (bin_mask.shape[0] - entry_points[0]
                     [0]) > entry_points[0][0]
                    else -1)

    return entry_points, top_or_bot_entry, dir_sign


def find_cocircular_point(new_polar, ref_angle, prev_ref_point, ref_radius,
                          entry_angle1, entry_angle2):
    resolution = np.sqrt(2) / 2.0
    # print 'new_polar',new_polar[new_polar[:,0].argsort(),:]
    # print 'ref_radius',ref_radius
    cocircular_points = new_polar[
        np.abs(new_polar[:, 0] - ref_radius) <= resolution, :]
    print 'reverting sign', np.abs(cocircular_points[:, 1] + pi) < 0.001
    cocircular_points[np.abs(cocircular_points[:, 1] + pi) < 0.001, 1] *= -1
    print 'cocircular_points', cocircular_points

    print 'segment_possible_angles', entry_angle1, entry_angle2

    check1 = np.abs(entry_angle1 - (cocircular_points[:, 1] + ref_angle))
    check2 = np.abs(entry_angle2 - (cocircular_points[:, 1] + ref_angle))
    _min1 = np.min(check1)
    _min2 = np.min(check2)
    if np.abs(_min1) < np.abs(_min2):
        print '1'
        farthest_cocirc_point = cocircular_points[
            check1 == _min1].ravel()[0:2]
    else:
        print '2'
        farthest_cocirc_point = cocircular_points[
            check2 == _min2].ravel()[0:2]
    print 'found point', farthest_cocirc_point
    print polar_to_cart(cocircular_points, prev_ref_point, ref_angle)
    far_cocirc_point_xy = [int(farthest_cocirc_point[0] *
                               (np.sin(farthest_cocirc_point[1] + ref_angle)) + prev_ref_point[0]),
                           int(farthest_cocirc_point[0] *
                               (np.cos(farthest_cocirc_point[1] + ref_angle)) + prev_ref_point[1])]

    print 'with xy', far_cocirc_point_xy
    return farthest_cocirc_point, far_cocirc_point_xy, cocircular_points


def find_box_defined_by_entry_segments_and_point(positions, entry_segment, point):
    '''
    Returns all positions belonging to an orthogonal, which has one side equal
    to the entry segment and all the other sides are defined by the other
    point, for which it is considered that it will belong to the opposite side
    of the side that is the entry_segment.
    '''
    entry_segment = np.array(entry_segment)
    e_diff = (entry_segment[1, :] - entry_segment[0, :])
    segment_mag2 = float(np.dot(e_diff, e_diff))
    _lambda0 = np.dot(entry_segment[
                      1, :] - entry_segment[0, :], point - entry_segment[0, :]) / segment_mag2
    # perp_to_unit_e has direction from point to segment
    # p_to_seg_vec has direction from point to segment ->mi ends up negative
    # when moving positively...
    p_to_seg_vec = _lambda0 * \
        (entry_segment[1, :] - entry_segment[0, :]) + \
        entry_segment[0, :] - point
    p_to_seg_dist = np.sqrt(_lambda0 * _lambda0 * segment_mag2 + np.dot(entry_segment[0, :] - point, entry_segment[0, :] - point) +
                            2 * _lambda0 * np.dot(e_diff, entry_segment[0, :] - point))
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
    flags = (_mi < _mi_max) * (_mi > _mi_min) * (_lambda < 1) * (_lambda > 0)
    return positions[flags, :], perp_to_unit_e


def find_1st_link_characteristics(binarm, positions, entry, top_or_bot_entry, dir_sign):
    if top_or_bot_entry == 0:
        entry = entry[entry[:, 0].argsort(), :]
    else:
        entry = entry[entry[:, 1].argsort(), :]

    # cv2.findContours(binarm,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    ref_radius = np.sqrt((entry[0][0] - entry[1][0])
                         ** 2 + (entry[0][1] - entry[1][1])**2) / 2
    if not top_or_bot_entry:
        if dir_sign > 0:  # means left
            ref_angle = 0.0
        else:  # means right
            ref_angle = -pi
    else:
        if dir_sign > 0:  # means top
            ref_angle = pi / 2
        else:  # means bot
            ref_angle = -pi / 2
    prev_ref_point = (entry[1, :] + entry[0, :]) / 2.0
    tmp = withLaplacian(binarm)
    # tmp=usingcontours(binarm)
    armpoints = positions[binarm > 0]
    edgepoints = positions[tmp > 0]
    points = edgepoints - prev_ref_point

    complex_entry = (entry[:, 0] - prev_ref_point[0]) * \
        1j + (entry[:, 1] - prev_ref_point[1])
    points = points[:, 0] * 1j + points[:, 1]
    tmp = np.angle(points) - ref_angle
    tmp[tmp < -pi] += 2 * pi
    tmp[tmp > pi] -= 2 * pi

    new_polar = np.concatenate(
        (np.absolute(points)[:, None], tmp[:, None]), axis=1)
    new_polar = new_polar[new_polar[:, 0].argsort(), :]

    entry_angle = np.angle(complex_entry) - ref_angle
    farthest_cocirc_point, far_cocirc_point_xy = find_cocircular_point(
        new_polar, ref_angle, prev_ref_point, ref_radius, entry_angle)

    tmp = [far_cocirc_point_xy -
           entry[0, :], far_cocirc_point_xy - entry[1, :]]

    incl = [np.arctan2(tmp[0][0], tmp[0][1]),
            np.arctan2(tmp[1][0], tmp[1][1])]
    # Names below_lo_line and above_hi_line refer to entry from the left
    if not top_or_bot_entry:
        if dir_sign == 1:  # left
            binarm_patch = binarm[entry[0][0]:entry[1][
                0] + 1, entry[0][1]:far_cocirc_point_xy[1] + 1]
        else:  # right
            binarm_patch = binarm[entry[0][0]:entry[1][
                0] + 1, far_cocirc_point_xy[1]:entry[0][1] + 1]

        patch_points = np.transpose(np.nonzero(binarm_patch))
        # Names refer to origin point, so they appear opposite in the image
        if dir_sign == 1:  # left
            below_lo_line = np.sum(binarm_patch[tuple(
                (patch_points[patch_points[:, 0] < (np.tan(incl[0]) * patch_points[:, 1]), :]).T)])
            above_hi_line = np.sum(binarm_patch[tuple((patch_points[patch_points[
                :, 0] - (entry[1][0] - entry[0][0]) > np.tan(incl[1]) * (
                    patch_points[:, 1]), :]).T)])
        else:  # right
            below_lo_line = np.sum(binarm_patch[tuple((patch_points[patch_points[:, 0] < np.tan(
                incl[0]) * (patch_points[:, 1] - (entry[0][1] - far_cocirc_point_xy[1])), :]).T)])
            above_hi_line = np.sum(binarm_patch[
                tuple((patch_points[patch_points[:, 0] - (entry[1][0] - entry[
                    0][0]) > np.tan(incl[1]) * (
                        patch_points[:, 1] - (
                            entry[1][1] - far_cocirc_point_xy[1])), :]).T)])

    else:
        if dir_sign == 1:  # top
            binarm_patch = binarm[entry[0][0]:far_cocirc_point_xy[
                0] + 1, entry[0][1]:entry[1][1] + 1]
        else:  # bottom
            binarm_patch = binarm[far_cocirc_point_xy[0]:entry[
                0][0] + 1, entry[0][1]:entry[1][1] + 1]
        patch_points = np.transpose(np.nonzero(binarm_patch))
        if dir_sign == 1:  # top
            below_lo_line = np.sum(binarm_patch[tuple(
                (patch_points[patch_points[:, 0] > np.tan(incl[0]) * (patch_points[:, 1]), :]).T)])
            above_hi_line = np.sum(binarm_patch[tuple((patch_points[patch_points[:, 0] > np.tan(
                incl[1]) * (patch_points[:, 1] - (entry[1][1] - entry[0][1])), :]).T)])
        else:  # bottom
            below_lo_line = np.sum(binarm_patch[tuple((patch_points[patch_points[:, 0] < np.tan(
                incl[0]) * (patch_points[:, 1] - (entry[0][0] - far_cocirc_point_xy[0])), :]).T)])
            above_hi_line = np.sum(binarm_patch[tuple((patch_points[patch_points[:, 0] - (
                entry[0][0] - far_cocirc_point_xy[
                    0]) < np.tan(incl[1]) * (patch_points[:, 1] -
                                             (entry[1][1] - entry[0][1])), :]).T)])
    if below_lo_line > above_hi_line:
        corrected_ref_angle = (entry_angle[0] +
                               farthest_cocirc_point[1]) / 2.0 + ref_angle
    else:
        corrected_ref_angle = (entry_angle[1] +
                               farthest_cocirc_point[1]) / 2.0 + ref_angle


def mod_correct(points):
    points[points[:, 1] > pi, 1] -= 2 * pi
    points[points[:, 1] < -pi, 1] += 2 * pi


def main(binarm3d, positions):
    # 0
    '''picture_entry(binarm3d,binarm_patch,dir_sign,entry,far_cocirc_point_xy,incl,
                  patch_points,ref_angle,prev_ref_point,ref_radius,
                  top_or_bot_entry)
    '''
    #
    binarm = binarm3d[:, :, 0]
    co.masks.calib_edges = np.pad(np.zeros((binarm.shape[
        0] - 2, binarm.shape[1] - 2), np.uint8), ((1, 1), (1, 1)), 'constant', constant_values=1)
    armpoints = positions[binarm > 0]
    tmp = withLaplacian(binarm)
    edgepoints = positions[tmp > 0]
    points = edgepoints
    points = points[:, 0] * 1j + points[:, 1]
    tmp = np.angle(points)
    tmp[tmp < -pi] += 2 * pi
    tmp[tmp > pi] -= 2 * pi
    new_polar = np.concatenate(
        (np.absolute(points)[:, None], tmp[:, None]), axis=1)
    new_polar = new_polar[new_polar[:, 0].argsort(), :]
    entry, top_or_bot_entry, dir_sign = detect_entry(binarm, positions)
    link_end_radius = 1 / 2.0 * \
        np.sqrt((entry[0][0] - entry[1][0])**2 +
                (entry[0][1] - entry[1][1])**2)
    link_end_segment = entry[:]
    new_ref_point = [0, 0]
    new_ref_angle = 0
    new_crit_ind = 0
    link_end_2nd = []
    resolution = np.sqrt(2) / 2.0
    new_corrected_segment = entry
    print entry
    for _count in range(3):
        #
        prev_ref_point = new_ref_point[:]
        prev_polar = new_polar[:]
        prev_ref_angle = new_ref_angle
        prev_ref_radius = link_end_radius
        prev_ref_point = new_ref_point[:]
        prev_crit_ind = new_crit_ind
        prev_corrected_segment = new_corrected_segment[:]

        time1 = time.clock()
        new_ref_point = [(link_end_segment[0][0] + link_end_segment[1][0]) /
                         2.0, (link_end_segment[0][1] + link_end_segment[1][1]) / 2.0]
        segment_angle = np.angle((link_end_segment[
                                 :, 0] - new_ref_point[0]) * 1j + link_end_segment[:, 1] - new_ref_point[1])

        # print 'new_ref_point',new_ref_point
        # new_polar=polar_change_origin(next_polar.copy(),corrected_ref_angle,prev_ref_point,new_ref_point)
        # print 'before polar change',\
        # prev_polar[prev_polar[:,0].argsort(),:][:40,:]
        new_polar = polar_change_origin(
            prev_polar.copy(), prev_ref_angle, prev_ref_point, new_ref_point)

        new_ref_radius = np.sqrt((link_end_segment[0][0] - link_end_segment[1][0])**2 + (
            link_end_segment[0][1] - link_end_segment[1][1])**2) / 2.0
        print segment_angle
        print 'ref_angle', new_ref_angle
        segment_angle = (segment_angle[0] + segment_angle[1]) / 2.0
        angle1 = segment_angle
        angle2 = segment_angle + pi
        if segment_angle == 0:
            angle2 = pi
        if segment_angle == pi:
            angle1 = 0
        if angle1 < -pi:
            angle1 += 2 * pi
        elif angle1 > pi:
            angle1 -= 2 * pi

        if angle2 < -pi:
            angle2 += 2 * pi
        elif angle2 > pi:
            angle2 -= 2 * pi
        farthest_cocirc_point, far_cocirc_point_xy, cocircular_points = find_cocircular_point(
            new_polar, 0, new_ref_point, new_ref_radius, angle1, angle2)
        print 'entry', entry
        print link_end_segment, far_cocirc_point_xy
        box, perp_to_segment_unit = find_box_defined_by_entry_segments_and_point(
            armpoints.reshape((-1, 2)), link_end_segment, np.array(far_cocirc_point_xy))
        new_ref_angle, new_corrected_segment = find_link_direction(
            box, link_end_segment, perp_to_segment_unit, np.array(far_cocirc_point_xy))
        print 'corrected_segment', new_corrected_segment
        # print 'new_ref_radius',new_ref_radius
        new_polar[:, 1] -= (new_ref_angle)
        mod_correct(new_polar)
        new_polar = new_polar[new_polar[:, 0].argsort(), :]
        new_polar = new_polar[new_polar[:, 0] >= new_ref_radius, :]
        new_crit_ind = np.argmin(np.abs(new_polar[:, 1]))
        # print 'new_crit_ind',new_crit_ind
        print new_crit_ind
        
        cocircular_crit = new_polar[
            np.abs(new_polar[new_crit_ind, 0] -new_polar[:, 0]) < resolution, :]
        cocircular_crit= cocircular_crit[cocircular_crit[:,1].argsort(),:]
        print cocircular_crit
        cocircular_crit_point_dists =\
        ((cocircular_crit[:-1,0]+cocircular_crit[1:,0])/2)* np.sqrt(2 * (
            1 - np.cos(mod_diff(cocircular_crit[:-1,1], cocircular_crit[1:,1])[0])))
        print 'new_crit_ind', new_crit_ind
        tmp = polar_to_cart(new_polar, new_ref_point, new_ref_angle)
        binarm3d[tuple(tmp.T)] = [255, 255, 0]
        binarm3d[tuple(tmp[np.abs(np.sqrt((tmp[:, 0] - new_ref_point[0])**2 + (tmp[:, 1] -
                                                                               new_ref_point[1])**2)
                                  - new_polar[new_crit_ind, 0]) <= resolution].T)] = [255, 255, 0]
        binarm3d[tuple(polar_to_cart(new_polar[np.abs(new_polar[new_crit_ind, 0] -
                                                          new_polar[:, 0]) < 0.1, :], new_ref_point, new_ref_angle).T)] = [255, 0, 255]
        print 'cocircular_crit_point_dists', cocircular_crit_point_dists
        # for ind in range(new_crit_ind):

        time2 = time.clock()
        print time2 - time1, 's'

        # binarm3d[tuple(polar_to_cart(new_polar,new_ref_point,new_ref_angle).T)]=[0,0,255]
        binarm3d[np.abs(np.sqrt((positions[:, :, 0] - new_ref_point[0])**2 + (positions[
                        :, :, 1] - new_ref_point[1])**2) - new_ref_radius) <= resolution] = [255, 255, 0]
        binarm3d[link_end_segment[0][0], link_end_segment[0][1]] = [0, 0, 255]
        binarm3d[link_end_segment[1][0], link_end_segment[1][1]] = [0, 0, 255]
        binarm3d[new_ref_point[0], new_ref_point[1]] = [255, 0, 0]
        binarm3d = picture_box(binarm3d, box)
        cv2.line(binarm3d, (new_corrected_segment[0][1], new_corrected_segment[0][0]),
                 (new_corrected_segment[1][1], new_corrected_segment[1][0]), [0, 0, 255])
        cv2.arrowedLine(binarm3d, (int(new_ref_point[1]), int(new_ref_point[0])), (
            int(new_ref_point[1] + new_ref_radius * np.cos(new_ref_angle)),
            int(new_ref_point[0] + new_ref_radius * np.sin(new_ref_angle))),
            [0, 0, 255], 2, 1)
        lo_threshold = new_ref_radius / 2
        hi_threshold = 3 * new_ref_radius
        print 'ref_radius', new_ref_radius
        print 'lo', lo_threshold
        print 'hi', hi_threshold
        print cocircular_crit_point_dists
        num = np.sum((cocircular_crit_point_dists < lo_threshold)*
                     (cocircular_crit_point_dists > 1)+
                     (cocircular_crit_point_dists > hi_threshold))
        
        if num:
            print 'num', num
            ref_dist = np.sqrt((new_corrected_segment[0][0] - new_corrected_segment[1][0])**2 +
                               (new_corrected_segment[0][1] - new_corrected_segment[1][1])**2)
            fl = new_polar[new_crit_ind::-1, 1].shape[0] - 1
            same_rad = []
            count = fl
            curr_rad = new_polar[new_crit_ind, 0]
            for (polar_point, polar_angle) in\
                    zip(new_polar[new_crit_ind::-1, :], new_polar[new_crit_ind::-1, 1]):

                if np.abs((polar_point[0] - curr_rad)) < resolution:
                    same_rad.append(polar_point)
                else:
                    same_rad = np.array(same_rad)

                    if len(same_rad.shape) > 1:
                        angles_bound = np.arctan2(
                            ref_dist / 2.0, polar_point[0]) + 0.1
                        print 'angles_bound', angles_bound

                        same_rad = same_rad[same_rad[:, 1].argsort()]

                        ang_thres = np.abs(same_rad[:, 1]) < angles_bound
                        same_rad_dists = (((same_rad[:-1, 0] + same_rad[1:, 0]) / 2) * np.sqrt(
                            2 * (1 - np.cos(mod_diff(same_rad[:-1, 1], same_rad[1:, 1])[0]))))
                        ang_thres = ang_thres[1:] * ang_thres[:-1]
                        threshold = same_rad_dists > 5

                        ang_thres = ang_thres[threshold]
                        tmp = np.where(threshold)[0]
                        true_inds = np.sort(np.unique(np.concatenate(
                            (np.array(tmp), np.array(tmp) + 1), axis=0)))
                        same_rad = same_rad[true_inds]
                        print 'with true_inds', true_inds
                        print 'with angles', same_rad[:, 1]
                        print 'and rads', same_rad[:, 0]
                        print 'and ang_thresh', ang_thres
                        same_rad_dists = same_rad_dists[threshold]
                        print 'and same_rad_dists', same_rad_dists

                        print 'with ref_dist and',\
                            'sampled_polar[fl,0]', ref_dist, curr_rad
                        # print 'and corrected_segment',\
                        #        new_corrected_segment
                        if same_rad_dists.size > 0 and same_rad.shape[0] == 2:
                            tmp = np.argmax(same_rad_dists)
                            if\
                                np.abs(same_rad_dists[tmp] - ref_dist) < ref_dist / 6.0 and\
                               same_rad_dists.size == 1 and ang_thres[tmp]:
                                # polar_wristpoints=np.pad(same_rad_angles[:,None],((0,0),(1,0)),'constant',constant_values=(curr_rad,))
                                print new_ref_point
                                print new_ref_angle
                                wristpoints = polar_to_cart(
                                    same_rad, new_ref_point, new_ref_angle)
                                # hand_center_of_mass=polar_to_cart(np.array([np.mean(new_polar[count:,:],axis=0)]),new_ref_point,new_ref_angle)
                                wrist_radius = np.sqrt((wristpoints[0][0] - wristpoints[1][0])**2 +
                                                       (wristpoints[0][1] - wristpoints[1][1])**2) / 2
                                # hand_center=((hand_center_of_mass[0,:]+(wristpoints[0,:]+wristpoints[1,:])/2.0)/2.0).astype(int)

                                binarm3d[tuple(polar_to_cart(np.array([np.mean(new_polar[count:, :], axis=0)]), new_ref_point, new_ref_angle).
                                               astype(int).T)] = [255, 0, 255]
                                hand_points = polar_to_cart(
                                    new_polar[count:], new_ref_point, new_ref_angle)
                                hand_box = binarm[np.min(hand_points[:, 0]):np.max(hand_points[:, 0]),
                                                  np.min(hand_points[:, 1]):np.max(hand_points[:, 1])]
                                # perform top hat transform, with suggested
                                # finger radius smaller than wrist radius
                                print 'wrist_radius', wrist_radius
                                print same_rad_dists[tmp]
                                struct_el = cv2.getStructuringElement(
                                    cv2.MORPH_ELLIPSE, tuple(2 * [int(0.9 * wrist_radius)]))
                                palm = cv2.morphologyEx(
                                    hand_box, cv2.MORPH_OPEN, struct_el)
                                fingers = hand_box - palm
                                fingers = cv2.morphologyEx(fingers, cv2.MORPH_OPEN, np.ones(
                                    tuple(2 * [int(struct_el.shape[0] / 4)]), np.uint8))
                                #print('struct_el size',struct_el.shape)
                                cv2.imshow('Fingers', fingers)
                                cv2.imshow('Palm', palm)
                                binarm3d[tuple(polar_to_cart(
                                    new_polar[count:], new_ref_point, new_ref_angle).T)] = [255, 0, 0]
                                binarm3d[tuple(wristpoints.T)] = [0, 0, 255]
                                # print 'Found wristpoints', wristpoints
                                #print('Found hand','center',hand_center)
                                # binarm3d[hand_center[0],hand_center[1]]=[0,0,255]
                                break
                    same_rad = []
                    curr_rad = polar_point[0]
                count = count - 1

        # binarm3d[tuple(polar_to_cart(new_polar,new_ref_point,new_ref_angle).T)]=[0,0,255]
        binarm3d[prev_corrected_segment[0][0],
                 prev_corrected_segment[0][1]] = [255, 255, 0]

        binarm3d[prev_corrected_segment[1][0],
                 prev_corrected_segment[1][1]] = [255, 255, 0]
        binarm3d[far_cocirc_point_xy[0],
                 far_cocirc_point_xy[1]] = [255, 0, 255]
        # break
        cv2.imshow('test', binarm3d)
        cv2.waitKey(0)
        link_end_1st = new_polar[new_crit_ind, :]
        link_end_radius = link_end_1st[0]
        tmp = new_polar[
            np.abs(new_polar[:, 0] - link_end_radius) < resolution, :]
        link_end_2nd = tmp[np.argmax(np.abs(tmp[:, 1])), :]
        link_end_segment[0] = polar_to_cart(
            np.array([link_end_1st]), new_ref_point, new_ref_angle)
        link_end_segment[1] = polar_to_cart(
            np.array([link_end_2nd]), new_ref_point, new_ref_angle)
        segment_angle = [pi / 2, -pi / 2]
        # print 'link_end_xy',link_end_segment
        new_polar = new_polar[new_crit_ind:, :]




def mod_diff(angles1, angles2):
    sgn = -1 + 2 * (angles1 > angles2)
    d1 = np.abs(angles1 - angles2)
    d2 = 2 * pi - d1
    return sgn * np.min([d1, d2], axis=0), np.argmin([d1, d2], axis=0)


def mod_between_vals(angles, min_bound, max_bound):
    res = mod_diff(max_bound, min_bound)[1]
    if res == 0:
        return((angles < max_bound) * (angles > min_bound))
    else:
        return((angles > max_bound) * (angles < pi) + (angles > -pi) * (angles < min_bound))


def find_link_direction(
        xy_points, entry_segment, perp_to_segment_unit, point):
    vecs = np.array(point) - np.array(entry_segment)
    incl = np.arctan2(vecs[:, 0], vecs[:, 1])
    init_incl_bounds = [np.arctan2(
        perp_to_segment_unit[0], perp_to_segment_unit[1])]
    if init_incl_bounds[0] + pi > pi:
        init_incl_bounds.append(init_incl_bounds[0] - pi)
    else:
        init_incl_bounds.append(init_incl_bounds[0] + pi)

    print 'init_incl_bounds', init_incl_bounds
    print 'incl', incl

    num = [0, 0]
    for count in range(2):
        complex_points = (xy_points[:, 0] - entry_segment[count][0]) * 1j + (xy_points[:, 1] -
                                                                             entry_segment[count][1])
        angles = np.angle(complex_points)

        incl_bound = init_incl_bounds[np.argmin(np.abs(mod_diff(
            np.array(init_incl_bounds), np.array([incl[count], incl[count]]))[0]))]
        incl_bounds = [incl[count], incl_bound]
        print 'incl_bound', incl_bound
        _max = max([incl[count], incl_bound])
        _min = min([incl[count], incl_bound])
        num[count] = np.sum(mod_between_vals(angles, _min, _max))
        '''
        if num[count]>xy_points.shape[0]/2:
            return incl[count],[entry_segment[count][:],point]
        elif num==0:
            return incl[(count+1)%2],[entry_segment[(count+1)%2][:],point]
        '''
    # Thales theorem:
    print 'link_direction_metric', num
    tmp = np.array(point) - \
        np.array(entry_segment[(np.argmax(num) + 1) % 2][:])

    return np.arctan2(tmp[0], tmp[1]), [entry_segment[np.argmax(num)][:], point]


def picture_box(binarm3d, points):
    tmp = np.zeros_like(binarm3d[:, :, 0])
    tmp[tuple(points.T)] = 255
    binarm3d[tmp[:, :] > 0] = [0, 255, 0]
    return binarm3d


def picture_entry(binarm3d, binarm_patch, dir_sign, entry, far_cocirc_point_xy, incl,
                  patch_points, ref_angle, prev_ref_point, ref_radius,
                  top_or_bot_entry):
    binarm3d[tuple(prev_ref_point.astype(int))] = [255, 0, 0]
    binarm_patch1 = binarm_patch.copy()
    binarm_patch2 = binarm_patch.copy()
    if not top_or_bot_entry:
        if dir_sign == 1:  # left
            binarm_patch1[tuple((patch_points[patch_points[:, 0] < (
                np.tan(incl[0]) * patch_points[:, 1]), :]).T)] = 3
        else:  # right
            binarm_patch1[tuple((patch_points[patch_points[:, 0] < np.tan(
                incl[0]) * (patch_points[:, 1] - (entry[0][1] -
                                                  far_cocirc_point_xy[1])), :]).T)] = 3
    else:
        if dir_sign == 1:  # top
            binarm_patch1[tuple((patch_points[patch_points[:, 0] > np.tan(
                incl[0]) * (patch_points[:, 1]), :]).T)] = 3
        else:  # bottom
            binarm_patch1[tuple((patch_points[patch_points[:, 0] < np.tan(
                incl[0]) * (patch_points[:, 1] - (entry[0][0] -
                                                  far_cocirc_point_xy[0])), :]).T)] = 3
    binarm_patch1 = (binarm_patch1 == 3) * 255
    if not top_or_bot_entry:
        if dir_sign == 1:  # left
            binarm_patch2[tuple((patch_points[patch_points[
                :, 0] - (entry[1][0] -
                         entry[0][0]) > np.tan(incl[1]) * (
                             patch_points[:, 1]), :]).T)] = 3
        else:  # right
            binarm_patch2[tuple((patch_points[patch_points[:, 0] - (
                entry[1][0] - entry[0][0]) > np.tan(
                    incl[1]) * (patch_points[:, 1] - (
                        entry[1][1] - far_cocirc_point_xy[1])), :]).T)] = 3
    else:
        if dir_sign == 1:  # top
            binarm_patch2[tuple((patch_points[patch_points[:, 0] > np.tan(
                incl[1]) * (patch_points[:, 1] - (entry[1][1] - entry[0][1])), :]).T)] = 3
        else:  # bottom
            binarm_patch2[tuple((patch_points[patch_points[:, 0] - (
                entry[0][0] - far_cocirc_point_xy[
                    0]) < np.tan(incl[1]) * (
                        patch_points[:, 1] - (entry[1][1] - entry[0][1])), :]).T)] = 3

    binarm_patch2 = (binarm_patch2 == 3) * 255
    tmp = np.concatenate((np.zeros(
        binarm_patch2.shape + (
            1,)), binarm_patch2[:, :, None], np.zeros(
                binarm_patch2.shape + (1,))), axis=2) +\
        np.concatenate((binarm_patch1[:, :, None], np.zeros(
            binarm_patch1.shape + (2,))), axis=2)
    if top_or_bot_entry:
        if dir_sign == 1:
            binarm3d[entry[0][0]:far_cocirc_point_xy[
                0] + 1, entry[0][1]:entry[1][1] + 1] = tmp
        else:
            binarm3d[far_cocirc_point_xy[0]:entry[0][0] +
                     1, entry[0][1]:entry[1][1] + 1, :] = tmp
    else:
        if dir_sign == 1:
            binarm3d[entry[0][0]:entry[1][0] + 1, entry[0]
                     [1]:far_cocirc_point_xy[1] + 1, :] = tmp
        else:
            binarm3d[entry[0][0]:entry[1][0] + 1,
                     far_cocirc_point_xy[1]:entry[0][1] + 1] = tmp
    # binarm3d[entry[0][0]:entry[1][0]+1,entry[0][1]:far_cocirc_point_xy[1]+1:dir_sign,:]=\

    if not top_or_bot_entry:
        tmp = np.arange(entry[1][1], far_cocirc_point_xy[1] + 1, dir_sign)
        tmp4 = np.concatenate(((entry[1][0] + np.tan(incl[1]) * (tmp - entry[1][1]))[
            :, None], tmp[:, None]), axis=1).astype(int)
    else:
        tmp = np.arange(entry[1][0], far_cocirc_point_xy[0] + 1, dir_sign)
        tmp4 = np.concatenate((tmp[:, None], (entry[1][
            1] + np.tan(incl[1]) * (tmp - entry[1][0]))[:, None]), axis=1).astype(int)
    for point in tmp4:
        binarm3d[tuple(point)] = [0, 0, 255]
    cv2.arrowedLine(binarm3d, (int(prev_ref_point[1]), int(prev_ref_point[0])), (
        int(prev_ref_point[1] + ref_radius * np.cos(ref_angle)),
        int(prev_ref_point[0] + ref_radius * np.sin(ref_angle))),
        [0, 0, 255], 2, 1)
    cv2.imshow('Entry', binarm3d)
    cv2.waitKey(0)

if not os.path.exists('arm_example.png'):
    urllib.urlretrieve("https://www.packtpub.com/\
                       sites/default/files/Article-Images/B04521_02_04.png",
                       "arm_example.png")
binarm3d = cv2.imread('arm_example.png')
binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
    binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))

# timeit.timeit(lambda:main(binarm3d.copy(),binarm3d_positions),number=100)/100
# gives 0.0029, might get better if I use more the binarm3d_positions

#main(binarm3d.copy(), binarm3d_positions)

for c in range(4):
    rows, cols, _ = binarm3d.shape
    M = cv2.getRotationMatrix2D(
        (np.floor(cols / 2.0), np.floor(rows / 2.0)), 90, 1)
    M[0, 2] += np.floor(rows / 2.0 - cols / 2.0)
    M[1, 2] += np.floor(cols / 2.0 - rows / 2.0)
    binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
        binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
    #if c == 3:
    main(binarm3d.copy(), binarm3d_positions)
    binarm3d = 255 * \
        ((cv2.warpAffine(binarm3d, M, (rows, cols))) > 0).astype(np.uint8)
