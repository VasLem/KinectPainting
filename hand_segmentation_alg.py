'''Input: Hand Mask nonzero xy_points
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
import time
import timeit
import urllib
import cProfile as profile
from matplotlib import pyplot as plt
import numpy as np
import cv2
import class_objects as co
import palm_detection_alg as pda

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

def find_rows_in_array(rows,arr):
    tmp=np.prod(np.swapaxes(
            rows[:,:,None],1,2)==arr,axis=2).T
    return np.sum(np.cumsum(tmp,axis=1)*tmp==1,
        axis=0)>0
def array_row_intersection(arr1,arr2):
    return arr1[find_rows_in_array(arr1,arr2)]


def detect_entry(bin_mask,positions):
    '''function to detect  intersection limits of mask with calib_edges'''
    #_,entries,_=cv2.findContours(cv2.dilate(co.masks.calib_edges
    #*bin_mask,np.ones((3,3),np.uint8)),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #entries_length=[cv2.arcLength(entry,1) for entry in entries]
    #(entries, entries_length) = zip(*sorted(zip(entries, entries_length),
    #                                               key=lambda b:b[1],
    #                                        reverse=1))
    # entry_points=cv2.convexHull(entries[0],0)
    #entry_segments=cv2.findContours((co.masks.calib_edges*bin_mask>0).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    '''
    #Assuming convex calib edges
    '''
    #entry_segments=positions[bin_mask*co.masks.calib_edges>0]
    entry_segments=co.meas.edges_positions[bin_mask[co.meas.edges_positions_indices]>0]
    if entry_segments.shape[0]==0:
        print 'No entry found'
        return([])
    approx_entry_segments=entry_segments.copy()
    approx_entry_segments[
        entry_segments[:,1]<
        co.meas.nonconvex_edges_lims[0],1]=co.meas.nonconvex_edges_lims[0]
    approx_entry_segments[
        entry_segments[:,0]<
        co.meas.nonconvex_edges_lims[1],0]=co.meas.nonconvex_edges_lims[1]
    approx_entry_segments[
        entry_segments[:,1]>
        co.meas.nonconvex_edges_lims[2],1]=co.meas.nonconvex_edges_lims[2]
    approx_entry_segments[
        entry_segments[:,0]>
        co.meas.nonconvex_edges_lims[3],0]=co.meas.nonconvex_edges_lims[3]
    approx_entry_points=cv2.convexHull(approx_entry_segments).squeeze()
    if approx_entry_points.shape[0]==2:
        if calculate_cart_dists(approx_entry_points)>np.min(bin_mask.shape)/10.0:
            return\
            [entry_segments[find_rows_in_array(approx_entry_segments,approx_entry_points)]]
        else:
            return (np.array([[]]))
    approx_entry_orient=np.diff(approx_entry_points,axis=0)
    approx_entry_orient=(approx_entry_orient/
                         calculate_cart_dists(
                             approx_entry_points)[:,None])
    approx_entry_vert_orient=np.dot(approx_entry_orient,np.array([[0,-1],[1,0]]))
    num=[]
    for count,orient in enumerate(approx_entry_vert_orient):
        pos=find_segment_to_point_box(approx_entry_segments,
                                      np.array([approx_entry_points[count,:]+orient*10,
                                                approx_entry_points[count,:]-orient*10]),
                                      approx_entry_points[count+1,:])[0]
        num.append(pos.shape[0])
    _argmax=np.argmax(num)
    entry_points=entry_segments[find_rows_in_array(approx_entry_segments,approx_entry_points[_argmax:_argmax+2,:])]
    return ([entry_points])

def detect_entry2(bin_mask):
    '''function to detect  intersection limits of mask with calib_edges'''
    #_,entries,_=cv2.findContours(cv2.dilate(co.masks.calib_edges
    #*bin_mask,np.ones((3,3),np.uint8)),cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    #entries_length=[cv2.arcLength(entry,1) for entry in entries]
    #(entries, entries_length) = zip(*sorted(zip(entries, entries_length),
    #                                               key=lambda b:b[1],
    #                                        reverse=1))
    # entry_points=cv2.convexHull(entries[0],0)
    entry_segments=cv2.findContours((co.masks.calib_edges*bin_mask>0).astype(np.uint8),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[1]
    entry=[]
    for entry_segment in entry_segments:
        '''
        if entry_segment.shape[1]==2:
            entry.append(np.fliplr(entry_segment))
        else:
        '''
        if cv2.arcLength(
                entry_segment,0)>min(
                    [bin_mask.shape[0],
                     bin_mask.shape[1]])/20.0:
                rect=cv2.minAreaRect(entry_segment)
                box = cv2.boxPoints(rect)
                entry.append(array_row_intersection(entry_segment.squeeze(),box))
    return entry

def find_cocircular_points(polar_points, radius, resolution):
    '''
    Find cocircular points given radius and suggested resolution
    '''
    return polar_points[np.abs(polar_points[:, 0] - radius) <= resolution, :]


def find_cocircular_farthest_point(new_polar, ref_angle, prev_ref_point, ref_radius,
                                   entry_angles):
    '''
    Find farthest cocircular point from entry segment
    '''
    resolution = np.sqrt(2) / 2.0
    cocircular_points = find_cocircular_points(
        new_polar, ref_radius, resolution)
    cocircular_points[np.abs(cocircular_points[:, 1] + pi) < 0.001, 1] *= -1
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
                               (np.sin(farthest_cocirc_point[1] + ref_angle)) + prev_ref_point[0]),
                           int(farthest_cocirc_point[0] *
                               (np.cos(farthest_cocirc_point[1] + ref_angle)) + prev_ref_point[1])]

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
    if p_to_seg_dist==0:
        return(np.array([[]]))
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

def calculate_cart_dists(cart_points,cart_point=[]):
    '''
    Input either numpy array either 2*2 list
    Second optional argument is a point
    '''
    if cart_point==[]:

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
    return np.fliplr(cv2.findNonZero(arr).squeeze())
def main_process(binarm3d, positions, display):
    '''Main processing function'''
    if len(binarm3d.shape)==3:
        binarm = binarm3d[:, :, 0].copy()
        if np.issubdtype(binarm3d[0,0,0],np.uint8):
            if np.max(binarm3d)==1:
                binarm3d*=255
        else:
            binarm3d=(255*binarm3d).astype(np.uint8)
    else:
        binarm=binarm3d.copy()
        if display:
            binarm3d=np.tile(binarm[:,:,None],(1,1,3))
            if np.issubdtype(binarm3d[0,0,0],np.uint8):
                if np.max(binarm3d)==1:
                    binarm3d*=255
            else:
                binarm3d=(255*binarm3d).astype(np.uint8)
    time1=time.clock()
    try:
        armpoints = find_nonzero(binarm)
    except AttributeError: # binarm is []
        return np.array([[]])
    points = find_nonzero(with_laplacian(binarm))
    points_size=points.shape[0]
    points = points[:, 0] * 1j + points[:, 1]
    tmp =  np.angle(points)
    tmp[tmp < -pi] += 2 * pi
    tmp[tmp > pi] -= 2 * pi
    new_polar = np.concatenate(
        (np.absolute(points)[:, None], tmp[:, None]), axis=1)
    try:
        entry = detect_entry(binarm,positions)[0]
    except IndexError:
        return np.array([[]])
    if entry.shape[0]<=1:
        print 'Arm in image corners, cannot produce fast results'
        return np.array([[]])
    link_end_radius = 1 / 2.0 * calculate_cart_dists(entry)
    link_end_segment = entry[:]
    new_ref_point = [0, 0]
    new_ref_angle = 0
    new_crit_ind = 0
    link_end_2nd = []
    resolution = np.sqrt(2) / 2.0
    new_corrected_segment = entry[:]
    #for _count in range(3):
    while True:
        prev_ref_point = new_ref_point[:]
        prev_polar = new_polar[:]
        prev_ref_angle = new_ref_angle
        prev_ref_point = new_ref_point[:]
        prev_corrected_segment = new_corrected_segment[:]
        prev_crit_ind=new_crit_ind
        if new_crit_ind>points_size-10:
            co.im_results.images.append(binarm3d)
            return np.array([[]])
        new_ref_point = [(link_end_segment[0][0] + link_end_segment[1][0]) /
                         2.0, (link_end_segment[0][1] + link_end_segment[1][1]) / 2.0]
        segment_angle = np.angle((link_end_segment[
            :, 0] - new_ref_point[0]) * 1j + link_end_segment[:, 1] - new_ref_point[1])

        new_polar = polar_change_origin(
            prev_polar.copy(), prev_ref_angle, prev_ref_point, new_ref_point)
        new_ref_radius = calculate_cart_dists(link_end_segment) / 2.0
        segment_angle = (segment_angle[0] + segment_angle[1]) / 2.0
        angle1 = segment_angle
        angle2 = segment_angle + pi
        angle1 = fix_angle(angle1)
        angle2 = fix_angle(angle2)
        try:
            (_, far_cocirc_point_xy, _) = find_cocircular_farthest_point(
                new_polar, 0, new_ref_point, new_ref_radius, [angle1, angle2])
            box, perp_to_segment_unit = find_segment_to_point_box(
                armpoints.reshape((-1, 2)), link_end_segment, np.array(far_cocirc_point_xy))
            new_ref_angle, new_corrected_segment = find_link_direction(
                box, link_end_segment, perp_to_segment_unit, np.array(far_cocirc_point_xy))
        except:
            tmp=link_end_segment[0,:]-link_end_segment[1,:]
            par_to_segment_unit=tmp/calculate_cart_dists(link_end_segment)
            seg_angle=np.arctan2(par_to_segment_unit[0],par_to_segment_unit[1])
            perp_to_segment_unit=np.dot(par_to_segment_unit,np.array([[0,-1],[1,0]]))
            point=np.mean(link_end_segment,axis=0)
            angdiff1=mod_diff(new_polar[:,1],seg_angle)
            angdiff2=mod_diff(new_polar[:,1], fix_angle(seg_angle+pi))
            tmp=new_polar[(np.abs(angdiff1)>=0.1)*(np.abs(angdiff2)>=0.1),:]
            closest_cocirc_polar_part=tmp[np.abs(tmp[:,0]-new_ref_radius)<new_ref_radius,:]
            closest_cocirc_polar=closest_cocirc_polar_part[
                np.argmin(mod_diff(closest_cocirc_polar_part[:,1],
                                      np.arctan2(par_to_segment_unit[0],par_to_segment_unit[1]))),:]
            far_cocirc_point_xy=polar_to_cart(np.array([closest_cocirc_polar]),new_ref_point,0)[0,:]
            new_ref_angle, new_corrected_segment = find_link_direction(
                np.concatenate((link_end_segment,(point)[None,:]),axis=0),link_end_segment,
                perp_to_segment_unit,far_cocirc_point_xy)
        new_polar[:, 1] -= (new_ref_angle)
        mod_correct(new_polar)
        new_polar = new_polar[new_polar[:, 0] >= new_ref_radius, :]
        new_polar = new_polar[new_polar[:, 0].argsort(), :]
        cand_crit_points = new_polar[np.abs(new_polar[:,1])<0.01,:]
        if len(cand_crit_points)==0:
            co.im_results.images.append(binarm3d)
            print 'No cocircular points found, reached end of hand'
            return np.array([[]])
        _min = cand_crit_points[np.argmin(np.abs(cand_crit_points[:, 0])),:]
        new_crit_ind=np.where(new_polar==_min)[0][0]
        cocircular_crit = find_cocircular_points(new_polar,
                                                 new_polar[new_crit_ind, 0],
                                                 resolution)
        '''
        new_crit_ind=np.argmin(np.abs(new_polar[:,1]))
        cocircular_crit = find_cocircular_points(new_polar,
                                                 new_polar[new_crit_ind,0],
                                                 resolution)
        '''
        if cocircular_crit==[]:
            print 'Reached end of hand'
            co.im_results.images.append(binarm3d)
            return np.array([[]])
        cocircular_crit = cocircular_crit[cocircular_crit[:, 1].argsort(), :]
        crit_chords = calculate_chords_lengths(cocircular_crit)



        if display:
            tmp = polar_to_cart(new_polar, new_ref_point, new_ref_angle)
            #binarm3d[tuple(tmp.T)] = [255, 255, 0]
            binarm3d[tuple(polar_to_cart(cocircular_crit,new_ref_point,new_ref_angle).T)]=[255,255,0]
            binarm3d[tuple(tmp[np.abs(np.sqrt((tmp[:, 0] - new_ref_point[0])**2
                                              + (tmp[:, 1] - new_ref_point[1])**2)
                                      - new_polar[new_crit_ind, 0]) <=
                               resolution].T)] = [255, 255, 0]
            binarm3d[tuple(polar_to_cart(new_polar[np.abs(new_polar[new_crit_ind, 0] -
                                                          new_polar[:, 0]) < 0.1, :],
                                         new_ref_point, new_ref_angle).T)] = [255, 0, 255]
            binarm3d[np.abs(np.sqrt((positions[:, :, 0] - new_ref_point[0])**2 + (positions[
                :, :, 1] - new_ref_point[1])**2) - new_ref_radius) <= resolution] = [255, 255, 0]
            binarm3d[link_end_segment[0][0], link_end_segment[0][1]] = [0, 0, 255]
            binarm3d[link_end_segment[1][0], link_end_segment[1][1]] = [0, 0, 255]
            binarm3d[int(new_ref_point[0]), int(new_ref_point[1])] = [255, 0, 0]
            try:
                binarm3d = picture_box(binarm3d, box)
            except:
                pass
            cv2.line(binarm3d, (new_corrected_segment[0][1], new_corrected_segment[0][0]),
                     (new_corrected_segment[1][1], new_corrected_segment[1][0]), [0, 0, 255])
            
            cv2.arrowedLine(binarm3d, (int(new_ref_point[1]), int(new_ref_point[0])), (
                int(new_ref_point[1] + new_ref_radius * np.cos(new_ref_angle)),
                int(new_ref_point[0] + new_ref_radius * np.sin(new_ref_angle))),
                            [0, 0, 255], 2, 1)

        width_lo_thres = new_ref_radius / 2
        #width_hi_thres = 3 * new_ref_radius
        reached_abnormality = np.sum((crit_chords < width_lo_thres) *
                                 (crit_chords > 1))
                                 #(crit_chords > width_hi_thres))
        #reached_fingers=0
        if reached_abnormality:
            hand_points=find_hand(binarm,binarm3d,armpoints,display,new_polar,new_corrected_segment,new_ref_angle,new_crit_ind,new_ref_point,resolution)
        if display:

            binarm3d[int(far_cocirc_point_xy[0]),
                     int(far_cocirc_point_xy[1])] = [255, 0, 255]
            #cv2.namedWindow('test',cv2.GUI_EXPANDED)
            if  __name__=='__main__':
                cv2.imshow('test', binarm3d)
                cv2.waitKey(0)
        if reached_abnormality:
            #print 'Found abnormality'
            co.im_results.images.append(binarm3d)
            return hand_points
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
        new_polar = new_polar[new_crit_ind:, :]

def mod_diff(angles1, angles2,ret_argmin=0):
    '''
    Angle substraction using modulo in (-pi,pi)
    '''
    sgn = -1 + 2 * (angles1 > angles2)
    if len(angles1.shape)==0:

        diff1 = np.abs(angles1 - angles2)
        diff2 = 2 * pi - diff1
        if ret_argmin:
            return sgn * min([diff1,diff2]), np.argmin([diff1,diff2])
        else:
            return  sgn * min([diff1,diff2])
    diff=np.empty((2,angles1.shape[0]))
    diff[0,:] = np.abs(angles1 - angles2)
    diff[1,:] = 2 * pi - diff[0,:]
    if ret_argmin:
        return sgn * np.min(diff, axis=0), np.argmin(diff, axis=0)
    else:
        return sgn * np.min(diff, axis=0)


def mod_between_vals(angles, min_bound, max_bound):
    '''
    Find angles between bounds, using modular (-pi,pi) logic
    '''
    if max_bound==min_bound:
        return np.zeros((0))
    res = mod_diff(max_bound, min_bound,1)[1]
    if res == 0:
        return (angles <= max_bound) * (angles >= min_bound)
    else:
        return ((angles >= max_bound) * (angles <= pi)
                + (angles >= -pi) * (angles <= min_bound))


from scipy import ndimage
from scipy import linalg


def find_hand(*args):
    #binarm,polar,ref_angle,ref_point,crit_ind,corrected_segment,resolution,display,binarm3d
    [binarm,binarm3d,armpoints,display,polar,corrected_segment,
     ref_angle,crit_ind,ref_point,resolution]=args[0:10]
    ref_dist = calculate_cart_dists(corrected_segment)
    same_rad = []
    count = crit_ind
    curr_rad = polar[crit_ind, 0]
    bins=np.arange(resolution,np.max(polar[:crit_ind+1,0])+resolution,resolution)
    dig_rad=np.digitize(polar[crit_ind::-1,0],bins)
    angles_bound=np.abs(np.arctan((ref_dist/2.0)/(bins)))+0.1
    angles_bound=angles_bound[dig_rad]
    angles_thres=np.abs(polar[crit_ind::-1,1])<angles_bound
    compmat=dig_rad[:,None]==np.arange(bins.shape[0])
    
    sameline=np.sum((((angles_thres[:,None]-1)*compmat)<0),axis=0)==0
    dig_rad_thres=np.sum(compmat[:,sameline],axis=1)>0
    used_polar=polar[crit_ind::-1,:][dig_rad_thres]
    sort_inds=np.lexsort((used_polar[:,1],bins[dig_rad[dig_rad_thres]]))[::-1]
    used_polar=used_polar[sort_inds,:]
    same_rad_dists=calculate_chords_lengths(used_polar)
    
    dist_threshold=np.abs(same_rad_dists-ref_dist)<=2
    dist_threshold=np.concatenate((dist_threshold,np.array([False])))
    try:
        ind=np.where(dist_threshold)[0][0]
        chosen=used_polar[ind:ind+2,:]
        
        '''
        if display:
       
            fig=plt.figure()
            ax1=fig.add_subplot(111)
            ax1.scatter(bins[dig_rad[dig_rad_thres]][sort_inds],used_polar[:,1])
            
            plt.draw()
            plt.pause(0.1)
            plt.waitforbuttonpress(timeout=-1)
            plt.close(fig)
        '''
        wristpoints = polar_to_cart(
            chosen, ref_point, ref_angle)
        wrist_radius = calculate_cart_dists(
            wristpoints) / 2.0

        hand_edges = polar_to_cart(
            polar[ind:,:], ref_point, ref_angle)
        
        '''
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
        '''
        if display:
            hand_edges=polar_to_cart(
                polar[ind:], ref_point,
                ref_angle)
            convexhand=cv2.convexHull(hand_edges)
            cv2.drawContours(binarm3d,convexhand,0,[0,0,255],2)
            binarm3d[tuple(polar_to_cart(np.array(
                [np.mean(polar[ind:, :],
                         axis=0)]), ref_point, ref_angle).
                           astype(int).T)] = [255, 0, 255]
            
            '''
            cv2.imshow('Fingers', fingers)
            cv2.imshow('Palm', palm)
            '''
            binarm3d[tuple(hand_edges.T)] = [255, 0, 0]
            binarm3d[tuple(wristpoints.T)] = [0, 0, 255]
            
        #binarm3d[tuple(polar_to_cart(polar[polar[:,1]>0],ref_point,ref_angle).T)]=[255,0,0]
        
        handpoints=armpoints[(armpoints[:,0]>np.min(hand_edges[:,0]))*
                             (armpoints[:,0]<np.max(hand_edges[:,0]))*
                             (armpoints[:,1]>np.min(hand_edges[:,1]))*
                             (armpoints[:,1]<np.max(hand_edges[:,1]))]
        return handpoints
    except IndexError:
        return np.array([[]])

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

def main():
    '''Main Caller Function'''
    if not os.path.exists('arm_example.png'):
        urllib.urlretrieve("https://www.packtpub.com/\
                           sites/default/files/Article-Images/B04521_02_04.png",
                           "arm_example.png")
    #binarm3d = cv2.imread('random.png')
    binarm3d= cv2.imread('arm_example.png')
    binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
        binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
    co.masks.calib_edges = np.pad(np.zeros((binarm3d.shape[
        0] - 2, binarm3d.shape[1] - 2), np.uint8), ((1, 1), (1, 1)), 'constant', constant_values=1)
    co.meas.find_non_convex_edges_lims(co.masks.calib_edges)
    print\
    timeit.timeit(lambda:main_process(binarm3d.copy(),binarm3d_positions,0),number=100)/100
    profile.runctx('main_process(binarm3d,binarm3d_positions,0)',globals(),locals())
    #main_process(binarm3d.copy(), binarm3d_positions)
    
    for c in range(4):
        rows, cols, _ = binarm3d.shape
        rot_mat = cv2.getRotationMatrix2D(
            (np.floor(cols / 2.0), np.floor(rows / 2.0)), 90, 1)
        rot_mat[0, 2] += np.floor(rows / 2.0 - cols / 2.0)
        rot_mat[1, 2] += np.floor(cols / 2.0 - rows / 2.0)
        binarm3d_positions = np.transpose(np.nonzero(np.ones_like(
            binarm3d[:, :, 0]))).reshape(binarm3d.shape[:-1] + (2,))
        main_process(binarm3d.copy(), binarm3d_positions,1)
        binarm3d = 255 * \
            ((cv2.warpAffine(binarm3d, rot_mat, (rows, cols))) > 0).astype(np.uint8)
        co.masks.calib_edges = np.pad(np.zeros((binarm3d.shape[
            0] - 2, binarm3d.shape[1] - 2), np.uint8), ((1, 1), (1, 1)), 'constant', constant_values=1)
        co.meas.find_non_convex_edges_lims(co.masks.calib_edges)
    
if __name__ == '__main__':
    main()
