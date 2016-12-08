import numpy as np
import math
import cv2
import itertools as it
import sys
import time
import helping_functs as hf
import class_objects as co


def detect_corners():
    '''function to detects  intersection limits of mask with calib_edges'''
    calib_set = set([tuple(i) for i in np.transpose(
        np.fliplr(np.nonzero(co.masks.calib_edges))).tolist()])
    contour_tuple = [tuple(i) for i in co.contours.arm_contour.tolist()]
    contour_dict = dict((k, i) for i, k in enumerate(contour_tuple))
    inter = set(contour_tuple).intersection(calib_set)
    co.contours.edges_inds = [contour_dict[x] for x in inter]
    co.contours.edges = [co.contours.arm_contour.tolist()[ind]
                         for ind in co.contours.edges_inds]
    if co.contours.edges:
        x_coord, y_coord, width, height = cv2.boundingRect(
            np.swapaxes(np.array([co.contours.edges]), 0, 1))
        contour_corners = np.reshape(np.array(co.contours.edges)
                                     [np.array(np.tile(np.array
                                                       (np.transpose(np.matrix(np.any(
                                                           ((co.contours.edges ==
                                                             np.array(x_coord)) +
                                                            (co.contours.edges == np.array(y_coord)) +
                                                            (co.contours.edges ==
                                                             np.array(x_coord + width)) +
                                                            (co.contours.edges ==
                                                             np.array(y_coord +
                                                                      height))) > 0, axis=1)))),
                                                       (1, 2)))], (-1, 2))
        contour_corners = cv2.convexHull(np.swapaxes(
            np.array([contour_corners]), 0, 1)).squeeze()
        if contour_corners.shape[0] != 2 or len(contour_corners.shape) != 2:
            return 'Object wrongly identified (too many entry co.points found)', []
        corn_ind = []
        for corner in contour_corners:
            corn_ind += [i for (j, i) in enumerate(co.contours.edges_inds)
                         if np.all(co.contours.edges[j] == corner)]
        if not corn_ind:
            return ("Warning:Detected object not touching image edges(Probably" +
                    " misidentification)", [])
        return contour_corners, sorted(corn_ind)
    else:
        return "Warning:Detected object not touching image edges(Probably misidentification)", []


def interpolate(points, winsize):
    '''interpolate co.points of a contour using a window of winsize'''
    interpolated_data = []
    for i in range(0, points.shape[0], (winsize - 1) / 2):
        interpolated_data.append(np.mean(points[max(
            0, i - (winsize - 1) / 2):(min(i + (winsize - 1) / 2,
                                           points.shape[0] - 1) + 1)], axis=0))
    return np.array(interpolated_data)


def perform_initial_grouping():
    '''group adjacent vectors with same angle'''
    tmp = [list(same_angle_vecs) for _, same_angle_vecs in
           it.groupby(co.meas.interpolated_contour_angles)]
    co.meas.segment_angle = [i[0] for i in tmp]
    co.meas.segment_points_num = [len(i) for i in tmp]
    # co.meas.segment_angle,meas.segment_points_num=[list(el) for el in
    # zip(*[(list(same_angle_vecs)[0],len(list(same_angle_vecs)))


def compute_contour_indices(winsize):
    '''find starting and ending contour indices of the grouped segments'''
    co.interpolated.vecs_starting_ind = (np.maximum(0,
                                                    (np.cumsum(np.array(co.meas.segment_points_num))
                                                     - np.array(co.meas.segment_points_num)
                                                     / 2 - 1)
                                                    * (winsize - 1) / 2 -
                                                    (np.array(co.meas.segment_points_num)
                                                        + 1) * (winsize - 1) / 4)).tolist()
    co.interpolated.vecs_ending_ind = (np.minimum(co.contours.hand_centered_contour.shape[0] - 1,
                                                  (np.cumsum(np.array(co.meas.segment_points_num))
                                                   +
                                                   np.array(
                                                      co.meas.segment_points_num)
                                                   / 2 - 1) *
                                                  (winsize - 1) / 2 +
                                                  (np.array(co.meas.segment_points_num)
                                                   + 1) *
                                                  (winsize - 1) / 4)).tolist()

def find_largest_line_segments():
    '''Find largest line segments with some angle tolerance specified by
    cutoff_angle_ratio and after that join them'''
    perform_initial_grouping()
    compute_contour_indices(co.CONST['interp_window'])
    angle_thres = math.pi / co.CONST['cutoff_angle_ratio']
    line_window_size = 1
    segment = []
    # The co.interpolated co.contours.arm_contour is transversed multiple times, in
    # order to get every possible applicant, and then the result is filtered,
    # so that largest applicants to be selected, which do not intersect with
    # each other
    candidate_segments = []
    for ind in range(0, len(co.meas.segment_points_num), line_window_size):
        forward_count = 1
        backward_count = 1
        keep_counting_forward = 1
        keep_counting_backward = 1
        freq = co.meas.segment_points_num[ind]
        goon_flag1 = 1
        goon_flag2 = 1
        candidate_segment = [0, 0]
        while goon_flag1 or goon_flag2:
            if keep_counting_forward and ind + forward_count < len(co.meas.segment_points_num):
                if abs(co.meas.segment_angle[ind] -
                       co.meas.segment_angle[ind + forward_count]) < angle_thres:
                    co.meas.segment_angle[ind] = (freq * co.meas.segment_angle[ind] +
                                                  co.meas.segment_angle[ind + forward_count]) / (freq + 1)
                    freq += co.meas.segment_points_num[ind + forward_count]
                else:
                    keep_counting_forward = 0
                forward_count += 1
            else:
                goon_flag1 = 0
                keep_counting_forward = 0
                candidate_segment[1] = co.interpolated.vecs_ending_ind[
                    ind + forward_count - 1]

            if keep_counting_backward and ind - backward_count > -1:
                if abs(co.meas.segment_angle[ind]
                       - co.meas.segment_angle[ind - backward_count]) < angle_thres:
                    co.meas.segment_angle[ind] = (freq * co.meas.segment_angle[ind] +
                                                  co.meas.segment_angle[ind
                                                                        - backward_count]) / (freq + 1)
                    freq += co.meas.segment_points_num[ind - backward_count]
                else:
                    keep_counting_backward = 0
                backward_count += 1
            else:
                goon_flag2 = 0
                keep_counting_backward = 0
                candidate_segment[0] = co.interpolated.vecs_starting_ind[
                    ind - (backward_count - 1)]
        candidate_segment += [co.meas.segment_angle[ind], freq]
        candidate_segments.append(candidate_segment)
    candidate_segments_im = 255 * np.ones((co.meas.imy, co.meas.imx))
    for segment in candidate_segments:
        cv2.line(candidate_segments_im, tuple(co.contours.hand_centered_contour[
            segment[0]]), tuple(co.contours.hand_centered_contour[segment[1]]), 0)
    co.im_results.images.append(candidate_segments_im)
    sorted_segments = sorted(
        candidate_segments, key=lambda segment: segment[3], reverse=True)
    final_segments = [sorted_segments[0]]
    held_inds = range(final_segments[0][0], final_segments[0][1] + 1)
    for segment in list(sorted_segments):
        if segment[0] not in held_inds and segment[1] not in held_inds:
            held_inds += range(segment[0], segment[1] + 1)
            final_segments.append(segment)
    co.contours.final_segments = sorted(
        final_segments, key=lambda segment: segment[0])


def detect_wrist():
    '''Detect wrist points'''
    # Find hand entry co.points into image (corners)
    contour_corners, corn_ind = detect_corners()

    if not isinstance(contour_corners, str):
        hand_centered_contour_ind = np.roll(
            np.array(range(co.contours.arm_contour.shape[0])), -corn_ind[0] - 1)
        co.contours.hand_centered_contour = co.contours.arm_contour[
            hand_centered_contour_ind]
        corn_ind[0] = co.contours.hand_centered_contour.shape[0] - 1
    else:
        return contour_corners, []  # returns warning
    co.contours.hand_centered_contour = co.contours.hand_centered_contour[
        corn_ind[1]:corn_ind[0] + 1]
    # Interpolate contour so as to get polygonal approximation
    winsize = co.CONST['interp_window']
    co.interpolated.points = interpolate(
        co.contours.hand_centered_contour, winsize)
    if co.interpolated.points.shape[0] <= 1:
        return "Object too small (misidentification)", []
    co.meas.interpolated_contour_angles = hf.compute_angles(
        co.interpolated.points)
    if len(co.meas.interpolated_contour_angles) == 0:
        print co.interpolated.points
        print "check helping_functs for errors in compute_angles"
    find_largest_line_segments()

    # Print found segments result.
    interpolated_contour_im = 255 * np.ones((co.meas.imy, co.meas.imx))
    # print 'final_segments found:',co.contours.final_segments
    for segment in co.contours.final_segments:
        cv2.line(interpolated_contour_im, tuple(co.contours.hand_centered_contour[
            segment[0]]), tuple(co.contours.hand_centered_contour[segment[1]]), 0, 1, 0)
    co.im_results.images.append(interpolated_contour_im)

    # The main part of the algorithm follows. A measure is constructed based
    # on lambda, length ratio and max length, so as to find the best segments
    # describing the forearm. Lambda is a parameter describing if two segments
    # can construct effectively a quadrilateral

    co.meas.lam = []
    co.meas.len = []
    total_meas = []
    approved_segments = []
    wearing_par1 = 1
    lengths = []
    wearing_rate = co.CONST['wearing_dist_rate']
    lam_weight = co.CONST['lambda_power_weight']
    len_rat_weight = co.CONST['length_ratio_power_weight']
    max_length_weight = co.CONST['length_power_weight']
    check_segments_num = co.CONST['num_of_checked_segments']
    for count1, segment1 in enumerate(co.contours.final_segments[0:check_segments_num - 1]):
        wearing_par2 = 1
        st1_ind, en1_ind, _, _ = segment1

        if len(co.contours.final_segments) == 1:
            break
        st1 = co.contours.hand_centered_contour[st1_ind]
        en1 = co.contours.hand_centered_contour[en1_ind]
        length1 = np.linalg.norm(en1 - st1)
        mid1 = (st1 + en1) / 2
        count2 = len(co.contours.final_segments) - 1
        if np.all(en1 == st1):
            continue
        for segment2 in co.contours.final_segments[:max(len(co.contours.final_segments) -
                                                        check_segments_num, count1 + 1):-1]:
            st2_ind, en2_ind, _, _ = segment2

            st2 = co.contours.hand_centered_contour[st2_ind]
            en2 = co.contours.hand_centered_contour[en2_ind]
            length2 = np.linalg.norm(en2 - st2)
            mid2 = (st2 + en2) / 2
            if np.all(en2 == st2):
                continue
            lambda1 = np.dot((st1 - en1), (st1 - mid2)) / \
                (float(np.linalg.norm((en1 - st1)))**2)
            lambda2 = np.dot((st2 - en2), (st2 - mid1)) / \
                (float(np.linalg.norm((en2 - st2)))**2)
            lower_b = -0.3
            upper_b = 1.3
            middle_b = (lower_b + upper_b) / 2
            if (lambda1 < 1.3) and (lambda1 > -0.3):
                if (lambda2 < 1.3) and (lambda2 > -0.3):
                    co.meas.lam.append(
                        abs(1 / np.sqrt((middle_b - lambda1)**2 + (middle_b - lambda2)**2)))
                    co.meas.len.append(
                        min(length1 / float(length2), length2 / float(length1)))
                    total_meas.append(co.meas.lam[-1]**lam_weight *
                                      co.meas.len[-1]**len_rat_weight *
                                      wearing_par1 * wearing_par2 *
                                      max(length1, length2)**max_length_weight)
                    approved_segments.append([segment1, segment2])
                    lengths.append(max(length1, length2))
            # Uncomment to view procedure. You have to add found flag above
            # for this to work
            '''line_segments_im=np.ones((co.meas.imy, meas.imx))
            for st_ind, en_ind, _, _ in line_segments:
             cv2.line(line_segments_im, tuple(co.contours.hand_centered_contour[st_ind]), tuple(contours.hand_centered_contour[en_ind]), 0, 2)
             cv2.imshow('line_segments_im', line_segments_im)
            im_tmp_result=np.empty((co.meas.imy, meas.imx)+(3, ), dtype=float)
            im_tmp_result[:, :, 0]=line_segments_im
            im_tmp_result[:, :, 1]=line_segments_im
            im_tmp_result[:, :, 2]=line_segments_im
            cv2.line(im_tmp_result, tuple(st1), tuple(en1), [0, 0, 1], 2)
            if found==1:
             cv2.line(im_tmp_result, tuple(st2), tuple(en2), [0, 1, 0], 2)
            else:
             cv2.line(im_tmp_result, tuple(st2), tuple(en2), [1, 0, 0], 2)
            cv2.imshow('im_tmp_result', im_tmp_result)
            cv2.waitKey(1000/co.CONST['framerate'])
            '''
            wearing_par2 += -wearing_rate
            count2 = count2 - 1
        wearing_par1 += -wearing_rate
    total_meas = [m / max(lengths) for m in total_meas]
    if not approved_segments:
        return "Warning: Wrist not found", []

    segment1, segment2 = approved_segments[total_meas.index(max(total_meas))]
    #lam = co.meas.lam[total_meas.index(max(total_meas))]
    #leng = co.meas.len[total_meas.index(max(total_meas))]
    st1_ind, en1_ind, _, _ = segment1
    st2_ind, en2_ind, _, _ = segment2
    st1 = co.contours.hand_centered_contour[st1_ind]
    en1 = co.contours.hand_centered_contour[en1_ind]
    st2 = co.contours.hand_centered_contour[st2_ind]
    en2 = co.contours.hand_centered_contour[en2_ind]
    # The best segment set is selected. Then ,it is known that the found
    # segments have opposite directions. We make a first estimate of the wrist
    # co.points by calculating the points of segments that are farthest from the
    # corners.

    co.points.wristpoints = np.zeros((2, 2), np.int64)
    dist_st = np.linalg.norm(st1 - contour_corners[0])
    dist_en = np.linalg.norm(en1 - contour_corners[0])
    if dist_st > dist_en:
        co.points.wristpoints[0] = st1
        co.points.wristpoints[1] = en2
    else:
        co.points.wristpoints[0] = en1
        co.points.wristpoints[1] = st2
    # The next estimation is made by supposing that that the wrist point
    # closer to image edge belongs to wrist
    wristpoint_ind = [np.asscalar(np.where(
        np.all(co.contours.hand_centered_contour ==
               point, axis=1))[0][0]) for point in co.points.wristpoints]
    hand_shape = np.zeros((co.meas.imy, co.meas.imx))
    cv2.drawContours(hand_shape, [np.swapaxes(np.array([co.contours.hand_centered_contour[
        min(wristpoint_ind):max(wristpoint_ind) + 1]]), 0, 1)], 0, 255, 1)
    #nzeroelems = np.transpose(np.array(np.nonzero(hand_shape)))
    #(starting from wrist point closer to image edge, find better second wrist point)
    wristpoints_dist_from_corn = [[np.linalg.norm(co.points.wristpoints[0] - corner)
                                   for corner in contour_corners],
                                  [np.linalg.norm(co.points.wristpoints[1] - corner)
                                   for corner in contour_corners]]
    wristpoint1 = co.points.wristpoints[
        np.argmin(np.array(wristpoints_dist_from_corn)) / 2]
    # To find the other wrist point, two bounds are identified, within which
    # it must be, those are the opposite corner from the first wrist point and
    # the 2nd estimated wrist point found above
    if np.all(wristpoint1 == co.points.wristpoints[0]):
        w_bound1 = wristpoint_ind[1]
        wristpoint_ind1 = wristpoint_ind[0]
    else:
        w_bound1 = wristpoint_ind[0]
        wristpoint_ind1 = wristpoint_ind[1]
    w_bound2 = -1
    for ind in corn_ind:
        if not (wristpoint_ind1 > min(w_bound1, ind) and wristpoint_ind1 < max(w_bound1, ind)):
            w_bound2 = ind
            break
    if w_bound2 == -1:
        return "no point found between the two bounds", []
    # Second wrist point is found to be the one that belongs between the
    # bounds and is least far from the first point
    wristpoint_ind2 = min(w_bound1,
                          w_bound2) + np.argmin(
                              np.array(
                                  [np.linalg.norm(
                                      point - wristpoint1)
                                   for point in
                                   co.contours.hand_centered_contour[
                                       min(w_bound1, w_bound2):max(w_bound1, w_bound2)]]))
    new_wristpoint = [wristpoint1,
                      co.contours.hand_centered_contour[wristpoint_ind2]]
    new_wristpoint_ind = [np.asscalar(np.where(np.all(
        co.contours.hand_centered_contour == point, axis=1))[0][0]) for point in new_wristpoint]
    if abs(new_wristpoint_ind[0] - new_wristpoint_ind[1]) <= 5:
        return 'too small hand region found, probably false positive', []
    hand_contour = np.swapaxes(np.array([co.contours.hand_centered_contour[min(
        new_wristpoint_ind):max(new_wristpoint_ind) + 1]]), 0, 1)
    return new_wristpoint, hand_contour
