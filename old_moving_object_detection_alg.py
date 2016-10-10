'''Functions for identifying moving object in almost static scene'''
import time
import numpy as np
import cv2
from scipy import signal
import class_objects as co

def scene_segmentation(constants):
    for c in range(7):    
        if isinstance(co.segclass.segment_values, str):
            co.segclass.segment_values = (co.data.depth_im*255).astype(np.uint8)
            co.segclass.segment_enum=np.zeros_like(co.data.depth_im)
            flag=1
        else:
            flag=0
       
        val=0.2
        thres =np.maximum(np.ceil(val * co.segclass.segment_values),5)
        kernel=np.ones((co.meas.erode_size,co.meas.erode_size),dtype=np.uint8)
        eroded_segment_values=cv2.erode(co.segclass.segment_values, kernel)
        enable_erosion=(co.segclass.segment_values-eroded_segment_values)<thres
        co.segclass.segment_values[enable_erosion]=eroded_segment_values[enable_erosion]
        if flag==1:
            co.segclass.segment_enum[enable_erosion]=np.arange(np.sum(enable_erosion))
        else:
            dilated_segment_enum=cv2.dilate(co.segclass.segment_enum,kernel)
            new_enum=dilated_segment_enum * co.segclass.segment_enum>0
            co.segclass.segment_enum[new_enum]=dilated_segment_enum[new_enum]
    if co.counters.im_number % constants['framerate']==0 and co.counters.im_number!=0:
        co.meas.erode_size-=2
            
def scene_segmentation_old2():
    '''Segment background'''
    if isinstance(co.segclass.segment_values, str):
        co.segclass.segment_values = (co.data.depth_im*255).astype(np.uint8)
    print np.max(co.segclass.segment_values)
    val=0.2
    thres =np.maximum(np.ceil(val * co.segclass.segment_values),5)
    time1 = time.clock()
    co.segclass.array_instances = np.zeros(
        (9, co.meas.imy + 2, co.meas.imx + 2),dtype=np.uint8)
    eroded_segment_values=co.segclass.segment_values.copy()
    co.segclass.array_instances[0, :-2, :-
                                    2] = eroded_segment_values
    co.segclass.array_instances[1, :-2, 1:-1] =\
    eroded_segment_values
    co.segclass.array_instances[2, :-2, 2:] =\
            eroded_segment_values
    co.segclass.array_instances[3, 1:-1, :-2] =\
            eroded_segment_values
    co.segclass.array_instances[4, 1:-1, 2:] =\
            eroded_segment_values
    co.segclass.array_instances[5, 2:, :-2] =\
            eroded_segment_values
    co.segclass.array_instances[6, 2:, 1:-1] =\
            eroded_segment_values
    co.segclass.array_instances[7, 2:, 2:] = \
            eroded_segment_values
    co.segclass.array_instances[8, 1:-1, 1:-1] = \
            eroded_segment_values
    time2 = time.clock()
    differences = np.abs(co.segclass.array_instances -
                         co.segclass.array_instances[8, :, :])
    differences[8, 1:-1, 1:-1] = thres
    differences[differences == 0] = 255
    time3 = time.clock()
    minargs = np.argmin(differences, axis=0).ravel()
    from numpy import histogram
    print histogram (minargs, bins=minargs.max()-1)
    time4 = time.clock()
    tmp_segment_values =\
    np.take(co.segclass.array_instances.ravel(),minargs*(co.meas.imx+2)*(co.meas.imy+2)+co.meas.nprange).reshape(co.meas.imy + 2, co.meas.imx + 2)
    time5 = time.clock() 
    co.segclass.segment_values = tmp_segment_values[1:-1, 1:-1]
    
    
    #cv2.imshow('segments', co.segclass.segment_values)
    #cv2.waitKey(100)
    '''print "Loop:", co.meas.im_count
    print time2 - time1, 's'
    print time3 - time2, 's'
    print time4 - time3, 's'
    print time5 - time4, 's'
    '''
    #cv2.imshow('depth_im',co.data.depth_im)
    #cv2.imshow('segments',co.segclass.segment_values)
    #cv2.waitKey(100)
def scene_segmentation_old(constants):
    '''Segment background'''
    if isinstance(co.segclass.segment_values, str):
        co.segclass.array_instances = np.zeros((9, co.meas.imy + 2, co.meas.imx + 2))
    #    co.segclass.segment_values[co.segclass.segment_values==0]=1
    #   co.segclass.start_segmentation(10, co.meas)
    #time1 = time.clock()
    # segment_values is the segmented image
    # old_segment_values is used for the iterations
    #old_segment_values = co.segclass.segment_values
    # firstly segment_values is dilated
    #dilated_segment_values = cv2.dilate(
    #    co.segclass.segment_values, co.segclass.kernel)
    #segment_values_im = co.data.depth_im * (dilated_segment_values > 0)
    #part = np.zeros((co.meas.imy + 2, co.meas.imx + 2))
    val = 0.15
    #thres = 1 / float(255) * \
    #    np.ceil(val * 255 * segment_values_im *
    #            (co.segclass.segment_values > 0))
    thres = 1/ float(255) * np.ceil(val *255 *co.data.depth_im)
    #thres[thres==0]=1
    #time2 = time.clock()
    # Proposed algorithm for an evolution kernel of size 3x3 and generative number
    # of seeds:
    #    keep only the difference between the depth_im*(segment_values>0) and
    #    depth_im*(dilated_segment_values>0)
    #    a.create 8 instances of depth_im,zero padded in every possible manner
    #    b.substract all of them from the fully zero padded original one and keep
    #      8 different values for each pixel. If the substraction is by zero,
    #      then leave the corresponding value out.
    #    c.find the min and index of the minimum value for each pixel
    #    d.if the min is smaller than a set threshold then goto (e), else leave
    #      the pixel value as it is
    #    e.the new value for the pixel is the corresponding value belonging to
    #      index-th instance
    time1 = time.clock()
    co.segclass.array_instances[0, :-2, :-2] = co.segclass.segment_values
    co.segclass.array_instances[1, :-2, 1:-1] = co.segclass.segment_values
    co.segclass.array_instances[2, :-2, 2:] = co.segclass.segment_values
    co.segclass.array_instances[3, 1:-1, :-2] = co.segclass.segment_values
    co.segclass.array_instances[4, 1:-1, 2:] = co.segclass.segment_values
    co.segclass.array_instances[5, 2:, :-2] = co.segclass.segment_values
    co.segclass.array_instances[6, 2:, 1:-1] = co.segclass.segment_values
    co.segclass.array_instances[7, 2:, 2:] = co.segclass.segment_values
    co.segclass.array_instances[8, 1:-1, 1:-1] = co.segclass.segment_values
    time2 = time.clock()
    differences = np.abs(co.segclass.array_instances - \
        np.tile(co.segclass.array_instances[8, :, :], (9, 1, 1)))
    differences[8, 1:-1, 1:-1] = thres
    differences[differences==0]=1
    time3 = time.clock
    minargs = np.argmin(differences, axis=0).ravel().tolist()
    time4 = time.clock
    tmp_segment_values=np.zeros((co.meas.imy+2,co.meas.imx+2))
    count=0
    for count1 in range(co.meas.imy+2):
        for count2 in range(co.meas.imx+2):
            tmp_segment_values[count1,count2]=co.segclass.array_instances[minargs[count],count1,count2]
            count+=1
    time5 = time.clock
    co.segclass.segment_values=tmp_segment_values[1:-1, 1:-1]

   # Following algorithm has an evolution kernel of size 3x3 and known number
   # of seeds
    '''
    for shift_count in range(8):
        y_shift, x_shift = co.segclass.shift_ind[shift_count, :]
        part[(-y_shift + 1):(-y_shift + (co.meas.imy + 1)),
             (-x_shift + 1):(-x_shift + (co.meas.imx + 1))
            ] += np.logical_and(abs(signal.convolve2d(co.data.depth_im, co.segclass.mask[
                shift_count, :, :], mode='same')) <= thres, thres != 0)
        part = (part > 0).astype(int)
    time3 = time.clock()
    print np.sum(part)
    co.segclass.segment_values = np.maximum(
        old_segment_values, dilated_segment_values * part[1:(co.meas.imy + 1), 1:(co.meas.imx + 1)])
    values_to_change_index = np.logical_and(np.logical_and((co.segclass.segment_values > 0), (
        old_segment_values > 0)), (co.segclass.segment_values !=
                                   old_segment_values))
    time4 = time.clock()
    values_to_change = np.unique(old_segment_values[values_to_change_index])
    for value in values_to_change:
        timee1 = time.clock()
        co.segclass.segment_values[co.segclass.segment_values == value] = value
        timee2 = time.clock()
        print timee2 - timee1
    time5 = time.clock()
    '''
    print "Loop:", len(co.meas.im_count)
    print time2 - time1, 's'
    print time3 - time2, 's'
    print time4 - time3, 's'
    print time5 - time4, 's'

    '''cv2.imwrite('segments.jpg',co.segclass.segment_values -
                                np.min(co.segclass.segment_values)) /
                float(np.max(co.segclass.segment_values) -
                     np.min(co.segclass.segment_values)))
    '''
    #cv2.imshow('segments', (co.segclass.segment_values - np.min(co.segclass.segment_values)) /
    #           float(np.max(co.segclass.segment_values) - np.min(co.segclass.segment_values)))
    #cv2.waitKey(100)


def init_noise_model(constants):
    '''Compute Initial Background and Noise Estimate from the first im_num
     images with no moving object presence'''

    # Assume Noise dependency from location
    imy, imx, im_num = co.data.initial_im_set.shape
    valid_freq = np.zeros(co.data.initial_im_set[:, :, 0].shape)
    co.data.background = np.zeros(co.data.initial_im_set[:, :, 0].shape)
    initial_zero_im_set = np.zeros(co.data.initial_im_set.shape)
    for count in range(im_num):
        co.data.background = co.data.background + \
            co.data.initial_im_set[:, :, count]
        valid_freq[co.data.initial_im_set[:, :, count] != 0] += 1
        initial_zero_im_set[:, :, count] += co.data.initial_im_set[
            :, :, count] != 0
    valid_freq[valid_freq == 0] = 1
    co.data.background = co.data.background / valid_freq
    cv2.imwrite('Background.jpg', (255 * co.data.background).astype(int))
    cv2.imwrite('Validfreq.jpg', (255 * valid_freq /
                                  float(np.max(valid_freq))).astype(int))
    noise_deviation = np.zeros((imy, imx))
    for c_im in range(im_num):
        # Remove white pixels from co.data.initial_im_set
        noise_deviation0 = ((co.data.background -
                             co.data.initial_im_set[:, :, c_im]) *
                            initial_zero_im_set[:, :, c_im])
        noise_deviation0 = noise_deviation0 * noise_deviation0
        noise_deviation = noise_deviation + noise_deviation0
    noise_deviation = np.sqrt(noise_deviation / valid_freq)
    cv2.imwrite('noise_deviation.jpg', (255 * noise_deviation /
                                        np.max(noise_deviation)).astype(int))

    noise_deviation = np.minimum(noise_deviation, 0.2)
    noise_deviation = np.maximum(
        noise_deviation, constants['lowest_pixel_noise_deviation'])
    max_background_thres = co.data.background + noise_deviation
    cv2.imwrite('Max_Background.jpg', (255 * max_background_thres).astype(int))
    cv2.imwrite('Zero_Background.jpg', (255 *
                                        (co.data.background == 0)).astype(int))
    min_background_thres = (co.data.background - noise_deviation) * \
        ((co.data.background - noise_deviation) > 0)
    cv2.imwrite('Min_Background.jpg', (255 * min_background_thres).astype(int))
    print "Recognition starts now"
    return min_background_thres, max_background_thres


def find_moving_object(constants):
    '''Find the biggest moving object in scene'''
    found_object0 = np.ones(co.data.depth_im.shape)
    min_background_thres, max_background_thres = co.models.noise_model
    co.masks.background = np.logical_and(np.logical_or((co.data.depth_im > max_background_thres), (
        co.data.depth_im < min_background_thres)), co.data.depth_im != 0)

    co.im_results.images += [co.masks.background, co.data.depth_im]
    found_object0[co.masks.background] = co.data.depth_im[co.masks.background]
    laplacian = cv2.Laplacian(found_object0, cv2.CV_64F)
    found_object0[np.abs(laplacian) > co.thres.lap_thres] = 1.0
    # Remove also 0 values
    found_object0[found_object0 == 0.0] = 1.0
    # Isolate considered Object by choosing a contour from the ones found
    thresh = np.zeros(found_object0.shape, dtype=np.uint8)
    thresh[found_object0 < 1] = 255
    co.meas.contours_areas = []
    _, co.scene_contours, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not co.scene_contours:
        return "No Contours Found", [], [], []
    for i, count1 in enumerate(co.scene_contours):
        area = cv2.contourArea(count1)
        co.meas.contours_areas.append(area)
    # Sort co.scene_contours by area
    sorted_contours = sorted(zip(co.meas.contours_areas, co.scene_contours),
                             key=lambda x: x[0], reverse=True)

    sorted_contours = [x for (_, x) in sorted_contours]
    mask = np.zeros(co.data.depth_im.shape, np.uint8)
    sorted_depth = np.zeros(constants['max_contours_num'])
    for i in range(min(len(co.scene_contours), constants['max_contours_num'] - 1)):
        # Find mean depth for each shape matching each contour
        mask[:] = 0
        cv2.drawContours(mask, sorted_contours, 0, 255, -1)
        sorted_depth[i] = np.mean(co.data.depth_im[mask > 0])
    if (co.meas.aver_depth == []) | (co.meas.aver_depth == 0):
        co.meas.aver_depth = sorted_depth[0]
    match = (sorted_depth < (co.meas.aver_depth + co.thres.depth_thres)
             ) & (sorted_depth > (co.meas.aver_depth - co.thres.depth_thres))
    if np.sum(match[:]) == 0:
        chosen_contour = sorted_contours[0]
        chosen_depth = sorted_depth[0]
        matched = 0
    else:
        found_match = np.nonzero(match)
        matched = found_match[0][0]
        chosen_contour = sorted_contours[matched]
        chosen_depth = sorted_depth[matched]
    if np.abs(chosen_depth - co.meas.aver_depth) < co.meas.aver_depth:
        co.counters.outlier_time = 0
        co.data.depth_mem.append(chosen_depth)
        co.counters.aver_count += 1
        if co.counters.aver_count > constants['running_mean_depth_count']:
            co.meas.aver_depth = ((chosen_depth -
                                   co.data.depth_mem[
                                       co.counters.aver_count -
                                       constants['running_mean_depth_count']] +
                                   co.meas.aver_depth *
                                   constants['running_mean_depth_count']) /
                                  constants['running_mean_depth_count'])
        else:
            co.meas.aver_depth = (chosen_depth + co.meas.aver_depth *
                                  (co.counters.aver_count - 1)) / co.counters.aver_count
    else:
        co.counters.outlier_time += 1
    if co.counters.outlier_time == constants['outlier_shape_time'] * constants['framerate']:
        co.counters.aver_count = 1
        co.meas.aver_depth = chosen_depth
        co.data.depth_mem = []
        co.data.depth_mem.append(chosen_depth)
    final_mask = np.zeros(co.data.depth_im.shape, np.uint8)
    cv2.drawContours(final_mask, sorted_contours, matched, 255, -1)
    if np.array(chosen_contour).squeeze().shape[0] <= 5:
        return "Arm not found", [], [], []
    return final_mask, chosen_contour, sorted_contours, matched
