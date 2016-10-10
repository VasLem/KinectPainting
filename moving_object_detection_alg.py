'''Functions for identifying moving object in almost static scene'''
import time
import os.path
import numpy as np
import cv2
import class_objects as co
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import cPickle as pickle




def find_partitions(obj):
    if np.size(obj) == np.size(co.data.depth_im):
        points = np.transpose(np.where(obj > 0))
    else:
        points = obj
    center = np.mean(points, axis=0)
    obj1 = np.array([i for i in points if i[0] <= center[0] and i[1] <=
                     center[1]])
    obj2 = np.array([i for i in points if i[0] <= center[0] and i[1] > center[1]])
    obj3 = np.array([i for i in points if i[0] > center[0] and i[1] <= center[1]])
    obj4 = np.array([i for i in points if i[0] > center[0] and i[1] > center[1]])
    return [obj1, obj2, obj3, obj4]


def object_partition(obj, constants):
    objs = find_partitions(obj)
    for obj in objs:
        if obj.size==2:
            return
        if obj.shape[0] > np.size(co.data.depth_im) * constants['max_objects_size_ratio']:
            object_partition(obj, constants)
        else:
            mask = np.zeros_like(co.data.depth_im)
            mask[tuple(np.transpose(obj))] = 1
            co.segclass.objects.mask.append(mask)
            co.segclass.objects.size.append(obj.shape[0])
            co.segclass.objects.obj_count += 1
            co.segclass.all_objects_im+=mask*co.segclass.objects.obj_count

def detection_by_scene_segmentation(constants):
    '''Detection of moving object by using Distortion field of centers of mass
    of background objects'''
    if co.counters.im_number == 0:
        co.segclass.exists_previous_segmentation = os.path.isfile(
            constants['already_segmented_path'])
        if co.segclass.exists_previous_segmentation:
            print 'Existing previous segmentation. Checking similarity...'
        else:
            print 'No existing segmentation. The initialisation will delay...'
    if co.segclass.exists_previous_segmentation and co.counters.im_number < 2:
        if co.counters.im_number == 0:
            old_im = cv2.imread(constants['already_segmented_path'], -1)
            co.segclass.check_if_segmented_1 = np.sqrt(np.sum(
                (co.data.depth_im - old_im)**2))
            co.segclass.prev_im = co.data.depth_im.copy()
        elif co.counters.im_number == 1:
            co.segclass.check_if_segmented_2 = np.sqrt(
                np.sum((co.data.depth_im - co.segclass.prev_im)**2))
            difference = abs(co.segclass.check_if_segmented_2 -
                             co.segclass.check_if_segmented_1)
            print 'Metric between old segmented and new background :', difference
            co.segclass.needs_segmentation = difference >= 100
            if co.segclass.needs_segmentation:
                print 'Segmentation is needed.The initialisation will delay...'
            else:
                print 'Found match with previous segmentation...'
    elif not co.segclass.exists_previous_segmentation\
            and co.counters.im_number < 2:
        co.segclass.needs_segmentation = 1
    if co.segclass.needs_segmentation and co.counters.im_number >= 2:
        if co.counters.im_number <\
                (constants['framerate'] * constants['calib_secs'] - 1):
            for count in range(constants['cond_erosions_num']):
                if isinstance(co.segclass.segment_values, str):
                    co.segclass.segment_values = (
                        256 * co.data.depth_im).astype(np.uint8)
                    co.segclass.segment_enum =\
                        np.zeros_like(co.data.depth_im, dtype=float)
                    flag = 1
                else:
                    flag = 0

                val = 0.04
                thres = np.ceil(val * co.segclass.segment_values)
                kernel = np.ones(
                    (co.meas.erode_size, co.meas.erode_size), dtype=np.uint8)
                eroded_segment_values = cv2.erode(
                    co.segclass.segment_values, kernel)
                enable_erosion = (co.segclass.segment_values -
                                  eroded_segment_values) < thres
                co.segclass.segment_values[
                    enable_erosion] = eroded_segment_values[enable_erosion]
            if (co.counters.im_number + 1) % constants['framerate'] == 0:
                co.meas.erode_size -= 2
        elif co.counters.im_number == (constants['framerate'] *
                                       constants['calib_secs'] - 1):
            co.segclass.objects.obj_count = 0
            for val in np.unique(co.segclass.segment_values):
                if val != 0:
                    objs = np.ones_like(co.data.depth_im) * \
                        (val == co.segclass.segment_values)
                    labeled, nr_objects = ndimage.label(objs)
                    lbls = np.arange(1, nr_objects + 1)
                    obj_size = ndimage.labeled_comprehension(objs, labeled, lbls,
                                                             len, float, 0)
                    for label in lbls[obj_size > 1000]:

                        obj = np.ones_like(co.data.depth_im) * \
                            (labeled == label)
                        if (np.sum(obj > 0) > np.size(co.data.depth_im)
                                * constants['max_objects_size_ratio']):
                            object_partition(obj, constants)
                        else:
                            co.segclass.objects.mask.append(obj)
                            co.segclass.objects.size.append(obj_size)
                            co.segclass.objects.obj_count += 1
                            co.segclass.all_objects_im+=obj*co.segclass.objects.obj_count

            co.segclass.objects.find_object_center()
            print 'Found or partitioned', co.segclass.objects.obj_count, 'background objects'
            with open(constants['segmentation_data'] + '.pkl', 'wb') as output:
                pickle.dump(co.segclass.objects, output, -1)
            cv2.imwrite(constants['already_segmented_path'], co.data.depth_im)
            cv2.imwrite('segments.jpg', co.segclass.segment_values)
            plt.imshow(co.segclass.all_objects_im)
            plt.savefig('partitioned_segments.jpg')
            print 'Saved segmentation data for future use.'
            co.segclass.needs_segmentation = 0
    elif (not co.segclass.needs_segmentation) and co.counters.im_number >= 2:
        if co.segclass.objects.initial_center == []:
            co.segclass.objects = pickle.load(
                open(constants['segmentation_data'] + '.pkl', 'rb'))
        co.segclass.objects.find_object_center()
        if co.segclass.objects.center:
            co.segclass.objects.find_centers_displacement()
            intersecting_points, moving_point = co.segclass.objects.find_intersecting_points()
            if isinstance(intersecting_points, str):
                print intersecting_points
                return 0
            points_on_im = co.data.depth3d.copy()
            for point in intersecting_points:
                
                cv2.circle(points_on_im, (point[1],point[0]), 2, [1,0,0],-1)

            for point in co.segclass.objects.initial_center:
               cv2.circle(points_on_im, (point[1],point[0]), 2, [0,1,0],-1)
            print 'number of found points allegedly belonging to moving object:',\
                len(intersecting_points)
            print 'point belonging to moving object with highest probability', moving_point
            cv2.circle(points_on_im, tuple(moving_point), 2, [0, 1, 1])
            co.im_results.images.append(points_on_im)
            co.im_results.images.append(co.data.depth_im)
def extract_background_values():
    '''function to extract initial background values from initial_im_set'''
    _, _, im_num = co.data.initial_im_set.shape
    valid_freq = np.zeros(co.data.initial_im_set[:, :, 0].shape)
    co.data.background = np.zeros(co.data.initial_im_set[:, :, 0].shape)
    initial_nonzero_im_set = np.zeros(co.data.initial_im_set.shape)
    initial_nonunary_im_set = np.zeros_like(co.data.depth_im)
    for count in range(im_num):
        co.data.background = co.data.background + \
            co.data.initial_im_set[:, :, count]
        valid_freq[co.data.initial_im_set[:, :, count] != 0] += 1
        initial_nonzero_im_set[:, :, count] += co.data.initial_im_set[
            :, :, count] != 0
        initial_nonunary_im_set += co.data.initial_im_set[
            :, :, count] != 1
    valid_freq[valid_freq == 0] = 1
    co.data.trusty_pixels = ((valid_freq *
                              initial_nonunary_im_set) > 0).astype(np.uint8)
    co.data.background = co.data.background / valid_freq
    return valid_freq, initial_nonzero_im_set, im_num


def init_noise_model(constants):
    '''Compute Initial Background and Noise Estimate from the first im_num
     images with no moving object presence'''

    valid_freq, initial_nonzero_im_set, im_num = extract_background_values()
    # Assume Noise dependency from location

    cv2.imwrite('Background.jpg', (255 * co.data.background).astype(int))
    cv2.imwrite('Validfreq.jpg', (255 * valid_freq /
                                  float(np.max(valid_freq))).astype(int))
    noise_deviation = np.zeros_like(co.data.depth_im)
    for c_im in range(im_num):
        # Remove white pixels from co.data.initial_im_set
        noise_deviation0 = ((co.data.background -
                             co.data.initial_im_set[:, :, c_im]) *
                            initial_nonzero_im_set[:, :, c_im])
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