'''Functions for identifying moving object in almost static scene'''
import time
import os.path
import cPickle as pickle
import numpy as np
import cv2
import class_objects as co
from scipy import ndimage
import matplotlib.pyplot as plt




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
            old_im = cv2.imread(constants['already_segmented_path'])
            co.segclass.check_if_segmented_1 = np.sqrt(np.sum(
                (co.data.color_im.astype(float) - old_im.astype(float))**2))
            co.segclass.prev_im = co.data.color_im.copy()
        elif co.counters.im_number == 1:
            co.segclass.check_if_segmented_2 = np.sqrt(
                np.sum((co.data.color_im.astype(float) -
                        co.segclass.prev_im.astype(float))**2))
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
        if co.counters.im_number == (constants['framerate']*
                                     constants['calib_secs']-1):
            co.segclass.nz_objects.image=np.zeros_like(co.data.depth_im)-1
            co.segclass.z_objects.image=np.zeros_like(co.data.depth_im)-1
            levels_num=8
            levels=np.linspace(np.min(co.data.depth_im[co.data.depth_im>0]),np.max(co.data.depth_im),levels_num)
            co.segclass.segment_values=np.zeros_like(co.data.depth_im)
            for count in range(levels_num-1):
                co.segclass.segment_values[(co.data.depth_im>=levels[count]) *
                                           (co.data.depth_im<=levels[count+1])]=count+1
        
        elif co.counters.im_number == (constants['framerate'] *
                                       constants['calib_secs'] ):
            co.segclass.nz_objects.count = -1
            co.segclass.z_objects.count = -1
            co.segclass.segment_values=co.segclass.segment_values*co.data.trusty_pixels
            for val in np.unique(co.segclass.segment_values):
                objs = np.ones_like(co.data.depth_im) * \
                    (val == co.segclass.segment_values)
                labeled, nr_objects =\
                ndimage.label(objs*co.masks.calib_frame)
                lbls = np.arange(1, nr_objects + 1)
                if val>0:
                    ndimage.labeled_comprehension(objs, labeled, lbls,
                                                  co.segclass.nz_objects.process, float, 0,
                                                  True)
                else:
                    test=ndimage.labeled_comprehension(objs, labeled, lbls,
                                                  co.segclass.z_objects.process,
                                                  float, 0, True)
            for points,pixsize,xsize,ysize in co.segclass.nz_objects.untrusty+co.segclass.z_objects.untrusty:
                co.segclass.z_objects.count+=1
                co.segclass.z_objects.image[
                    tuple(points)]=co.segclass.z_objects.count
                co.segclass.z_objects.pixsize.append(pixsize)
                co.segclass.z_objects.xsize.append(xsize)
                co.segclass.z_objects.ysize.append(ysize)
            print 'Found or partitioned',\
                    co.segclass.nz_objects.count+\
                    co.segclass.z_objects.count+2, 'background objects'
            with open(constants['segmentation_data'] + '.pkl', 'wb') as output:
                pickle.dump(co.segclass, output, -1)
            cv2.imwrite(constants['already_segmented_path'],
                        co.segclass.prev_im)
            cv2.imwrite('segments.jpg', co.segclass.segment_values)
            plt.imshow(co.segclass.nz_objects.image)
            plt.savefig('nz_partitioned_segments.jpg')
            plt.imshow(co.segclass.z_objects.image)
            plt.savefig('z_partitioned_segments.jpg')
            print 'Saved segmentation data for future use.'
            co.segclass.needs_segmentation = 0
    elif (not co.segclass.needs_segmentation) and co.counters.im_number >= 2:
        if co.segclass.nz_objects.initial_center == []:
            print 'Loading scene objects from memory.'
            co.segclass= pickle.load(
                open(constants['segmentation_data'] + '.pkl', 'rb'))
        time1 = time.clock()
        if co.counters.im_number ==constants['framerate'
                                            ]*constants['calib_secs']+1 :
            co.data.depth_im=co.data.initial_im_set[:,:,0]
            try:
                co.segclass.nz_objects.find_object_center(1)
            except:
                exit()
            try:
                co.segclass.z_objects.find_object_center(0)


            except :
                exit()
            co.segclass.initialise_neighborhoods()
        else:
            try:
                co.segclass.nz_objects.find_object_center(1)
            except:
                exit()
        if co.segclass.nz_objects.center.size>0:
            co.segclass.nz_objects.find_centers_displacement()
            found_objects_mask=co.segclass.find_objects(constants)
            time2 = time.clock()
            print 'Total time needed for single frame', time2 - time1
            points_on_im = co.data.depth3d.copy()
            #points_on_im[np.sum(points_on_im,axis=2)==0,:]=np.array([1,0,1])
            for calc, point1, point2 in zip(
                co.segclass.nz_objects.centers_to_calculate,
                co.segclass.nz_objects.initial_center,
                co.segclass.nz_objects.center):
                if point1[0] != -1 :
                    if calc:
                         cv2.arrowedLine(points_on_im,
                                        (point1[1], point1[0]),
                                        (point2[1], point2[0]), [0, 1, 0], 2, 1)    
                    else:
                        cv2.arrowedLine(points_on_im,
                                        (point1[1], point1[0]),
                                        (point2[1], point2[0]), [0, 0, 1], 2, 1)
            
            co.im_results.images.append(points_on_im)
            return found_objects_mask

def extract_background_values():
    '''function to extract initial background values from initial_im_set'''
    _, _, im_num = co.data.initial_im_set.shape
    valid_freq = np.zeros(co.data.initial_im_set[:, :, 0].shape)
    co.data.background = np.zeros(co.data.initial_im_set[:, :, 0].shape)
    initial_nonzero_im_set = np.zeros(co.data.initial_im_set.shape)
    initial_nonunary_im_set = np.zeros_like(co.data.depth_im)
    valid_values=np.zeros_like(co.data.background)
    for count in range(im_num):
        valid_values[
            co.data.initial_im_set[:,:,count]>0
            ]=co.data.initial_im_set[:,:,count][
                co.data.initial_im_set[:,:,count]>0]
        co.data.background = co.data.background + \
            co.data.initial_im_set[:, :, count]
        valid_freq[co.data.initial_im_set[:, :, count] != 0] += 1
        
        initial_nonzero_im_set[:, :, count] += co.data.initial_im_set[
            :, :, count] != 0
    co.data.valid_values=(valid_values*255).astype(np.uint8)

    co.data.trusty_pixels = ((valid_freq) == np.max(valid_freq)).astype(np.uint8)
    #cv2.imshow('test1',co.data.trusty_pixels.astype(float))
    #cv2.waitKey(0)
    valid_freq[valid_freq == 0] = 1
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
