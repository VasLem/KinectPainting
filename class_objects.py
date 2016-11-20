import numpy as np
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError

def find_nonzero(arr):
    return np.fliplr(cv2.findNonZero(arr).squeeze())

class ConvexityDefect(object):
    '''Convexity_Defects holder'''

    def __init__(self):
        self.hand = None


class Interp(object):
    '''Interpolated Contour variables'''

    def __init__(self):
        self.vecs_starting_ind = None
        self.vecs_ending_ind = None
        self.points = None
        self.final_segments = None


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
            # print "No results to show"
            return 1
        shapes = [(im_shape[0], im_shape[1], c, len(im_shape))
                  for (c, im_shape) in enumerate([im.shape for im in self.images])]
        isrgb = 1 if sum(zip(*shapes)[3]) != 2 * len(shapes) else 0
        sorted_images = [self.images[i] for i in list(
            zip(*sorted(shapes, key=lambda x: x[0], reverse=True))[2])]
        imy, imx, _, _ = tuple([max(coord) for
                                coord in zip(*shapes)])
        yaxis = len(self.images) / self.maxdim + 1
        xaxis = min(self.maxdim, len(self.images))
        if not isrgb:
            montage = 255 * \
                np.ones((imy*yaxis, imx*xaxis), dtype=np.uint8)
        elif isrgb:
            montage = 255 * \
                np.ones((imy*yaxis, imx*xaxis, 3), dtype=np.uint8)
        x_init = 0
        for count, image in enumerate(sorted_images):
            image = ((image - np.min(image)) / float(np.max(image) - np.min(image)) *
                     255).astype(np.uint8)
            if isrgb:
                if len(image.shape) == 2:
                    image = np.tile(image[:, :, None], (1, 1, 3))
            if not isrgb:
                montage[(count / self.maxdim) * imy:(count / self.maxdim + 1)
                        * imy, x_init:x_init + image.shape[1]] = image
            else:
                montage[(count / self.maxdim) * imy:(count / self.maxdim + 1)
                        * imy, x_init:x_init + image.shape[1], :] = image
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
                print(e)
                raise(e)
        else:
            cv2.imwrite(var2, montage)
        return

    def print_results(self, filename):
        '''class to print data to file'''
        with open(filename, 'w') as fil:
            for line in self.data:
                fil.write(line + '\n')


class Flag(object):
    '''necessary flags for program flow'''

    def __init__(self):
        self.exists_lim_calib_image = 0
        self.read = ''
        self.save = ''


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
        self.calib_edges = None
        self.calib_frame = np.zeros(0)


class Hull(object):
    '''Convex Hulls of contours'''

    def __init__(self):
        self.hand = None


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
        #lims: l->t->r->b
        self.nonconvex_edges_lims=[]
        self.convex_edges_lims=[]
        self.edges_positions_indices=[]
        self.edges_positions=[]
        #trusty_pixels is 1 for the pixels that remained nonzero during
        #initialisation
        self.trusty_pixels = np.zeros(1)
        #valid_values hold the last seen nonzero value of an image pixel
        #during initialisation
        self.valid_values = np.zeros(1)
        self.all_positions=np.zeros(1)
        self.background = np.zeros(1)
    def find_non_convex_edges_lims(self,edges_mask,edge_tolerance=10):
        '''
        Find non convex symmetrical edges minimum orthogonal lims with some tolerance
        Inputs: positions,edges mask[,edges tolerance=10]
        '''
        self.edges_positions_indices=np.nonzero(cv2.dilate(
            edges_mask,np.ones((3,3),np.uint8),cv2.CV_8U)>0)
        self.edges_positions=np.transpose(np.array(self.edges_positions_indices))
        lr_positions=self.edges_positions[np.abs(self.edges_positions[:,0]-edges_mask.shape[0]/2.0)<1,:]
        tb_positions=self.edges_positions[np.abs(self.edges_positions[:,1]-edges_mask.shape[1]/2.0)<1,:]
        self.nonconvex_edges_lims=np.array(
            [np.min(lr_positions[:,1])+edge_tolerance,
             np.min(tb_positions[:,0])+edge_tolerance,
             np.max(lr_positions[:,1])-edge_tolerance,
             np.max(tb_positions[:,0])-edge_tolerance])

    def find_convex_edges_lims(self,positions,edges_mask):
        '''
        Find convex edges minimum orthogonal lims
        '''
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
        calib_positions=positions[edges_mask>0,:]
        calib_dists=calculate_cart_dists(
           calib_positions,
          np.array([0,0]))
        
        upper_left=calib_positions[np.argmin(calib_dists),:]
        calib_dists2=calculate_cart_dists(
            calib_positions,
            np.array([edges_mask.shape[0],
                      edges_mask.shape[1]]))
        lower_right=calib_positions[np.argmin(calib_dists2),:]
        #Needs filling
        self.convex_edges_lims=[]

class Model(object):
    '''variables for modelling nonlinearities'''

    def __init__(self):
        self.noise_model = (0, 0)
        self.med = None
        self.var = None


class KalmanFilter:

    def __init__(self):
        self.prev_objects_mask = np.zeros(0)
        self.cur_objects_mask = np.zeros(0)
        self.can_exist = np.zeros(0)


class NoiseRemoval:

    def remove_noise(self):
        # All noisy pixels are either white or black. We must remove this shit.
        time1 = time.clock()
        data.depth_im *= data.depth_im < 0.35
        data.depth_im /= 0.35
        time2 = time.clock()


class ExistenceProbability:

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
        if self.init_val == 0:
            self.wearing_mat = np.zeros(segclass.total_obj_num)
            self.init_val = 1
            # self.distance_mask = np.pad(
            # np.ones(tuple(np.array(data.depth_im.shape) - 2)), 1, 'constant')
            sums = np.sum(masks.calib_frame[
                          :, 1:masks.calib_frame.shape[1] / 2], axis=0) > 0
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
            self.wearing_mat[:segclass.nz_objects.count] > 0)[0]

        self.wearing_mat -= 1
        # print 'Segments checked: ', self.can_exist
        '''
        im_res = np.zeros_like(data.depth_im, dtype=np.uint8)
        for val in self.can_exist:
            im_res[segclass.objects.image == val] = 255
        im_results.images.append(im_res)
        '''



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
        if len(points) == 1:
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
        time01 = time.clock()
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
                    self.locs[:locs.shape[0], count] = locs[:, 0] + locs[:, 1] * 1j
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
            time02 = time.clock()
            '''
            print 'time to load variables and compute \
                existence:', time02 - time01, 's'
            '''
        time001 = time.clock()
        '''cv2.imshow('test',self.uint8_depth_im)
        cv2.waitKey(0)
        '''

        for count in np.arange(self.count + 1)[self.centers_to_calculate]:
            xcoords = self.locs[
                :self.pixsize[count], count].real.astype(int)
            ycoords = self.locs[
                :self.pixsize[count], count].imag.astype(int)
            self.masses[count] = np.sum(data.uint8_depth_im[xcoords, ycoords])
            '''if self.masses[count]==0:
                print 'masses=0',xcoords,ycoords,self.pixsize[count]
            '''
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
                    count, :self.pixsize[count]] = \
                    data.uint8_depth_im[xcoords, ycoords]
        time002 = time.clock()
        '''
        print 'time to fill vals:', time002 - time001, 's'
        '''
        time02 = time.clock()
        '''
        print 'found centers in', time02 - time01, 's'
        '''
        if first_time:
            self.initial_center = self.center.copy()


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


class Segmentation(object):
    ''' Objects used for background segmentation '''

    def __init__(self):
        self.bounding_box = []
        self.needs_segmentation = 1
        self.check_if_segmented_1 = 0
        self.check_if_segmented_2 = 0
        self.prev_im = np.zeros(0)
        self.exists_previous_segmentation = 0
        self.initialised_centers=0
        self.z_objects = SceneObjects()
        self.nz_objects = SceneObjects()
        self.neighborhood_existence = np.zeros(0)
        self.proximity_table = np.zeros(0)
        self.filled_neighborhoods = []
        self.found_objects = np.zeros(0)
        self.total_obj_num = 0
    
    def flush_previous_segmentation(self):
        self.bounding_box=[]
        self.z_objects= SceneObjects()
        self.nz_objects= SceneObjects()
        self.proximity_table = np.zeros(0)
        self.filled_neighborhoods=[]
        self.found_objects=np.zeros(0)
        self.total_obj_num=0

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
            self.neighborhood_existence[count, :] =\
                np.sum(self.proximity_table
                       == count, axis=0)

    def find_objects(self,constants):
        time1 = time.clock()
        check_atoms = []
        # nz_objects.center at the beginning of center list so following is
        # valid
        for count1, vec in\
                enumerate(self.nz_objects.center_displacement):
            if (abs(vec[0]) > constants['min_displacement'] or abs(vec[1]) >
                constants['min_displacement']):
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
                            constants['depth_tolerance']) * (vals > 0)
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
                            constants['depth_tolerance']) * (last_vals > 0)
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
        #euclid_dists = np.abs(np.transpose(complex_inters) - complex_inters)
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

# pylint: disable=C0103
contours = Contour()
counters = Counter()
data = Data()
flags = Flag()
lims = Lim()
masks = Mask()
meas = Measure()
models = Model()
paths = Path()
points = Point()
thres = Threshold()
im_results = Result()
segclass = Segmentation()
existence = ExistenceProbability()
interpolated = contours.interpolated
noise_proc = NoiseRemoval()
