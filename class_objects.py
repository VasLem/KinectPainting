import numpy as np
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
from scipy import ndimage

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
        self.trusty_pixels = np.zeros(1)
        self.depth3d = np.zeros(1)
        self.depth = []
        self.color = []
        self.depth_im = np.zeros(1)
        self.color_im = np.zeros(1)
        self.hand_shape = np.zeros(1)
        self.hand_im = np.zeros(1)
        self.initial_im_set = np.zeros(1)
        self.depth_mem = []
        self.background = np.zeros(1)
        self.trusty_pixels = np.zeros(1)


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
        montage = 255 * \
            np.tile(np.ones((imy, imx), dtype=np.uint8), (yaxis, xaxis))
        if isrgb:
            montage = np.tile(montage[:, :, None], (1, 1, 3))
        x_init = 0
        for count, image in enumerate(sorted_images):
            image = ((image - np.min(image)) / float(np.max(image) - np.min(image)) *
                     255).astype(int)
            if isrgb:
                if len(image.shape) == 2:
                    image = np.tile(image[:, :, None], (1, 1, 3))
            if not isrgb:
                montage[(count / self.maxdim) * imy:(count / self.maxdim + 1)
                        * imy, x_init:x_init + image.shape[1]] = image
                #print (count / self.maxdim) * imy,(count / self.maxdim + 1)* imy,x_init,x_init + image.shape[1]
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
        else:
            cv2.imwrite(var2, montage)
        # print 'isrgb',isrgb
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


class ExistenceProbability:

    def __init__(self):
        self.init_val = 0
        self.distance = np.zeros(0)
        self.distance_mask = np.zeros(0)
        self.objects_mask = np.zeros(0)
        self.can_exist = np.zeros(0)
        self.max_distance = 50
        self.framex1 = 0
        self.framex2 = 0
        self.framey1 = 0
        self.framey2 = 0
        self.always_checked = []
        self.wearing_par=10
        self.wearing_mat=np.zeros(0)
    def calculate(self):
        if self.init_val == 0:
            self.wearing_mat=np.zeros(segclass.objects.obj_count+1)
            self.init_val = 1
            # self.distance_mask = np.pad(
            # np.ones(tuple(np.array(data.depth_im.shape) - 2)), 1, 'constant')
            sums = np.sum(masks.calib_frame[
                          :, 1:masks.calib_frame.shape[1] / 2], axis=0) > 0
            self.framex1 = np.where(np.diff(sums))[0] + self.max_distance
            self.framex2 = meas.imx - self.framex1
            self.framey1 = self.max_distance
            self.framey2 = meas.imy - self.max_distance
            for count, center in enumerate(segclass.objects.initial_center):
                if center[0] < self.framey1 or center[0] > self.framey2 or\
                        center[1] < self.framex1 or center[1] > self.framex2:
                    self.always_checked.append(count)

            #self.distance_mask *= masks.calib_frame

            # self.distance = cv2.distanceTransform(
            #    self.distance_mask.astype(np.uint8), cv2.DIST_L1, 3)
            #self.objects_mask = np.ones_like(data.depth_im)
            # plt.imshow(self.distance)
            # plt.savefig('initial_existence_im,jpg')
        
        new_arrivals=[]
        for neighborhood in segclass.objects.filled_neighborhoods:
             new_arrivals += neighborhood
        self.wearing_mat[new_arrivals+self.always_checked]=self.wearing_par
        self.can_exist=np.where(self.wearing_mat>0)[0]
        self.wearing_mat-=1
        print 'Segments checked: ', self.can_exist
        '''
        im_res = np.zeros_like(data.depth_im, dtype=np.uint8)
        for val in self.can_exist:
            im_res[segclass.objects.all_objects_im == val] = 255
        im_results.images.append(im_res)
        '''

class SceneObjects(object):
    '''Class to describe found scene objects'''

    def __init__(self):
        self.pixsize = []
        self.xsize = []
        self.ysize = []
        self.contour = []
        self.center = np.zeros(0)
        self.initial_center = np.zeros(0)
        self.initial_vals_set = []
        self.valid_vals_loc = np.zeros(0)
        self.center_displacement = []
        self.obj_count = 0
        self.all_objects_im = np.zeros(0)
        self.is_unreliable = np.zeros(0)
        self.found_objects = []
        self.proximity_table = np.zeros(0)
        self.neighborhood_existence = np.zeros(0)
        self.filled_neighborhoods = []
        self.valid_vals=np.zeros(0)
        self.masses=np.zeros(0)
        self.centers_to_calculate=np.zeros(0)

    def find_centers_displacement(self):
        self.center_displacement = self.center - self.initial_center
        print self.center_displacement
    def find_object_center(self):
        '''Find scene objects centers of mass'''
        time1 = time.clock()
        preload =self.valid_vals_loc.size>0
        # objects_on_im=np.zeros_like(data.depth_im)
        if preload:
            time01 = time.clock()
            existence.calculate()
            self.centers_to_calculate = np.zeros((self.obj_count+1),dtype=bool)
            
            self.centers_to_calculate[np.array(existence.can_exist)]= 1
            time02 = time.clock()
            print 'time to load variables and compute \
                existence:', time02 - time01, 's'
        else:

            self.center = np.zeros((self.obj_count + 1,2),dtype=int)
            self.centers_to_calculate = np.ones((self.obj_count+1),dtype=bool) 
            self.is_unreliable = np.zeros_like(self.centers_to_calculate)


        dim1=max(self.pixsize)
        if not preload:
            self.valid_vals_loc=np.zeros((dim1,self.obj_count+1),dtype=complex)
            for count in range(self.obj_count+1):
                if self.xsize[count] < 3 or self.ysize[count] < 3 or\
                self.pixsize[count] < 900:
                    self.is_unreliable[count] = 1
                
                valid_vals_mask = ((self.all_objects_im == count) *
                                   data.trusty_pixels)
                if np.all(valid_vals_mask == 0):
                    self.is_unreliable[count] = 1
                valid_vals_loc = np.array(
                    np.nonzero(valid_vals_mask),dtype=int).T
                self.valid_vals_loc[:valid_vals_loc.shape[0],count]=valid_vals_loc[:,0]+valid_vals_loc[:,1]*1j
                
                self.pixsize[count]=valid_vals_loc.shape[0]
        time01=time.clock()
        if self.valid_vals.size==0:
            self.valid_vals=np.empty((self.obj_count+1,dim1))
            self.masses=np.ones(self.obj_count+1,dtype=int)
        
        if preload:
            self.center=self.initial_center.copy()
        tmp = (255*data.depth_im).astype(np.uint8)
        
        time001=time.clock()
        for count in np.arange(self.obj_count+1)[self.centers_to_calculate]:
            xcoords=self.valid_vals_loc[:self.pixsize[count],count].real.astype(int)
            ycoords=self.valid_vals_loc[:self.pixsize[count],count].imag.astype(int)
            self.masses[count]=np.sum(tmp[xcoords,ycoords])
            if not self.is_unreliable[count] and self.masses[count]>0:

                self.valid_vals[count,:self.pixsize[count]]=tmp[xcoords,ycoords
                                                                ][None,:]
                complex_res=np.dot(
                    self.valid_vals[count,:self.pixsize[count]],
                    self.valid_vals_loc[:self.pixsize[count],count])/self.masses[count]
                self.center[count,:]=np.array([complex_res.real,complex_res.imag])
            else:
                if not preload:
                    complex_res=np.mean(self.valid_vals_loc[:self.pixsize[count],count])
                    self.center[count,:]=np.array([complex_res.real.astype(int),complex_res.imag.astype(int)])

        #'''        
        #self.center=np.array(np.unravel_index(self.center.astype(int),data.depth_im.shape)).T
        time002=time.clock()
        print 'time to fill valid_vals:',time002-time001,'s'
        #time001=time.clock()
        # mult=np.dot(self.valid_vals,self.valid_vals_loc)
        #time002=time.clock()
        #print 'time to multiply two arrays',time002-time001

        # self.center=np.array(np.unravel_index((np.diagonal(mult)/self.masses).astype(int),data.depth_im.shape)).T
        time02=time.clock()
        print 'found centers in', time02-time01, 's'
        if not preload:
            self.initial_center = self.center.copy()
            zcenters = (self.center[:, 0] + self.center[:, 1] * 1j)[:,None]
            distances = abs(zcenters-np.transpose(zcenters))
            sorted_indices = np.argsort(distances, axis=0)
            self.proximity_table = sorted_indices[:9, :]
            self.neighborhood_existence = np.zeros((self.obj_count + 1,
                                                    self.obj_count + 1))
            for count in range(self.obj_count + 1):
                self.neighborhood_existence[count, :] =\
                    np.sum(self.proximity_table
                           == count, axis=0)
        time2 = time.clock()
    def find_minmax_coords(self, count):
        [y_max, x_max] = np.max(self.valid_vals_loc[count], axis=1)
        [y_min, x_min] = np.min(self.valid_vals_loc[count], axis=1)
        return (y_min, x_min, y_max, x_max)

    def find_objects(self):
        inters = []
        count = 0
        time1 = time.clock()
        check_atoms = []

        for count1, vec in\
                enumerate(self.center_displacement):
            if vec[0] != 0 and vec[1] != 0:
                if np.linalg.norm(vec) > 1:
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

        print 'Created cluster', neighborhoods, 'in', time2 - time1, 's'
        print 'Filled cluster is', self.filled_neighborhoods
        time1 = time.clock()
        found_objects = np.zeros(data.depth_im.shape)
        for count, neighborhood in enumerate(self.filled_neighborhoods):
            neighborhood_mask = np.in1d(self.all_objects_im.ravel(), neighborhood
                                       ).reshape(data.depth_im.shape)

            neighborhood_vals = data.depth_im[neighborhood_mask]
            mean_val = np.mean(neighborhood_vals)

            #singular_vals = neighborhood_vals[check_singularities]
            #singular_range = [np.min(singular_vals), np.max(singular_vals)]

            found_objects[neighborhood_mask] =\
                (np.abs(neighborhood_vals - mean_val)
                 < 0.05) * (1 / float(count + 1))
        im_results.images.append(found_objects)
        time2 = time.clock()
        print 'Found objects in', time2 - time1, 's'

        '''for count1 in range(len(self.size)):
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
        self.mask = np.array([np.matrix('[1 0 0;0 -1 0;0 0 0]'),
                              np.matrix('[0 1 0;0 -1 0;0 0 0]'),
                              np.matrix('[0 0 1;0 -1 0;0 0 0]'),
                              np.matrix('[0 0 0;1 -1 0;0 0 0]'),
                              np.matrix('[0 0 0;0 -1 1;0 0 0]'),
                              np.matrix('[0 0 0;0 -1 0;1 0 0]'),
                              np.matrix('[0 0 0;0 -1 0;0 1 0]'),
                              np.matrix('[0 0 0;0 -1 0;0 0 1]')])
        self.shift_ind = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1],
                                   [0, 1], [1, -1], [1, 0], [1, 1]])
        self.segments = ''
        self.segment_values = ''
        self.array_instances = np.ones(1)
        self.kernel = np.ones((3, 3))
        self.seed_x_coord = None
        self.seed_y_coord = None
        self.segment_enum = ''
        self.objects = SceneObjects()
        self.needs_segmentation = 1
        self.check_if_segmented_1 = 0
        self.check_if_segmented_2 = 0
        self.prev_im = np.zeros(0)
        self.exists_previous_segmentation = 0
        self.all_objects_im = np.zeros(0)

    def start_segmentation(self, seed_num, meas):
        tmpx = np.linspace(0, meas.imx - 1, seed_num, dtype=int)
        tmpy = np.linspace(0, meas.imy - 1, seed_num, dtype=int)
        self.seed_x_coord = np.tile(tmpx[:, None], (1, seed_num)).ravel()
        self.seed_y_coord = np.transpose(np.tile(tmpy[:, None],
                                                 (1, seed_num))).ravel()
        self.segment_values = np.zeros((meas.imy, meas.imx))
        self.segment_values[self.seed_y_coord,
                            self.seed_x_coord] = np.array(range(seed_num**2))


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
