import numpy as np
import cv2
import time
from cv_bridge import CvBridge, CvBridgeError
import matplotlib.pyplot as plt
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
        self.depth3d = np.zeros(1)

class Result(object):
    '''class to keep results'''

    def __init__(self):
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
        montage = 255 * np.tile(np.ones((imy, imx),dtype=np.uint8), (yaxis, xaxis))
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
            cv2.imshow('results',montage)
        elif var1 == 'ros':
            try:
                var2[0].publish(var2[1].cv2_to_imgmsg(montage,"bgr8"))
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
        self.erode_size = 0


class Model(object):
    '''variables for modelling nonlinearities'''

    def __init__(self):
        self.noise_model = (0, 0)
        self.med = None
        self.var = None


class SceneObjects(object):
    '''Class to describe found scene objects'''

    def __init__(self):
        self.mask = []
        self.size = []
        self.contour = []
        self.center = []
        self.initial_center = []
        self.valid_vals = []
        self.valid_vals_loc = []
        self.center_displacement = []
        self.obj_count=0

    def find_centers_displacement(self):
        self.center_displacement = [i - j for i, j in
                                    zip(self.center, self.initial_center)]

    def find_object_center(self):
        time1=time.clock()
        '''Find scene objects centers of mass'''
        preload = bool(self.valid_vals_loc)
        centers = []
        #objects_on_im=np.zeros_like(data.depth_im)
        for count, obj in enumerate(self.mask):
            if preload:
                valid_vals_loc = self.valid_vals_loc[count]
                valid_vals = data.depth_im[valid_vals_loc]
            else:
                valid_vals_mask = obj.astype(np.uint8) * data.trusty_pixels
                valid_vals_loc = np.nonzero(valid_vals_mask)
                valid_vals = data.depth_im[valid_vals_mask>0]
                self.valid_vals_loc.append(valid_vals_loc)
            weights=np.maximum(valid_vals,1/float(256))
            total_mass = np.sum(np.maximum(valid_vals,1/float(256)))
            center=(np.sum((np.transpose(valid_vals_loc) *
                                    valid_vals[:,None]),axis=0) /float(
                                        total_mass)).astype(int)
            #objects_on_im[valid_vals_loc]+=data.depth_im[valid_vals_loc]
            centers.append(center)

        if preload:
            self.center = centers
        else:
            self.initial_center = centers
        #im_results.images.append(data.depth_im)
        #im_results.images.append(objects_on_im)
        time2=time.clock()
        #print "find_object_center:",time2-time1,"s"
    def find_intersecting_points(self):
        time1=time.clock()
        inters = []
        count = 0
        for count1 in range(len(self.size)):
            for count2 in range(count1, len(self.size)):
                count += 1
                vec2 = self.center_displacement[count2]
                vec1 = self.center_displacement[count1]
                if np.sqrt(vec1[0]**2+vec1[1]**2)>2 and\
                   np.sqrt(vec2[0]**2+vec2[1]**2)>2:
                    pnt1 = self.initial_center[count1]
                    pnt2 = self.initial_center[count2]
                    v1v2cross = np.cross(vec1, vec2)
                    if np.any(v1v2cross>0):
                        lam = np.sqrt(np.sum(np.cross(pnt2 - pnt1, vec2) ** 2)) /\
                            np.sqrt(np.sum(v1v2cross**2))
                        pnt = pnt1 + lam * vec1
                        if pnt[0] > 0 and pnt[0] < meas.imy\
                           and pnt[1] > 0 and pnt[1] < meas.imx and\
                            masks.calib_frame[int(pnt[0]),int(pnt[1])]>0:
                            inters.append(pnt.astype(int))
        complex_inters = np.array(
            [[complex(point[0], point[1]) for point in inters]])
        #euclid_dists = np.abs(np.transpose(complex_inters) - complex_inters)
        time2=time.clock()
        print "find_intersecting_points:",time2-time1,"s"
        if inters:
            desired_center = np.median(np.array(inters), axis=0)
            real_center_ind = np.argmin(
                np.abs(complex_inters -
                    np.array([[complex(desired_center[0],desired_center[1])]])))
            return inters, inters[real_center_ind]
        else:
            return 'Not found intersection points',[]

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
        self.needs_segmentation=1
        self.check_if_segmented_1=0
        self.check_if_segmented_2=0
        self.prev_im=np.zeros(0)
        self.exists_previous_segmentation=0
        self.all_objects_im=np.zeros(0)      
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
interpolated = contours.interpolated
