import time
import numpy as np
import os
import urllib
import cv2
import cProfile as profile


class QuadTree(object):

    def __init__(self):
        self.image = None
        self.quads = []
        self.discr_val = 10
        self.partition = [0, 0]

    def im_update(self, im):
        self.image = im.copy().astype(float)
        self.quads = []
        self.image_partition([np.array([0, 0]), np.array([0, im.shape[1] - 1]),
                              np.array([im.shape[0] - 1, im.shape[1] - 1]),
                              np.array([im.shape[0] - 1, 0])])

    def check_points(self, plist):
        for count, point1 in enumerate(plist):
            for point2 in plist[count + 1:]:
                if np.abs(self.image[point1[0], point1[1]] -
                          self.image[point2[0], point2[1]]) > self.discr_val:
                    diff = point1 - point2
                    if diff[0] != 0:
                        self.partition[0] = 1
                    if diff[1] != 0:
                        self.partition[1] = 1
                    if self.partition[0] and self.partition[1]:
                        break
        return self.partition

    def image_partition(self, plist):
        '''
        Input 4 points 2d list, refering to corners of block, plus image
        '''
        [p0, p2, p4, p6] = plist[:]
        if np.abs((p0[0] - p4[0]) * (p0[0] - p4[0]) + (p0[1] - p4[1]) * (p0[1] - p4[1])) < 400:
            self.quads.append(plist)
            return 0
        p1 = (p0 + p2) / 2
        p3 = (p2 + p4) / 2
        p8 = np.array([p3[0], p1[1]])
        allow_partition = self.check_points([p8] + plist)
        if allow_partition[0] == 1 and allow_partition[1] == 1:
            p5 = (p4 + p6) / 2
            p7 = (p6 + p0) / 2
            self.image_partition([p0, p1, p8, p7])
            self.image_partition([p1, p2, p3, p8])
            self.image_partition([p7, p8, p5, p6])
            self.image_partition([p8, p3, p4, p5])
        elif allow_partition[0] == 1:
            p5 = (p4 + p6) / 2
            self.image_partition([p0, p1, p5, p6])
            self.image_partition([p1, p2, p4, p5])

        elif allow_partition[1] == 1:
            p7 = (p6 + p0) / 2
            self.image_partition([p0, p2, p3, p7])
            self.image_partition([p7, p3, p4, p6])
        else:
            self.quads.append(plist)


def main():
    quadtree = QuadTree()
    if not os.path.exists('lena.jpg'):
        urllib.urlretrieve("http://www.ece.rice.edu/~wakin/images/lenaTest2.jpg",
                           "lena.jpg")

    lena = cv2.imread('lena.jpg', -1)
    # time1=time.clock()
    quadtree.im_update(lena)
    # time2=time.clock()
    # print time2-time1
    lena = np.tile(lena[:, :, None], (1, 1, 3))
    for quad in quadtree.quads:
        for point in quad:
            lena[point[0], point[1], :] = np.array([255, 0, 0])
    cv2.imshow('Lena', lena)

    cv2.waitKey(0)

main()
# profile.run('main()')
