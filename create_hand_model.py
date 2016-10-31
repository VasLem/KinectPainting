
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi
import time
from numpy import cos, sin
import numpy as np


def linedraw(*vargs):
    '''Draw line/s.
    Input: list of args, with arg: arg[point[,line],dim,1]
    '''
    ax = vargs[0]
    args = vargs[1]
    colors = plt.cm.Set1(np.linspace(0, 1, len(args)))
    count1 = 0
    linehandles = []
    for arg in list(args):
        if len(arg.shape) == 4:
            for count2 in range(arg.shape[1]):
                linehandles.append(ax.plot(arg[:, count2, 0, 0], arg[:, count2, 1, 0], arg[:, count2, 2, 0], c=plt.cm.Set1(count1),
                                           marker='.', linewidth=2, markersize=4))
                count1 += 11
                count1 = count1 % 251
        elif len(arg.shape) == 3:  # not tested
            linehandles.append(ax.plot(arg[:, 0, 0], arg[:, 1, 0], arg[:, 2, 0],
                                       c=plt.cm.Set1(count1),
                                       marker='.', linewidth=2, markersize=4))
            count1 += 11
            count1 = count1 % 251
    return linehandles

def gmm3d(point,means,sig):
    res=0
    for mean in means:
        res+=np.exp(np.dot((point-mean).T,point-mean)/(2*sig^2))
    return res
    


class HandMetrics:

    def __init__(self):

        # Accept as inputs the scaling of the dimensions of the hand
        # lets say
        scale = 1

        # Carpals->CMC->Metacarpals->MCP->Proximal phalanges->PIP->Intermediate phalanges->
        # DIP->Distal phalanges->TCP
        # The thumb has no intermediate phalanges
        # FE=Flexion-Extension, AA=Adduction-Abduction ,
        # PS=Pronation-Suspination

        # self.w_dh_pars[joint,theta|d|alpha|a,0]:wrist dh_pars
        self.w_dh_pars = np.zeros((3, 4, 1))
        self.w_dh_rot_lims_plus = np.zeros_like(self.w_dh_pars)
        self.w_dh_rot_lims_minu = np.zeros_like(self.w_dh_pars)
        # Wrist FE
        self.w_dh_pars[0, :, 0] = np.array([pi / 2, 0, -pi / 2, 0])
        self.w_dh_rot_lims_plus[0, 0, 0] = 0.7
        self.w_dh_rot_lims_minu[0, 0, 0] = 0.7
        # Wrist AA
        self.w_dh_pars[1, :, 0] = np.array([pi / 2, 0, pi / 2, 0])
        self.w_dh_rot_lims_plus[1, 0, 0] = 0.93
        self.w_dh_rot_lims_minu[1, 0, 0] = 0.93
        # Wrist PS
        self.w_dh_pars[2, :, 0] = np.array([-pi / 2, 0, pi / 2, 0])
        self.w_dh_rot_lims_plus[2, 0, 0] = 1.05
        self.w_dh_rot_lims_minu[2, 0, 0] = 1.75

        # self.nf_dh_pars[joint,theta|d|alpha|a,finger]: finger dh_pars, not including
        # thumb, starting from index.
        self.nf_dh_pars = np.zeros((6, 4, 4))
        self.nf_dh_rot_lims_plus = np.zeros((6, 4, 4))
        self.nf_dh_rot_lims_minu = np.zeros((6, 4, 4))

        # CMC-FE
        self.nf_dh_pars[0, :, :] = np.array([[pi, pi, pi, pi],
                                             [71, 71, 65, 63],
                                             [pi / 2.0, pi / 2.0,
                                              pi / 2.0, pi / 2.0],
                                             [0, 0, 0, 0]]).astype(float)
        self.nf_dh_rot_lims_plus[0, 0, :] = np.array(
            [[0, 0, pi / 27, pi / 27.0]])
        self.nf_dh_rot_lims_minu[0, 0, :] = np.array(
            [[0, 0, 2 * pi / 27, 2 * pi / 27]])
        # MCP-AA
        self.nf_dh_pars[1, :, :] = np.array([[pi / 2.0, pi / 2.0, pi / 2.0, pi / 2.0],
                                             [-11, 0, 8, 19],
                                             [pi / 2.0, pi / 2.0,
                                              pi / 2.0, pi / 2.0],
                                             [0, 0, 0, 0]]).astype(float)

        self.nf_dh_rot_lims_plus[1, 0, :] = np.array([[pi / 9, pi / 9, pi / 9,
                                                       pi / 9]])

        self.nf_dh_rot_lims_minu[1, 0, :] = np.array([[0, 0,
                                                       0, 0]])

        # MCP-FE
        self.nf_dh_pars[2, :, :] = np.array([[0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [-pi / 2.0, -pi / 2.0, -
                                              pi / 2.0, -pi / 2.0],
                                             [30, 35, 33, 24]]).astype(float)

        self.nf_dh_rot_lims_plus[2, 0, :] = np.array([[0.61, 0.7, pi / 4,
                                                       0.83]])

        self.nf_dh_rot_lims_minu[2, 0, :] = np.array(
            [[0.61, 0.7, pi / 4, 0.83]])
        # PIP-FE
        self.nf_dh_pars[3, :, :] = np.array([[0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [20, 26, 25, 20]]).astype(float)
        self.nf_dh_rot_lims_plus[3, 0, :] = np.array(
            [[1.75, 1.75, 1.75, 1.75]])
        self.nf_dh_rot_lims_minu[3, 0, :] = np.array(
            [[0.09, 0.09, 0.09, 0.09]])
        # DIP-FE
        self.nf_dh_pars[4, :, :] = np.array([[0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0]]).astype(float)
        self.nf_dh_rot_lims_plus[4, 0, :] = np.array(
            [[1.58, 1.58, 1.58, 1.58]])
        self.nf_dh_rot_lims_minu[4, 0, :] = np.array(
            [[0.09, 0.09, 0.09, 0.09]])

        # TCP:tool center point
        self.nf_dh_pars[5, :, :] = np.array([[0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [0, 0, 0, 0],
                                             [18, 18, 16, 15]]).astype(float)
        # self.tf_dh_pars:thumb parameters
        self.tf_dh_pars = np.zeros((6, 4, 1))
        self.tf_dh_rot_lims_plus = np.zeros_like(self.tf_dh_pars)
        self.tf_dh_rot_lims_minu = np.zeros_like(self.tf_dh_pars)
        # CMC-AA
        self.tf_dh_pars[0, :, 0] = np.array([-0.4636, 13, pi / 2, 15])
        self.tf_dh_rot_lims_plus[0, 0, 0] = 0.17
        self.tf_dh_rot_lims_minu[0, 0, 0] = 0.17
        # CMC-FE
        self.tf_dh_pars[1, :, 0] = np.array(
            [pi / 3.0, 5, -110 * pi / 180.0, 42.236])
        self.tf_dh_rot_lims_plus[1, 0, 0] = 1
        self.tf_dh_rot_lims_minu[1, 0, 0] = 0.35

        # MCP-AA
        self.tf_dh_pars[2, :, 0] = np.array([0, 0, pi / 2, 0])

        self.tf_dh_rot_lims_plus[2, 0, 0] = 4 * pi / 10
        self.tf_dh_rot_lims_minu[2, 0, 0] = pi / 10
        # MCP-FE
        self.tf_dh_pars[3, :, 0] = np.array([0, 0, -pi / 2.0, 25])
        self.tf_dh_rot_lims_plus[3, 0, 0] = 0.17
        self.tf_dh_rot_lims_minu[3, 0, 0] = 0.17
        # IP-FE
        self.tf_dh_pars[4, :, 0] = np.array([0, 0, pi / 2.0, 0])
        self.tf_dh_rot_lims_plus[4, 0, 0] = 1.59
        self.tf_dh_rot_lims_minu[4, 0, 0] = 0.09
        # TCP
        self.tf_dh_pars[5, :, 0] = np.array([0, 0, 0, 20])
        self.tf_dh_pars = self.tf_dh_pars.astype(float)
        # scale data
        '''
        self.tf_dh_pars[:, :, 1] *= scale
        self.tf_dh_pars[:, :, 3] *= scale
        self.nf_dh_pars[:, :, 1] *= scale
        self.nf_dh_pars[:, :, 3] *= scale
        '''

    def forward_kinematics(self, dh_pars, init_transform, target):
        '''Inputs : 
            -DH parameters[joint,theta|d|alpha|a,chain]
            -Initial Transform
            -Target position relative to tool frame
        '''
        def z_rt(d, theta):
            # Z axis rotation and translation
            return np.array([[cos(theta), -sin(theta), 0, 0],
                             [sin(theta), cos(theta), 0, 0],
                             [0, 0, 1, d],
                             [0, 0, 0, 1]])

        def x_rt(a, alpha):
            # X axis rotation and translation
            return np.array([[1, 0, 0, a],
                             [0, cos(alpha), -sin(alpha), 0],
                             [0, sin(alpha), cos(alpha), 0],
                             [0, 0, 0, 1]])

        def tot_rt(arr):
            # total rotation and translation of a type of joint for all fingers
            res = []
            for c in range(arr.shape[1]):
                res.append(
                    np.dot(z_rt(arr[1, c], arr[0, c]), x_rt(arr[3, c], arr[2, c])))
            return res
        jpos = []
        in_transform = init_transform.copy()
        curr = [in_transform] * dh_pars.shape[2]
        for joint_c in range(dh_pars.shape[0]):
            jpos.append([])
            transforms = tot_rt(dh_pars[joint_c, :, :])
            for chain_c, transform in enumerate(transforms):
                curr[chain_c] = np.matmul(curr[chain_c], transform)
                jpos[-1].append(np.dot(curr[chain_c], target))
        return np.array(jpos)[:, :, :-1], curr

    def add_volume(self,flist):
        '''Input: list of fingers joints
        '''
        gmm3d_means=[]
        for fing in flist:
            for joint1,joint2 in zip(fing[:-1],fing[1:]):
                gmm3d_means.append((joint2+joint1)/2)
        return gmm3d_means

            



hand = HandMetrics()
origin = np.array([[0], [0], [0], [1]])
plt.ion()
fig = plt.figure()
ax0 = fig.add_subplot(1, 1, 1, projection='3d')
#fig = plt.figure(figsize=plt.figaspect(1/2.))
#ax0 = fig.add_subplot(2,3,2,projection='3d')
#ax1 = fig.add_subplot(2,3,4, projection='3d')
#ax2 = fig.add_subplot(2,3,5, projection='3d')
#ax3 = fig.add_subplot(2,3,6, projection='3d')
try:
    while True:
        wpos, wtransform = hand.forward_kinematics(hand.w_dh_pars - hand.w_dh_rot_lims_minu +
                                                   np.random.rand(3, 4, 1)
                                                   * (hand.w_dh_rot_lims_minu +
                                                      hand.w_dh_rot_lims_plus), np.eye(4), origin)
        nfpos, nftransforms = hand.forward_kinematics(hand.nf_dh_pars - hand.nf_dh_rot_lims_minu +
                                                      np.random.rand(6, 4, 4)
                                                      * (hand.nf_dh_rot_lims_minu +
                                                         hand.nf_dh_rot_lims_plus), wtransform[0], origin)
        tfpos, tftransforms = hand.forward_kinematics(hand.tf_dh_pars - hand.tf_dh_rot_lims_minu +
                                                      np.random.rand(6, 4, 1)
                                                      * (hand.tf_dh_rot_lims_minu +
                                                         hand.tf_dh_rot_lims_plus), wtransform[0], origin)
        wpos = wpos[0, :, :, :][None, :, :, :]
        wpostiled = np.tile(wpos, (1, nfpos.shape[1], 1, 1))
        tottfpos = np.concatenate((wpos, tfpos), axis=0)
        totnfpos = np.concatenate((wpostiled, nfpos), axis=0)
        linehandles = linedraw(ax0, [tottfpos, totnfpos])
        ax0.set_xlim3d((-200, 200))
        ax0.set_ylim3d((-200, 200))
        ax0.set_zlim3d((-200, 200))
        ''' 
        linehandles=linedraw(ax1,[tottfpos,totnfpos])
        linedraw(ax2,[tottfpos,totnfpos])
        linedraw(ax3,[tottfpos,totnfpos])
        ax1.view_init(elev=0, azim=0) 
        ax2.view_init(elev=90, azim=0)
        ax3.view_init(elev=90,azim=90)
        ax1.set_ylim3d((-200,200))
        ax2.set_ylim3d((-200,200))
        ax3.set_ylim3d((-200,200))
        ax1.set_xlim3d((-200,200))
        ax2.set_xlim3d((-200,200))
        ax3.set_xlim3d((-200,200))
        ax1.set_zlim3d((-200,200))
        ax2.set_zlim3d((-200,200))
        ax3.set_zlim3d((-200,200))
        '''
        fig.canvas.draw()
        plt.pause(0.001)
        ax0.cla()
        '''
        ax1.cla()
        ax2.cla()
        ax3.cla()
        '''
except KeyboardInterrupt:
    print 'Exiting'


