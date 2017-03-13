import os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import class_objects as co
import cv2
import action_recognition_alg as ara
from textwrap import wrap
fil = os.path.join(co.CONST['rosbag_location'],
                   'gestures_type.csv')
passive_actions = None
if os.path.exists(fil):
    with open(fil, 'r') as inp:
        for line in inp:
            if line.split(':')[0] == 'Passive':
                passive_actions = line.split(
                    ':')[1].rstrip('\n').split(',')
else:
    raise Exception()
SHOWN_IMS = 10
actions = os.listdir(co.CONST['actions_path'])
ispassive = []
images=[]
for action in actions:
    ispassive.append(action in passive_actions)
    whole = os.path.join(co.CONST['actions_path'],action)
    masks = os.path.join(whole, co.CONST['hnd_mk_fold_name'], str(0))
    data = os.path.join(whole, co.CONST['mv_obj_fold_name'], str(0))
    angles = []
    with open(os.path.join(data, 'angles.txt'), 'r') as inp:
        angles += map(float, inp)
    centers = []
    with open(os.path.join(data, 'centers.txt'), 'r') as inp:
        for line in inp:
            center = [
                float(num) for num
                in line.split(' ')]
            centers += [center]
    fils = sorted(os.listdir(masks))
    inds = np.array([int(filter(str.isdigit,fil)) for fil in fils])
    imgset = []
    prev_size = 0
    for ind in (np.linspace(0,len(fils)-1,SHOWN_IMS)).astype(int):
        count = ind
        while True:
            mask = cv2.imread(os.path.join(masks, fils[ind]),0)>0
            if np.sum(mask) > 0.6 * (prev_size) or count == len(inds)-1:
                prev_size = np.sum(mask)
                break
            else:
                count += 1
        img = (cv2.imread(os.path.join(data,fils[count]),-1)*
                mask)
        processed_img = co.pol_oper.derotate(
            img,
            angles[count], centers[count])
        img,_,_ = ara.prepare_im(processed_img,square=True)
        imgset.append(img)
    images.append(imgset)
images = np.array(images)
images = list(images)
im_indices = np.arange(SHOWN_IMS)
left, width = .25, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

gs = gridspec.GridSpec(len(images), 2+SHOWN_IMS)
gs.update(wspace=0.0, hspace=0.0)
fig = plt.figure(figsize=(2+SHOWN_IMS, len(images)))
fig_axes = fig.add_subplot(gs[:,:],adjustable='box-forced')
fig_axes.set_xticklabels([])
fig_axes.set_yticklabels([])
fig_axes.set_xticks([])
fig_axes.set_yticks([])
fig_axes.set_aspect('auto')
fig.subplots_adjust(wspace=0, hspace=0)
im_inds = np.arange(len(images)*(2+SHOWN_IMS)).reshape(
    len(images),2+SHOWN_IMS)[:,2:].ravel()
txt_inds = np.arange(len(images)*(2+SHOWN_IMS)).reshape(
    len(images),2+SHOWN_IMS)[:,:2].ravel()

axes = [fig.add_subplot(gs[i]) for i in range(len(images)*
                                              (2+SHOWN_IMS))]
im_axes = list(np.array(axes)[im_inds])
for axis in axes:
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.set_xticks([])
    axis.set_yticks([])
ax_count = 0
for im_set_count in range(len(images)):
    for im_count in list(im_indices):
        im_shape = list(images[im_set_count])[im_count].shape
        axes[im_inds[ax_count]].imshow(list(images[im_set_count])[
            im_count], aspect='auto')
        axes[im_inds[ax_count]].set_xlim((0,max(im_shape)))
        axes[im_inds[ax_count]].set_ylim((0,max(im_shape)))
        ax_count += 1
ax_count = 0
info = np.vstack((np.array(actions), np.array(ispassive))).T
for im_count in range(len(images)):
    for ind in range(2):
        text = ('\n').join(wrap(info[im_count][ind],10))
        axes[txt_inds[ax_count]].text(0.5*(left+right), 0.5*(bottom+top),
                                      text,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=9)
        ax_count+=1
cellText = [['Action','Passive']+[str(i) for i in range(SHOWN_IMS)]]
col_table = fig_axes.table(cellText=cellText,
                           cellLoc='center',
                          loc='top')

save_fold = os.path.join(co.CONST['results_fold'],
                         'Classification',
                         'Total')
co.makedir(save_fold)
plt.savefig(os.path.join(save_fold,'actions_vocabulary.pdf'))
plt.show()
