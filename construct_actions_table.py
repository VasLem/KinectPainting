import os
from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import class_objects as co
import cv2
import action_recognition_alg as ara
from textwrap import wrap
def extract_valid_action_utterance(action, testing=False, *args, **kwargs):
        '''
        Visualizes action or a testing dataset using predefined locations in
        config.yaml and the method co.draw_oper.plot_utterances
        '''
        dataset_loc = '/media/vassilis/Thesis/Datasets/PersonalFarm/'
        results_loc = '/home/vassilis/Thesis/KinectPainting/Results/DataVisualization'
        ground_truth,breakpoints,labels = co.gd_oper.load_ground_truth(action, ret_labs=True,
                                         ret_breakpoints=True)
        images_base_loc = os.path.join(dataset_loc, 'actions',
                                      'sets' if not testing else 'whole_result')
        images_loc = os.path.join(images_base_loc, action.replace('_',' ').title())
        imgs, masks, sync, angles, centers, samples_indices = co.imfold_oper.load_frames_data(images_loc,masks_needed=True)

        masks_centers = []
        xdim = 0
        ydim = 0
        conts = []
        tmp = []
        for mask,img in zip(masks,imgs):
            conts = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
            conts_areas = [cv2.contourArea(cont) for cont in conts]
            tmp.append(np.sum(mask*img>0))
            if np.sum(mask*img>0) < 500:
                masks_centers.append(None)
            else:
                cont = conts[np.argmax(conts_areas)]
                x,y,w,h = cv2.boundingRect(cont)
                if w == 0 or h == 0:
                    masks_centers.append(None)
                else:
                    masks_centers.append([y+h/2,x+w/2])
                    xdim = max(w,xdim)
                    ydim = max(h,ydim)

        cropped_imgs = []
        for img, center in zip(imgs, masks_centers):
            if center is not None:
                cropped_img =img[max(0,center[0]-ydim/2)
                                        :min(img.shape[0],center[0]+ydim/2),
                                    max(0,center[1]-xdim/2)
                                        :min(img.shape[0],center[1]+xdim/2)]
                inp_img = np.zeros((ydim, xdim))
                inp_img[:cropped_img.shape[0],:cropped_img.shape[1]] = cropped_img
                cropped_imgs.append(inp_img)
            else:
                cropped_imgs.append(None)
        return cropped_imgs, sync, ground_truth, breakpoints, labels


def construct_table(action_type):

    fil = os.path.join(co.CONST['rosbag_location'],
                       'gestures_type.csv')
    if os.path.exists(fil):
        with open(fil, 'r') as inp:
            for line in inp:
                if line.split(':')[0].lower() == action_type.lower():
                    used_actions = line.split(
                        ':')[1].rstrip('\n').split(',')
    else:
        raise Exception()

    SHOWN_IMS = 10
    actions = [action for action in os.listdir(co.CONST['actions_path'])
               if action in used_actions]
    print actions
    images=[]
    for action in actions:
        print 'Processing', action
        whole = os.path.join(co.CONST['actions_path'],action)
        cnt = 0
        (frames, frames_sync,
                 ground_truth, breakpoints, labels) =\
        extract_valid_action_utterance(action.replace(' ','_').lower())
        for (start, end) in zip(breakpoints[action][0],
                                breakpoints[action][1]):
            if (start in frames_sync
                and end in frames_sync and
                end-start > SHOWN_IMS):

                rat_of_nans = sum([img is None for img
                                   in frames[frames_sync.index(start):
                                                   frames_sync.index(end)]]) / float(
                    end-start+1)
                if rat_of_nans < 0.1:
                    break
            cnt += 1
        masks = os.path.join(whole, co.CONST['hnd_mk_fold_name'], str(cnt))
        data = os.path.join(whole, co.CONST['mv_obj_fold_name'], str(cnt))

        start = breakpoints[action][0][cnt]
        end = breakpoints[action][1][cnt]
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
            img = np.pad(cv2.equalizeHist(img.astype(np.uint8)),[[0,0],[5,5]],
                            mode='constant', constant_values=155)
            imgset.append(img)
        images.append(imgset)
    images = np.array(images)
    images = list(images)
    im_indices = np.arange(SHOWN_IMS)
    left, width = .25, .5
    bottom, height = .25, .5
    right = left + width
    top = bottom + height

    gs = gridspec.GridSpec(len(images), 1+SHOWN_IMS)
    gs.update(wspace=0.0, hspace=0.0)
    fig = plt.figure(figsize=(1+SHOWN_IMS, len(images)))
    fig_axes = fig.add_subplot(gs[:,:],adjustable='box-forced')
    fig_axes.set_xticklabels([])
    fig_axes.set_yticklabels([])
    fig_axes.set_xticks([])
    fig_axes.set_yticks([])
    fig_axes.set_aspect('auto')
    fig.subplots_adjust(wspace=0, hspace=0)
    im_inds = np.arange(len(images)*(1+SHOWN_IMS)).reshape(
        len(images),1+SHOWN_IMS)[:,1:].ravel()
    txt_inds = np.arange(len(images)*(1+SHOWN_IMS)).reshape(
        len(images),1+SHOWN_IMS)[:,:1].ravel()

    axes = [fig.add_subplot(gs[i]) for i in range(len(images)*
                                                  (1+SHOWN_IMS))]
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
                im_count], aspect='auto',cmap='gray')
            axes[im_inds[ax_count]].set_xlim((0,max(im_shape)))
            axes[im_inds[ax_count]].set_ylim((0,max(im_shape)))
            ax_count += 1
    ax_count = 0
    info = np.array(actions)
    for im_count in range(len(images)):
        text = ('\n').join(wrap(info[im_count],10))
        axes[txt_inds[ax_count]].text(0.5*(left+right), 0.5*(bottom+top),
                                      text,
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=9)
        ax_count+=1
    cellText = [['Gesture']+[str(i) for i in range(SHOWN_IMS)]]
    col_table = fig_axes.table(cellText=cellText,
                               cellLoc='center',
                              loc='top')

    save_fold = os.path.join(co.CONST['results_fold'],
                             'Classification',
                             'Total')
    co.makedir(save_fold)
    plt.savefig(os.path.join(save_fold,action_type + 'actions_vocabulary.pdf'))

construct_table('dynamic')
construct_table('passive')
