import class_objects as co
from matplotlib import pyplot as plt
from class_objects import CONST


def montage_raw_frames(action_name=None,
                         data=None,
                         masks_needed=True,
                         mv_obj_fold_name=None,
                         hnd_mk_fold_name=None,
                         derot_centers=None,
                         derot_angles=None,
                         n_vis_frames=9,
                         output='frames_sample.pdf'):
        (imgs, masks, sync, angles,
         centers,
         samples_inds) = co.imfold_oper.load_frames_data(
             data, mv_obj_fold_name,
             hnd_mk_fold_name, masks_needed,
             derot_centers, derot_angles)
        montage = co.draw_oper.create_montage(imgs[:],
                                              max_ims=n_vis_frames,
                                              draw_num=False)
        fig = plt.figure()
        tmp_axes = fig.add_subplot(111)
        tmp_axes.imshow(montage[:, :, :-1])
        plt.axis('off')
        fig.savefig('frames_sample.pdf',
                    bbox_inches='tight')


def visualize_features(descriptor, imgs):
    for img_count, img in enumerate(imgs):
    descriptor.visualize()
    descriptor.draw()

