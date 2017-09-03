import os
import errno
import numpy as np
import cv2
from math import pi
import time
from cv_bridge import CvBridge, CvBridgeError
import yaml
from __init__ import *
from matplotlib import pyplot as plt
import logging
LOG = logging.getLogger('__name__')
CH = logging.StreamHandler()
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.addHandler(CH)
LOG.setLevel('INFO')


def timeit(method):

    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        print method.__name__, (te - ts) * 1000, 'ms'
        return result

    return timed


def find_nonzero(arr):
    return np.fliplr(cv2.findNonZero(arr).squeeze())


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def tag_im(img, msg, loc='top left', in_place=True,
           font=cv2.FONT_HERSHEY_SIMPLEX,
           fontscale=0.5,
           vspace=0.5,
           thickness=1,
           centered=False,
           color=(0, 0, 255)):
    locs = loc.split(' ')
    if len(locs) == 1:
        locs.append('mid')
    t_x_shapes = []
    t_y_shapes = []
    t_lens = []
    lines = msg.split('\n')
    for line in lines:
        text_shape, text_len = cv2.getTextSize(line, font,
                                               fontscale, thickness)
        t_x_shapes.append(text_shape[0])
        t_y_shapes.append(text_shape[1])
        t_lens.append(text_len)
    line_height = max(t_y_shapes)
    text_shape = (max(t_x_shapes), int((line_height) *
                                       (1 + vspace) *
                                       len(lines)))
    vpos = {'top': 0,
            'mid': img.shape[0] / 2 - text_shape[1] / 2,
            'bot': img.shape[0] - text_shape[1] - 10}
    hpos = {'left': text_shape[0] / 2 + 5,
            'mid': img.shape[1] / 2,
            'right': img.shape[1] - text_shape[0] / 2}
    if in_place:
        cop = img
    else:
        cop = img.copy()

    for count, line in enumerate(lines):
        xpos = hpos[locs[1]]
        if not centered:
            xpos -= text_shape[0] / 2
        else:
            xpos -= t_x_shapes[count] / 2
        ypos = (vpos[locs[0]] +
                int(line_height * (1 + vspace)
                    * (1 + count)))
        cv2.putText(cop, line, (xpos, ypos),
                    font, fontscale, color, thickness)


class CircularOperations(object):
    '''
    Holds operations on positions in circular lists
    '''

    def diff(self, pos1, pos2, leng, no_intersections=False):
        '''
        Returns the smallest distances between
        two positions vectors inside a circular 1d list
        of length leng
        '''
        if no_intersections and pos2.size == 2:
            # some strolls are forbidden
            argmindiffs = np.zeros_like(pos1)
            mindiffs = np.zeros_like(pos1)
            pos_i = np.min(pos2)
            pos_j = np.max(pos2)
            check = pos1 >= pos_j
            tmp = np.concatenate(
                ((leng - pos1[check] + pos_i)[:, None],
                 (-pos_j + pos1[check])[:, None]), axis=1)

            argmindiffs[check > 0] = np.argmin(tmp, axis=1)
            mindiffs[check > 0] = np.min(tmp, axis=1)
            check = pos1 <= pos_i
            tmp = np.concatenate(
                ((pos_i - pos1[check])[:, None],
                 (leng - pos_j + pos1[check])[:, None]), axis=1)
            argmindiffs[check > 0] = np.argmin(tmp, axis=1)
            check = (pos1 <= pos_j) * (pos1 >= pos_i)
            tmp = np.concatenate(
                ((-pos_i + pos1[check])[:, None],
                 (pos_j - pos1[check])[:, None]), axis=1)
            argmindiffs[check > 0] = np.argmin(tmp, axis=1)
            return argmindiffs[mindiffs.argmin()]
        pos1 = pos1[:, None]
        pos2 = pos2[None, :]
        return np.min(np.concatenate((
            np.abs(pos1 - pos2)[:, None],
            np.abs((leng - pos2) + pos1)[:, None],
            np.abs((leng - pos1) + pos2)[:, None]),
            axis=2), axis=2)

    def find_min_dist_direction(self, pos1, pos2, leng, filter_len=None):
        '''
        find pos1 to pos2 minimum distance direction
        '''
        to_count = -1
        if filter_len is not None:
            mask = np.zeros(leng)
            mask[min(pos1, pos2):max(pos1, pos2) + 1] = 1
            if np.sum(mask * leng) > 0:
                to_count = 0
            else:
                to_count = 1
        dists = np.array([np.abs(pos1 - pos2),
                          np.abs(leng - pos2 + pos1),
                          np.abs(leng - pos1 + pos2)])
        if to_count == 1:
            res = np.sign(pos2 - pos1)
            return res, dists[0]
        if to_count == 0:
            dists[0] = 1000000000
        choose = np.argmin(dists)
        if choose == 0:
            res = np.sign(pos2 - pos1)
            if res == 0:
                res = 1
        elif choose == 1:
            res = -1
        else:
            res = 1
        return res, np.min(dists)

    def filter(self, pos1, pos2, length, direction):
        '''
        output: filter_mask of size length with filter_mask[pos1--direction-->pos2]=1, elsewhere 0
        '''
        filter_mask = np.zeros(length)
        if pos2 >= pos1 and direction == 1:
            filter_mask[pos1: pos2 + 1] = 1
        elif pos2 >= pos1 and direction == -1:
            filter_mask[: pos1 + 1] = 1
            filter_mask[pos2:] = 1
        elif pos1 > pos2 and direction == 1:
            filter_mask[: pos2 + 1] = 1
            filter_mask[pos1:] = 1
        elif pos1 > pos2 and direction == -1:
            filter_mask[pos2: pos1 + 1] = 1
        return filter_mask

    def find_single_direction_dist(self, pos1, pos2, length, direction):
        if pos2 >= pos1 and direction == 1:
            return pos2 - pos1
        elif pos2 >= pos1 and direction == -1:
            return length - pos2 + pos1
        elif pos1 > pos2 and direction == 1:
            return length - pos1 + pos2
        elif pos1 > pos2 and direction == -1:
            return pos1 - pos2


class ConvexityDefect(object):
    '''Convexity_Defects holder'''

    def __init__(self):
        self.hand = None


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


class CountHandHitMisses(object):
    '''
    class to hold hand hit-misses statistics
    '''

    def __init__(self):
        self.no_obj = 0
        self.found = 0
        self.no_entry = 0
        self.in_im_corn = 0
        self.rchd_mlims = 0
        self.no_cocirc = 0
        self.no_abnorm = 0
        self.rchd_abnorm = 0

    def reset(self):
        self.no_obj = 0
        self.found = 0
        self.no_entry = 0
        self.in_im_corn = 0
        self.rchd_mlims = 0
        self.no_cocirc = 0
        self.no_abnorm = 0
        self.rchd_abnorm = 0

    def print_stats(self):
        members = [attr for attr in dir(self) if not
                   callable(getattr(self, attr))
                   and not attr.startswith("__")]
        for member in members:
            print member, ':', getattr(self, member)


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
        self.depth_raw = None


class DrawingOperations(object):
    '''
    Methods for advanced plotting using matplotlib
    '''
    


    def plot_utterances(self, breakpoints,
                        labels,
                        ground_truth=None,
                        frames_sync=None,
                        frames=None,
                        real_values = None,
                        show_legend=False,
                        leg_labels=[],
                        show_breaks=True, show_occ_tab=True,
                        show_zoomed_occ=True, show_fig_title=True,
                        show_im_examples=True, categories_to_zoom = None,
                        fig_width=12, examples_height=2, zoomed_occ_size=4,
                        break_height = 0,
                                        examples_pad_size=10,show_res=True,
                        examples_num=None, min_im_zoom_num = 5,
                        max_im_zoom_num = 40,
                        title=None, dataset_name='',
                        show_warnings=True, *args, **kwargs):
        '''
        An advanced plotter that shows utterances, dependent on time.
        <breakpoints> is a dictionary,with keys the names of the classes to be recognized
        and each entry holding the starting and ending time points (2 lists)
        for each occurence, based on the indices in ground truth.
        <frames> is the list of frames of the whole video sequence, which is
        necessary in case <show_zoomed_occ> or <show_im_examples> is True.
        <ground_truth> is the corresponding ground truth, a numpy array, which
        is needed in case <show_im_examples> is True.
        <frames_sync> is a vector holding the ground_truth index of each frame, which is
        necessary in case <show_zoomed_occ> or <show_im_examples> is True.
        <labels> are the names of the classes inside ground truth.
        Flags starting by 'show' are there, so that to determine which element
        to plot or not. <categories_to_zoom> is a string or a list of strings with the names
        of the classes, from which a sample is going to be plotted in a montage.
        '''
        if kwargs and show_warnings:
            LOG.warning('Given Invalid Arguments: ' + str(kwargs.keys()))
        if categories_to_zoom is None:
            categories_to_zoom = labels
        if not isinstance(categories_to_zoom, list):
            categories_to_zoom = [categories_to_zoom]
        for ind in range(len(categories_to_zoom)):
            categories_to_zoom[ind] = categories_to_zoom[ind].lower()
        lower_labels = []
        for label in labels:
            lower_labels.append(label.lower())
        gs_width = fig_width
        gs_examples = examples_height
        zoom_size = zoomed_occ_size
        pad_size = examples_pad_size
        from matplotlib.cm import get_cmap
        from matplotlib.patches import ArrowStyle, ConnectionPatch
        from matplotlib import gridspec

        # Initialize constants
        
        cmap = get_cmap('Spectral')
        arr_cmap = get_cmap('tab20b')
        tab_x_size = zoom_size * show_occ_tab
        if show_fig_title:
            gs_title = 1
        else:
            gs_title = 0
        break_width = (gs_width-tab_x_size) * show_breaks
        if not break_height:
            break_height = break_width
        if show_occ_tab:
            if show_breaks:
                tab_y_size = break_height
            else:
                tab_y_size = 2 * zoom_size
        else:
            tab_y_size = 0
        if examples_num is None and show_im_examples:
            if show_breaks and show_im_examples:
                examples_num = break_width
            else:
                examples_num = 8
            LOG.warning('Specifying explicitely examples_num to %d',
                        examples_num)
        else:
            examples_num = 0

        gs_examples = gs_examples * show_im_examples
        
        ex_categories_to_zoom = [act for act in breakpoints if act.lower() in categories_to_zoom]
        zoom_cat_num = len(ex_categories_to_zoom)
        zooms_per_row = gs_width / zoom_size
        zooms_columns = np.ceil(zoom_cat_num / float(zooms_per_row)) * show_zoomed_occ
        gs_height = int(gs_title + max(tab_y_size,break_height) + gs_examples + zooms_columns*zoom_size)

        if not gs_height:
            LOG.warning('All show flags are set to false, returning None')
            return None
       
        # Initialize GridSpec and figure objects
        f = plt.figure(figsize=(gs_width,gs_height))
        gs = gridspec.GridSpec(gs_height,gs_width)
        gs.update(wspace=0.025, hspace=0.05)
        if show_fig_title:
            ax_title = plt.subplot(gs[0,:])
            if title is None:
                title = 'Utterances in dataset \"' + dataset_name +'\"'
            ax_title.set_title(title,
                                    fontsize=20)
            ax_title.axis('off')

        if show_breaks:
            ax_plot = plt.subplot(gs[gs_title:gs_title+break_height,
                                      :break_width])
            f.add_subplot(ax_plot)
            bbox = ax_plot.get_window_extent().transformed(f.dpi_scale_trans.inverted())
            width, height = bbox.width, bbox.height
            width *= f.dpi
            height *= f.dpi
            norm_lw = min(width, height)/200
            
        if show_im_examples:
            ax_ex = plt.subplot(gs[gs_title+break_height:
                                   gs_title+break_height+gs_examples,:-tab_x_size])
            f.add_subplot(ax_ex)
        else:
            ax_ex = None
        if show_occ_tab:
            ax_table = plt.subplot(gs[gs_title:tab_y_size+gs_title,-tab_x_size:])
            f.add_subplot(ax_table)
        else:
            ax_table = None
        if show_zoomed_occ:
            axzooms=[]
            for count,lab in enumerate(ex_categories_to_zoom):
                zoom_x = int(count  % np.floor(((gs_width)/zoom_size)))
                zoom_y = int(count  / np.floor(((gs_width)/zoom_size)))
                axzooms.append(plt.subplot(gs[gs_title+break_height + gs_examples +
                                              zoom_y * zoom_size:
                                              gs_title+break_height + gs_examples +
                                             (zoom_y + 1) * zoom_size,
                                              zoom_x * zoom_size:
                                              (zoom_x + 1) * zoom_size
                                              ]))
                axzooms[-1].axis('off')


        if show_im_examples:
            if frames is None:
                raise Exception('frames must not be None.\n'
                                + self.plot_utterances.__doc__)

            selected_imgs = []
            selected_inds = []
            selected_acts = []
            for count in np.arange(examples_num)/float(examples_num-1)*(len(ground_truth)-1):
                count2 = int(count)
                sgn = 1
                cnt = 1
                while True:
                    try:
                        if (count2 in frames_sync and 
                            frames[frames_sync.index(count2)] is not None and 
                            0 not in np.shape(frames[frames_sync.index(count2)]) and
                            np.isfinite(ground_truth[count2])):
                            break
                    except:
                        pass
                    count2 = int(count) + sgn*cnt
                    sgn = -sgn
                    if sgn == 1:
                        cnt+=1
                selected_imgs.append(cv2.equalizeHist(frames[frames_sync.index(count2)].astype(np.uint8)))
                selected_inds.append(count2)
                selected_acts.append(ground_truth[count2]+1)


            selected_imgs = np.hstack(
                [np.pad(array=img,mode='constant',pad_width=((0,0),(0,pad_size)), constant_values=255)
                 for img in selected_imgs]).astype(np.uint8)
            selected_imgs = selected_imgs[:,:-pad_size]

            rat = (ground_truth.size-1)/float(selected_imgs.shape[1]-1)
            ax_ex.imshow(selected_imgs,interpolation="nearest",cmap='gray',zorder=1)
            ax_ex.set_title('Example Frames')
            ax_ex.set_aspect('auto')
            if not show_breaks:
                ax_ex.set_xlabel('Frames')
                ax_ex.set_xticklabels((ax_ex.get_xticks() * rat).astype(int))
            else:
                ax_ex.xaxis.set_visible(0)
            ax_ex.yaxis.set_visible(0)
            

        else:
            rat = 1


        if show_breaks:

            max_plotpoints_num = 0
            for act in breakpoints:
                max_plotpoints_num = max(max_plotpoints_num,
                                          len(breakpoints[act][0]))
            c_num = max_plotpoints_num

            for act_cnt,act in enumerate(breakpoints):
                drawn = 0
                for cnt,(start, end) in enumerate(zip(breakpoints[act][0],
                                      breakpoints[act][1])):
                    gest_dur = np.arange(int(start/rat),int(end/rat))
                    ax_plot.plot(gest_dur, np.ones(gest_dur.size)*(
                        lower_labels.index(act.lower())+1),

                                    color=cmap(cnt/float(c_num)),linewidth=norm_lw,
                                 solid_capstyle="butt",zorder=0)

            if real_values is not None:
                ax_plot.plot(real_values, linewidth=norm_lw/2, color='black',
                             label='Predicted Values',zorder=1)
                if show_legend:
                    ax_plot.legend()
            ax_plot.set_title('Gestures Utterances\nAlong Time')
            ax_plot.set_aspect('auto')
            ax_plot.set_ylim(0,len(labels)+1)
            ax_plot.set_yticks(np.arange(len(labels)+1))
            ax_plot.set_yticklabels(['']+[label.title() for label in labels]+[''])
            ax_plot.set_ylabel('Gestures')
            ax_plot.set_xlabel('Frames')
            ax_plot.set_xticklabels((ax_plot.get_xticks() * rat).astype(int))
        if show_zoomed_occ:
            if frames is None:
                raise Exception('frames must not be None.\n'
                                + self.plot_utterances.__doc__)
            for img in frames:
                if img is not None and 0 not in img.shape:
                    imi = img.shape[0]
                    imj = img.shape[1]
                    break
            occ_montages = []
            break_spans = []
            break_labels = []
            for act_cnt,act in enumerate(breakpoints):
                drawn = 0
                for cnt,(start, end) in enumerate(zip(breakpoints[act][0],
                                      breakpoints[act][1])):
                    if drawn:
                        break
                    if (act.lower() in categories_to_zoom
                        and start in frames_sync and end in frames_sync and end-start > min_im_zoom_num):
                        rat_of_nans = sum([img is None for img
                                           in frames[frames_sync.index(start):
                                                           frames_sync.index(end)]]) / float(
                            end-start+1)
                        if rat_of_nans < 0.2:
                            occ_montage = self.create_montage(frames[
                                frames_sync.index(start):
                                frames_sync.index(end)],
                                max_ims=max_im_zoom_num,
                                im_shape=(imi, imj))
                            occ_montages.append(occ_montage)
                            break_spans.append([int(start/rat),
                                          int(end/rat)])
                            break_labels.append(lower_labels.index(act.lower())+1)
                            drawn = 1
                        else:
                            continue
            for axzoom,occ_montage,break_span, break_label in zip(
                axzooms,occ_montages,break_spans, break_labels) :
                occ_montage = occ_montage/255.0
                mont = axzoom.imshow(occ_montage,zorder=1)
                axzoom.set_title(labels[break_label-1].title())

        if show_breaks and show_zoomed_occ:
            for cnt,(axzoom,occ_montage,break_span, break_label) in enumerate(zip(
                axzooms,occ_montages,break_spans, break_labels)) : 
                con1 = ConnectionPatch(xyA=(break_span[0],break_label), xyB=[0,0], coordsA="data", coordsB="data",
                                  axesA=ax_plot, axesB=axzoom, color=arr_cmap(cnt), linewidth=1, linestyle='dashdot',
                                  alpha=1,zorder=25)
                con2 =  ConnectionPatch(xyA=(break_span[1],break_label), xyB=[occ_montage.shape[1],0], coordsA="data", coordsB="data",
                                  axesA=ax_plot, axesB=axzoom, color=arr_cmap(cnt), linewidth=1, linestyle='dashdot',
                                   alpha=1,zorder=25)

                ax_plot.add_patch(con1)
                ax_plot.add_patch(con2)

        if show_im_examples and show_breaks:
            for count, ind in enumerate(selected_inds):
                xyB = [(0.5+count)*imj
                       +count*pad_size, 0]
                xyA = [ind/float(ground_truth.size) * selected_imgs.shape[1],
                       selected_acts[count]]


                con = ConnectionPatch(
                    xyA=xyA,
                    xyB =xyB,
                    coordsA="data", coordsB="data",
                    axesA=ax_plot, axesB=ax_ex,
                    color="red",alpha=0.5,arrowStyle='-|>',
                    linewidth=0.5, zorder=25)
                con.set_arrowstyle(ArrowStyle('-|>',head_length=.4,
                                              head_width=0.2))

                ax_plot.add_artist(con)



        if show_occ_tab:
            columns = ('Gestures','#Occurences')
            cell_text = []
            for key in breakpoints:
                cell_text.append([key,len(breakpoints[key][0])])

            ax_table.table(cellText=cell_text,
                           colLabels=columns,
                           loc='center',
                           cellLoc='center',
                           )
            ax_table.axis('off')
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            # This raises warnings since tight layout cannot
            # handle gridspec automatically. We are going to
            # do that manually so we can filter the warning.
            gs.tight_layout(f)

        if show_res:
            plt.show()
        return f

    def save_pure_image(self, img, path, dpi=1200):
        fig = plt.figure(frameon=False)
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.imshow(img)
        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(path, bbox_inches=extent, pad_inches=0, dpi=dpi,
                    transparent=True)
        plt.close(fig)

    def create_montage(self, imgs, draw_num=True, num_rat=2.5,
                       num_col=(255, 0, 0, 125),
                       num_font_family='Arial', max_ims=None,
                       im_shape=None,
                       *args,
                       **kwargs):
        '''
        draws list of images into a montage. If <draw_num> then the
        corresponding index of each image is drawn above it, with
        font size <num_rat> times smaller than height and color <num_col>,
        while using font family <num_font_family>. Returns montage or None
        '''
        if max_ims is not None and len(imgs) > max_ims:
            inds = np.linspace(0, len(imgs) - 1,max_ims).astype(
                int)
        else:
            inds = np.arange(len(imgs))
        not_nan_imgs = sum([imgs[cnt] is not None for cnt in inds])
        num_im_y = int(np.ceil(np.sqrt(not_nan_imgs)))
        num_im_x = int(np.ceil(not_nan_imgs / float(num_im_y)))
        if im_shape is not None:
            imi, imj = im_shape
        else:
            imi = None
            for img in imgs:
                if img is not None and 0 not in img.shape:
                    imi = img.shape[0]
                    imj = img.shape[1]
                    break

            if imi is None:
                return None

        montage_shape = (num_im_x * imi,
                         num_im_y * imj, 4)
        montage = np.zeros(montage_shape)
        montage[:, :, 3] = 255
        im_cnt = 0

        for ind_cnt,cnt in enumerate(inds):
            im_cnt = cnt
            try:
                while imgs[im_cnt] is None:
                    im_cnt += 1
            except BaseException:
                break
            i_rat = ind_cnt / (num_im_y)
            j_rat = ind_cnt % (num_im_y)
            if len(imgs[im_cnt].shape) <= 2:
                imgs[im_cnt] = cv2.equalizeHist(
                    imgs[im_cnt].astype(np.uint8))
            else:
                imgs[im_cnt] = imgs[im_cnt].astype(np.uint8)
            if draw_num:
                imgs[im_cnt] = self.watermark_image_with_text(
                    imgs[im_cnt], str(im_cnt), rat=num_rat, color=num_col,
                    fontfamily=num_font_family)
            else:
                imgs[im_cnt] = self.convert_to_rgba(imgs[im_cnt])
            montage[i_rat * imi:(i_rat + 1) * imi,
                    j_rat * imj:(j_rat + 1) * imj, :] = imgs[im_cnt]

            im_cnt += 1
        return montage

    def convert_to_rgba(self, img, alpha=255):
        '''
        converts grayscale or rgb to rgba image. Alpha channel has <alpha>=255 value
        '''
        img = self.convert_to_rgb(img)
        if img.shape[2] == 3:
            img = np.concatenate([img, np.uint8(alpha * np.ones((img.shape[0], img.shape[1])))[..., None]],
                                 axis=2)
        return img

    def convert_to_rgb(self, img):
        if len(img.shape) == 2:
            img = np.tile(img[:, :, None], (1, 1, 3))
        elif img.shape[2] == 1:
            img = np.tile(img, (1,1,3))
        return img

    def watermark_image_with_text(self, img, text, rat=2, color=(
            255, 0, 0, 125), fontfamily='Arial', *args, **kwargs):
        '''
        Places watermark above image
        '''
        from PIL import Image, ImageDraw, ImageFont
        img = self.convert_to_rgba(img)
        image = Image.fromarray(img.astype(np.uint8)).convert("RGBA")
        imageWatermark = Image.new('RGBA', image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(imageWatermark)

        width, height = image.size
        frat = int(height / rat)
        margin = 10
        font = ImageFont.truetype(fontfamily, 1)
        textWidth, textHeight = draw.textsize(text, font)
        font = ImageFont.truetype(fontfamily, textHeight * frat)
        textWidth, textHeight = draw.textsize(text, font)
        x = width / 2 - textWidth / 2
        y = height / 2 - textHeight

        draw.text((x, y), text, color, font)

        return np.array(Image.alpha_composite(image, imageWatermark))

    def draw_nested(self, nested_object, parent=None):
        '''
        Creates tree from a nested object. Nested objects can be
        tuples, dicts and lists. Dicts and tuples are drawn, lists are
        used for making the tree. tuple[0] is used as the name of the node,
        tuple[1] as the next structure to draw. dict[key] is used as the next
        structure to draw, key is used as the name of the node.
        If tuple[1] or dict[key] are None, they are not drawn.
        '''
        import pydot
        import ast

        def add2graph(graph, parent=None, struct=None):
            try:
                struct = ast.literal_eval(str(struct))
            except BaseException:
                edge = pydot.Edge(parent, str(struct))
                graph.add_edge(edge)
                return
            if (not isinstance(struct, list) and
                not isinstance(struct, tuple) and
                    not isinstance(struct, dict)):
                edge = pydot.Edge(parent, str(struct))
                graph.add_edge(edge)
                return
            for item in struct:
                if isinstance(struct, dict):
                    sub_categ = struct[item]
                else:
                    sub_categ = item
                if (not isinstance(sub_categ, tuple)
                        and not isinstance(sub_categ, dict)):
                    add2graph(graph, parent, sub_categ)
                else:
                    if isinstance(sub_categ, dict):
                        if len(sub_categ) > 1:
                            add2graph(graph, parent,
                                      sub_categ)
                            continue
                        else:
                            node_name = sub_categ.keys[0]
                            node_val = sub_categ[sub_categ.keys[0]]
                    else:
                        node_name = sub_categ[0]
                        node_val = sub_categ[1]
                    if parent is not None:
                        if node_val is not None:
                            edge = pydot.Edge(parent, node_name)
                            graph.add_edge(edge)
                        else:
                            continue
                    try:
                        add2graph(graph, node_name, node_val)
                    except BaseException:
                        print node_name
                        raise
        graph = pydot.Dot(graph_type='graph')
        add2graph(graph, parent, nested_object)
        return graph


class DictionaryOperations(object):
    def create_sorted_dict_view(self, x):
        import operator
        return sorted(x.items(), key=operator.itemgetter(0))

    def join_list_of_dicts(self, L):
        return {k: v for d in L for k, v in d.items()}

    def dict_from_tuplelist(self, x):
        return dict(x)

    def lookup(self, dic, key, *keys):
        if keys:
            return self.lookup(dic.get(key, {}), *keys)
        return dic.get(key)


class TypeConversions(object):

    def isfloat(self, value):
        try:
            float(value)
            return True
        except ValueError:
            return False


class Edges(object):

    def __init__(self):
        self.calib_edges = None
        self.calib_frame = None
        self.exists_lim_calib_image = False
        self.edges_positions_indices = None
        self.edges_positions = None
        self.nonconvex_edges_lims = None
        self.exist = False

    def construct_calib_edges(self, im_set=None, convex=0,
                              frame_path=None, edges_path=None,
                              whole_im=False,
                              img=None,
                              write=False):
        if not whole_im:
            im_set = np.array(im_set)
            tmp = np.zeros(im_set.shape[1:], dtype=np.uint8)
            tmp[np.sum(im_set, axis=0) > 0] = 255
        else:
            if img is None:
                raise Exception('img argument must be given')
            tmp = np.ones(img.shape, dtype=np.uint8)
        _, frame_candidates, _ = cv2.findContours(
            tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        frame_index = np.argmax(
            np.array([cv2.arcLength(contour, 1) for contour in
                      frame_candidates]))
        self.calib_edges = np.zeros(
            tmp.shape, dtype=np.uint8)
        self.calib_frame = np.zeros(
            tmp.shape, dtype=np.uint8)
        cv2.drawContours(
            self.calib_edges, frame_candidates,
            frame_index, 255, 1)
        cv2.drawContours(
            self.calib_frame, frame_candidates,
            frame_index, 255, -1)
        if edges_path is None:
            edges_path = CONST['cal_edges_path']
        if frame_path is None:
            frame_path = CONST['cal_edges_path']
        if write:
            cv2.imwrite(CONST['cal_edges_path'], self.calib_edges)
            cv2.imwrite(CONST['cal_frame_path'], self.calib_frame)

    def load_calib_data(self, convex=0, edges_path=None, frame_path=None,
                        whole_im=False, img=None):
        if whole_im:
            if img is None:
                raise Exception('img argument must be given')
            self.construct_calib_edges(
                img=img, whole_im=True, write=False)
        else:
            if (frame_path is None) ^ (edges_path is None):
                if frame_path is None:
                    LOG.error('Missing frame_path input, but edges_path is' +
                              ' given')
                else:
                    LOG.error('Missing edges_path input, but frame_path is' +
                              ' given')
            if edges_path is None:
                edges_path = CONST['cal_edges_path']
                frame_path = CONST['cal_frame_path']
            if not os.path.isfile(edges_path):
                self.exists_lim_calib_image = 0
            else:
                self.exists_lim_calib_image = 1
                self.calib_frame = cv2.imread(frame_path, 0)
                self.calib_edges = cv2.imread(frame_path, 0)
                self.calib_frame[
                    self.calib_frame < 0.9 * np.max(self.calib_frame)] = 0
                self.calib_edges[
                    self.calib_edges < 0.9 * np.max(self.calib_edges)] = 0
        if not convex:
            self.find_non_convex_edges_lims()
        else:
            self.find_convex_edges_lims()
        self.exist = True
        return edges_path, frame_path

    def find_non_convex_edges_lims(self, edge_tolerance=10):
        '''
        Find non convex symmetrical edges minimum orthogonal lims with some tolerance
        Inputs: positions,edges mask[,edges tolerance=10]
        '''
        self.edges_positions_indices = np.nonzero(cv2.dilate(
            self.calib_edges, np.ones((3, 3), np.uint8), cv2.CV_8U) > 0)
        self.edges_positions = np.transpose(
            np.array(self.edges_positions_indices))
        lr_positions = self.edges_positions[
            np.abs(self.edges_positions[:, 0] - self.calib_edges.shape[0] / 2.0) < 1, :]
        tb_positions = self.edges_positions[
            np.abs(self.edges_positions[:, 1] - self.calib_edges.shape[1] / 2.0) < 1, :]
        self.nonconvex_edges_lims = np.array(
            [np.min(lr_positions[:, 1]) + edge_tolerance,
             np.min(tb_positions[:, 0]) + edge_tolerance,
             np.max(lr_positions[:, 1]) - edge_tolerance,
             np.max(tb_positions[:, 0]) - edge_tolerance])

    def find_convex_edges_lims(self, positions=None):
        '''
        Find convex edges minimum orthogonal lims
        '''
        def calculate_cart_dists(cart_points, cart_point=[]):
            '''
            Input either numpy array either 2*2 list
            Second optional argument is a point
            '''
            if cart_point == []:

                try:
                    return np.sqrt(
                        (cart_points[1:, 0] - cart_points[: -1, 0]) *
                        (cart_points[1:, 0] - cart_points[: -1, 0]) +
                        (cart_points[1:, 1] - cart_points[: -1, 1]) *
                        (cart_points[1:, 1] - cart_points[: -1, 1]))
                except (TypeError, AttributeError):
                    return np.sqrt((cart_points[0][0] - cart_points[1][0])**2 +
                                   (cart_points[0][1] - cart_points[1][1])**2)

            else:
                return np.sqrt(
                    (cart_points[:, 0] - cart_point[0]) *
                    (cart_points[:, 0] - cart_point[0]) +
                    (cart_points[:, 1] - cart_point[1]) *
                    (cart_points[:, 1] - cart_point[1]))
        calib_positions = positions[self.calib_edges > 0, :]
        calib_dists = calculate_cart_dists(
            calib_positions,
            np.array([0, 0]))

        upper_left = calib_positions[np.argmin(calib_dists), :]
        calib_dists2 = calculate_cart_dists(
            calib_positions,
            np.array([self.calib_edges.shape[0],
                      self.calib_edges.shape[1]]))
        lower_right = calib_positions[np.argmin(calib_dists2), :]
        # Needs filling
        self.convex_edges_lims = []


class ExistenceProbability(object):
    '''
    Class to find activated cells
    '''

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
        '''
        calculate activated cells
        '''
        if self.init_val == 0:
            self.wearing_mat = np.zeros(segclass.total_obj_num)
            self.init_val = 1
            # self.distance_mask = np.pad(
            # np.ones(tuple(np.array(data.depth_im.shape) - 2)), 1, 'constant')
            sums = np.sum(edges.calib_frame[
                          :, 1: edges.calib_frame.shape[1] / 2], axis=0) > 0
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
            self.wearing_mat[: segclass.nz_objects.count] > 0)[0]

        self.wearing_mat -= 1
        '''
        im_res = np.zeros_like(data.depth_im, dtype=np.uint8)
        for val in self.can_exist:
            im_res[segclass.objects.image == val] = 255
        im_results.images.append(im_res)
        '''


class FileOperations(object):
    '''
    Class that holds all loading and saving operations needed for a file
    database to work
    '''

    def save_obj(self, obj, filename):
        import dill
        with open(filename, 'w') as out:
            dill.dump(obj, out)

    def load_obj(self, filename):
        import dill
        try:
            with open(filename, 'r') as inp:
                obj = dill.load(inp)
        except BaseException:
            obj = None
        return obj

    def save_using_ujson(self, obj, filename):
        import ujson as json
        with open(filename, 'w') as out:
            json.dump(obj, out)

    def fix_keys(self, name):
        '''
        for every json file inside name, changes dictionaries with more than
        one element to tuples list, sorted by keys
        '''
        import os
        for path in os.listdir(name):
            if os.path.isdir(os.path.join(name, path)):
                self.fix_keys(os.path.join(name, path))
            elif path.endswith('.json'):
                catalog = self.load_using_ujson(
                    os.path.join(name, path))
                catalog = self.replace_dicts(catalog)
                catalog = dict(catalog)
                self.save_using_ujson(
                    catalog, os.path.join(name, path))

    def replace_dicts(self, catalog):
        '''
        replace dicts with lists of tuples recursively inside a list/dictionary
        '''
        import ast
        from class_objects import dict_oper
        try:
            evalcatalog = ast.literal_eval(str(catalog))
            if isinstance(evalcatalog, dict):
                for key in evalcatalog.keys():
                    newkey = self.replace_dicts(
                        key)
                    val = evalcatalog.pop(key)
                    if str(newkey) not in evalcatalog:
                        evalcatalog[str(newkey)] = val
                catalog = dict_oper.create_sorted_dict_view(
                    evalcatalog)
            elif isinstance(evalcatalog, list):
                newcatalog = []
                for item in evalcatalog:
                    newcatalog.append(
                        self.replace_dicts(item))
                catalog = newcatalog
        except (ValueError, SyntaxError) as e:
            pass
        return catalog

    def remove_keys(self, name, keys, startswith=None):
        import ast
        import os
        for path in os.listdir(name):
            if os.path.isdir(os.path.join(name, path)):
                self.remove_keys(os.path.join(name, path), keys,
                                 startswith)
            elif path.endswith('.json'):
                catalogs = self.load_using_ujson(
                    os.path.join(name, path))
                if not isinstance(catalogs, list):
                    catalogsislist = False
                    catalogs = [catalogs]
                else:
                    catalogsislist = True
                for catalog in catalogs:
                    for entry in catalog.keys():
                        try:
                            ds = ast.literal_eval(str(entry))
                        except BaseException:
                            continue
                        if not isinstance(ds, list):
                            dsislist = False
                            ds = [ds]
                        else:
                            dsislist = True
                        for dcount, d in enumerate(ds):
                            try:
                                evald = ast.literal_eval(str(d))
                            except BaseException:
                                continue
                            for key in evald.keys():
                                check = False
                                if startswith:
                                    check = key.startswith(
                                        startswith)
                                check = check or any([key == k for k in keys])
                                if check:
                                    evald.pop(key)
                            ds[dcount] = str(evald)
                        if not dsislist:
                            ds = ds[0]
                        val = catalog.pop(entry)
                        if str(ds) not in catalog:
                            catalog[str(ds)] = val
                if not catalogsislist:
                    catalogs = catalogs[0]
                self.save_using_ujson(
                    catalogs,
                    os.path.join(name, path))

    def load_using_ujson(self, filename):
        import ujson as json
        with open(filename, 'r') as inp:
            obj = json.load(inp)
        return obj

    def load_catalog(self, name, include_catalog=True):
        if include_catalog and 'include_catalog.json' in os.listdir(name):
            return self.load_using_ujson(os.path.join(name,
                                                      'include_catalog.json'))
        elif 'catalog.json' in os.listdir(name):
            return self.load_using_ujson(os.path.join(name,
                                                      'catalog.json'))
        else:
            return None

    def load_all_inside(self, name, keys_list=None, fold_lev=0, _id=None):
        if keys_list is None:
            import ast
            all_catalog = self.load_catalog(name, include_catalog=True)
            all_catalog = dict(all_catalog)
            keys_list = [ast.literal_eval(entry) for entry in all_catalog]
        else:
            all_catalog = None
        catalog = self.load_catalog(name, include_catalog=False)
        if catalog is None:
            return None
        if isinstance(keys_list[0][0], tuple):
            keys_list = [keys_list]
        import ast
        import sys
        entry_list = {}
        keys_result = {}
        for entry in keys_list:
            if all_catalog is not None:
                _id = all_catalog[str(entry)]
            if np.any([os.path.isdir(os.path.join(name, path))
                       for path in os.listdir(name)]):
                if str(entry[0]) in catalog:
                    search_key = str(entry[0])
                    next_keys_list = entry[1:]
                elif str([entry[0]]) in catalog:
                    search_key = str([entry[0]])
                else:
                    LOG.error('Attempted to access inexistent key: ' +
                              str([entry[0]]))
                    LOG.error('Existent keys are: ' + str(catalog.keys()))
                    sys.exit()
                loaded = self.load_all_inside(
                    os.path.join(name,
                                 str(catalog[search_key])),
                    entry[1:],
                    fold_lev - 1,
                    _id=_id)
                if not fold_lev:
                    entry_list[_id] = loaded
                    keys_result[_id] = entry
                else:
                    return loaded
            else:
                if _id is None:
                    return None
                try:
                    return self.load_obj(os.path.join(
                        name, str(catalog[str(keys_list[0])])) + '.pkl')
                except KeyError:
                    return self.load_obj(os.path.join(
                        name, str(catalog[str(keys_list[0][0])])) + '.pkl')
        return entry_list, keys_result

    def load_labeled_data(self, keys_list, name=None, fold_lev=-1,
                          just_catalog=False, all_inside=False,
                          include_all_catalog=False):
        import os
        if name is None:
            name = CONST['save_path']
        name = str(name)
        direct = name.split(os.sep)
        if not os.path.isdir(name):
            return None
        if 'catalog.json' in os.listdir(name):
            catalog = self.load_using_ujson(os.path.join(name,
                                                         'catalog.json'))
        else:
            return None
        if fold_lev == -1:
            fold_lev = len(keys_list)
        if (fold_lev > 0 and len(keys_list) > 1 or
                fold_lev > 0 and (just_catalog or all_inside)):
            if str(keys_list[0]) not in catalog:
                return None
            data = self.load_labeled_data(
                keys_list[1:],
                os.path.join(name, str(catalog[str(keys_list[0])])),
                fold_lev=fold_lev - 1, just_catalog=just_catalog,
                all_inside=all_inside,
                include_all_catalog=include_all_catalog)
        else:
            if just_catalog:
                return self.load_catalog(name, include_all_catalog)
            if all_inside:
                return self.load_all_inside(name)
            key = '.'.join(map(str, keys_list))
            if key not in catalog:
                return None
            if str(catalog[key]) + '.pkl' not in os.listdir(
                    name):
                return None
            return self.load_obj(os.path.join(name,
                                              str(catalog[key]))
                                 + '.pkl')
        return data

    def save_labeled_data(self, keys_list, data, name=None, fold_lev=-1):
        '''
        <keys_list> is a list of the keys that work as labels for <data>.
        <name> is the name of the folder to be used as hyperfolder to save
        labeled <data>. The <fold_lev> defines the number of keys inside <keys_list>
        to be used to create subfolders.
        The rest of the keys are used as entry of a catalogue inside the final
        subfolder. If fold_lev==-1, all the keys are used for subfolder
        creation, except of the last one.
        '''
        import os
        if name is None:
            name = CONST['save_path']
        name = str(name)
        direct = name.split(os.sep)
        if not os.path.isdir(name):
            makedir(name)
        if 'catalog.json' in os.listdir(name):
            catalog = self.load_using_ujson(os.path.join(name,
                                                         'catalog.json'))
        else:
            catalog = {}
        if fold_lev == -1:
            fold_lev = len(keys_list)
        if fold_lev > 0 and len(keys_list) > 1:
            if str(keys_list[0]) not in catalog:
                catalog[str(keys_list[0])] = len(catalog)
                self.save_using_ujson(catalog, os.path.join(
                    name, 'catalog.json'))
            self.save_labeled_data(
                keys_list[1:],
                data,
                os.path.join(name, str(catalog[str(keys_list[0])])),
                fold_lev=fold_lev - 1)
            if 'include_catalog.json' in os.listdir(name):
                all_catalog = self.load_using_ujson(os.path.join(
                    name, 'include_catalog.json'))
            else:
                all_catalog = {}
            if str(keys_list) not in all_catalog:
                all_catalog[str(keys_list)] = str(len(all_catalog))
                self.save_using_ujson(all_catalog, os.path.join(name,
                                                                'include_catalog.json'))
        else:
            key = '.'.join(map(str, keys_list))
            if key not in catalog:
                catalog[str(key)] = len(catalog)
                self.save_using_ujson(catalog, os.path.join(
                    name, 'catalog.json'))
            self.save_obj(data, os.path.join(name,
                                             str(catalog[key]))
                          + '.pkl')


class ImagesFolderOperations(object):

    def load_frames_data(self,
                         input_data,
                         imgs_fold_name=None,
                         masks_fold_name=None,
                         masks_needed=False,
                         derot_centers=None, derot_angles=None,
                         filetype='png'):
        '''
        <input_data> is the name of the folder including images.
        If there are two subfolders, the first one including the images and
        the second the masks to segment them, then <masks_needed> should be
        set to True and <imgs_fold_name> and <masks_fold_name> need to be set.
        The <derot_centers> and <derot_angles> can be provided if a rotation
        of each input image is wanted, but if not provided, such lists will be
        loaded from the subfolders, if <angles.txt> and <centers.txt> exist.
        The directory structure inside <input_data> or in both <imgs_fold_name> and
        <masks_fold_name> should be either in the form ./occurence number/frames
        or ./frames, with each frame being of <filetype> type.

        Returns images, masks, a vector <sync> keeping the numeric part of the name of each
        image, the angles if found, the centers if found and a vector <utterance_indices>
        keeping the number of the occurence folder of each frame, if it exists.
        '''
        if masks_needed:
            if imgs_fold_name is None:
                imgs_fold_name = CONST['mv_obj_fold_name']
            if masks_fold_name is None:
                masks_fold_name = CONST['hnd_mk_fold_name']
        files = []
        masks = []
        utterance_indices = []
        angles = []
        centers = []
        # check if multiple subfolders/samples exist
        try:
            mult_samples = (os.path.isdir(os.path.join(input_data, '0')) or
                            os.path.isdir(os.path.join(input_data, imgs_fold_name, '0')))
        except BaseException:
            mult_samples = False
        sync = []
        for root, dirs, filenames in os.walk(input_data):
            if not mult_samples:
                folder_sep = os.path.normpath(root).split(os.sep)
            for filename in sorted(filenames):
                fil = os.path.join(root, filename)
                if mult_samples:
                    folder_sep = os.path.normpath(fil).split(os.sep)
                if filename.endswith('png'):
                    ismask = False
                    if masks_needed:
                        if mult_samples:
                            ismask = folder_sep[-3] == masks_fold_name
                        else:
                            ismask = folder_sep[-2] == masks_fold_name
                    par_folder = folder_sep[-2]
                    try:
                        ind = int(par_folder) if mult_samples else 0
                        if ismask:
                            masks.append(fil)
                        else:
                            files.append(fil)
                            sync.append(int(filter(
                                str.isdigit, os.path.basename(fil))))
                            utterance_indices.append(ind)
                    except ValueError:
                        pass
                elif filename.endswith('angles.txt'):
                    with open(fil, 'r') as inpf:
                        angles += map(float, inpf)
                elif filename.endswith('centers.txt'):
                    with open(fil, 'r') as inpf:
                        for line in inpf:
                            center = [
                                float(num) for num
                                in line.split(' ')]
                            centers += [center]
        utterance_indices = np.array(utterance_indices)
        imgs = [cv2.imread(filename, -1) for filename
                in files]
        if masks_needed:
            masks = [cv2.imread(filename, -1) for filename in masks]
        else:
            masks = [None] * len(imgs)
        if derot_angles is not None and derot_centers is not None:
            centers = derot_centers
            angles = derot_angles

        act_len = sync[-1]
        return (imgs, masks, sync, angles, centers, utterance_indices)


class GroundTruthOperations(object):

    def __init__(self):
        self.ground_truths = {}
        self.gd_breakpoints = {}
        self.gd_labels = {}


    def pad_ground_truth(self, ground_truth, data, padding_const=0):
        '''
        Pad ground_truth if its length is lower than length of data.
        <padding_const=0> sets the padding constant.
        '''
        if len(data) != len(ground_truth):
            ground_truth = np.hstack(ground_truth,
                                     np.zeros(len(data) -
                                              len(ground_truth)) +
                                     padding_const)
        return ground_truth

    def load_ground_truth(self, gd_name, path=None,
                          ret_breakpoints=False,
                          ret_labs=False,
                          *args, **kwargs):
        '''
        Auxiliary function for processing a ground truth .csv file.
        Loads saved ground truth, using <path>/<gd_name> name identifier,
        or constructs it and saves it. Loading and
        saving is performed inside <path>. The <gd_name>.csv and the
        <gd_name>.pkl files should be present in the <path>.
        Other arguments are needed, if ground truth is to be constructed.
        Refer to construct_ground_truth.__doc__
        '''
        if path is None:
            path = CONST['ground_truth_fold']
        from os.path import getmtime
        clas_file = os.path.join(path, gd_name)
        csv_file = clas_file + '.csv'
        pkl_file = clas_file + '.pkl'
        try:
            csv_mtime = getmtime(csv_file)
        except OSError:
            LOG.error('Invalid Location for csv file: ' + csv_file)
            raise
        import pickle
        import datetime
        import time
        threshold = datetime.timedelta(seconds=3)
        recreate = False
        try:
            with open(pkl_file, 'r') as inp:
                pkl_mtime, loaded_data = pickle.load(inp)
                if ret_breakpoints:
                    if ret_labs:
                        ground_truth, breakpoints, labs = loaded_data
                    else:
                        ground_truth, breakpoints = loaded_data
                        if isinstance(breakpoints[0], str):
                            raise Exception
                else:
                    if ret_labs:
                        ground_truth, labs = loaded_data
                        if not isinstance(labs[0], str):
                            raise Exception
                    else:
                        ground_truth = loaded_data
                        if isinstance(ground_truth, tuple):
                            raise Exception

                if ret_breakpoints:
                    ground_truth, breakpoints = loaded_data
                else:
                    ground_truth = loaded_data
                delta = datetime.timedelta(seconds=pkl_mtime
                                           - csv_mtime)
                if delta > threshold:
                    recreate = True
        except BaseException:
            recreate = True
            pkl_mtime = 0
        if recreate:
            res = self.construct_ground_truth(
                ground_truth_type=csv_file,
                ret_breakpoints=ret_breakpoints,
                ret_labs=ret_labs,
                *args, **kwargs)

            with open(pkl_file, 'w') as out:
                pickle.dump((time.time(), res), out)
        return res

    def load_all_ground_truth_data(self):
        '''
        Loads all ground truth csv files inside <ground_truth_fold> defined in
        file config.yaml
        '''
        for fil in os.listdir(CONST['ground_truth_fold']):
            if fil.endswith('csv'):
                try:
                    name = os.path.splitext(fil)[0]
                    gd , br, lb = self.load_ground_truth(
                        name, ret_labs=True, ret_breakpoints=True)
                    name = name.replace('_',' ').title()
                    self.ground_truths[name] = gd
                    self.gd_breakpoints[name] = br
                    self.gd_labels[name] = lb
                except:
                    print fil
                    raise

    def create_utterances_vectors(self, breakpoints, frames_num=0):
        '''
        Using classes breakpoints, creates a corresponding dictionary of
        vectors, keeping numbered the instances of each class. If an
        element has no class, then -1 is given to it. If <frames_num> is not
        given, the resulting vectors will have size equal to the (maximum index
        in breakpoints + 1)
        '''
        res = {}
        if not frames_num:
            frames_num = max([max(breakpoints[key][1]) for key in
                              breakpoints]) + 1
        cnt = 0
        for key in breakpoints:
            res[key] = np.zeros(frames_num) - 1
            for (start, end) in zip(
                breakpoints[key][0], breakpoints[key][1]):
                res[key][start:end + 1] = cnt
                cnt += 1
        return res

    def merge_utterances_vectors(self, utt_vectors, classes_to_merge):
        '''
        Merge <utt_vectors> dictionary entries, with keys in
        <classes_to_merge>. The 0-th dimension of the result is the same as
        the mutiplicity of the classes, that is the number of the classes
        a vector element can have after the merge.
        '''
        selected = np.array([utt_vectors[key] for key in utt_vectors if key in
               classes_to_merge])
        flag = selected != -1
        check = np.sum(flag, axis=0)
        if 0:#np.max(check) == 1:
            fill = np.zeros((flag.shape[0],1))
            fill[0] = 1
            flag[:,check==0] = fill
            return selected[flag][None,:]
        else:
            res = np.zeros((np.max(check),selected.shape[1])) - 1
            for cnt,col in enumerate(selected.T):
                uni = np.unique(col[col!=-1])
                res[:len(uni),cnt] = uni
            return res
    def create_utterances_frequency_plots(self, val_filt='validation',
                                     test_filt='test'):
        '''
        Creates ground truth bar plots, that refer to the utterances frequency
        of each class, while separating the plots to validation, testing and
        training. The result is saved to GroundTruth/Utterances inside
        <results_fold> folder, defined in file config.yaml
        '''

        import pandas as pd
        import matplotlib.pyplot as plt
        self.load_all_ground_truth_data()
        results_loc = os.path.join(CONST['results_fold'],
                                   'GroundTruth',
                                   'Utterances')
        if not os.path.isdir(results_loc):
            os.makedirs(results_loc)

        val_ocs = {}
        test_ocs = {}
        train_ocs = {}
        max_val = {'val':0, 'train':0, 'test':0}
        for datakey in self.gd_breakpoints:
            if val_filt.lower() in datakey.lower():
                dic = val_ocs
                ind='val'
            elif test_filt.lower() in datakey.lower():
                dic = test_ocs
                ind='test'
            else:
                dic = train_ocs
                ind='train'
            dic[datakey] = {}
            for classkey in self.gd_breakpoints[datakey]:
                dic[datakey][classkey] = len(
                    self.gd_breakpoints[datakey][classkey][0])
                if (dic[datakey][classkey] == 1 and
                    not (val_filt.lower() in datakey.lower())
                    and not
                    (test_filt.lower() in datakey.lower())):
                    dic[datakey][classkey] = 100
                max_val[ind] = max(max_val[ind],
                                   dic[datakey][classkey])

        val_ocs_df = pd.DataFrame(val_ocs).fillna(0).astype(int)
        test_ocs_df = pd.DataFrame(test_ocs).fillna(0).astype(int)
        train_ocs_df = pd.DataFrame(train_ocs).fillna(0).astype(int)

        def plot_barh_frame(df,title=None,save_path=None,
                           max_val=None, change_last_xtick=None):
            '''
            Takes a pandas dataframe as input
            '''
            from matplotlib.ticker import MaxNLocator
            import matplotlib.pyplot as plt
            ax = df.plot(kind='barh')
            plt.ylabel('Gestures')
            plt.xlabel('Utterances')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if max_val is not None:
                ax.set_xlim([0,max_val])
                xticks = ax.get_xticks()
                xticks = np.unique(np.hstack((xticks,max_val)))
                ax.set_xticks(xticks)
            if change_last_xtick is not None:
                xticks = ax.get_xticks()
                xticklabels = [str(int(tick)) for tick in xticks]
                xticklabels[-1] = change_last_xtick
                ax.set_xticklabels(xticklabels)
            if title is not None:
                plt.title(title)
            plt.grid('off')
            leg = plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5),
                             title='Dataset')
            if save_path is not None:
                fig = ax.get_figure()
                fig.savefig(save_path, bbox_extra_artists=(leg,), bbox_inches='tight')
        plot_barh_frame(val_ocs_df,'Gestures Utterances in\n Validation Sets',
                save_path=os.path.join(results_loc,'validocs.pdf'),
                       max_val=max_val['val'])
        plot_barh_frame(test_ocs_df,'Gestures Utterances in\n Test Set',
                save_path=os.path.join(results_loc,'testocs.pdf'),
                       max_val=max_val['test'])
        plot_barh_frame(train_ocs_df,'Gestures Utterances in\n Training Sets',
                save_path=os.path.join(results_loc,'trainocs.pdf',
                                      ),
                       max_val=max_val['train'], change_last_xtick='1')


    def construct_ground_truth(self, data=None, classes_namespace=None,
                               length=None, ground_truth_type=None,
                               all_actions=True, ret_labs=False,
                               ret_breakpoints=False):
        '''
        <ground_truth_type>:'*.csv'(wildcard) to load csv
                                whose rows have format
                                class:start_index1,start_index2,..
                                start_indexn:end_index1,end_index2,...
                                end_indexn
                            'filename' to load class from datapath
                                filenames which have format
                                'number-class.png' or 'number.png'
                                if it has no class
                            'datapath' to load class from the name
                                of the directory filenames,
                                which were saved inside
                                <datapath/class/num1/num2.png> with <num1>
                                and <num2> integers, <num1> denoting sample
                                and <num2> frame number
                            'constant-*'(wildcard) to add the same ground
                                truth label for all .png files inside data
                                if * is a valid action and exists
                                inside <classes_namespace>
        <data> should be either a string refering to the datapath
        of the numbered frames or a boolean list/array.
        if <testing> is True, compare ground truth classes with
            <classes_namespace> and remove classes that do not exist
            inside <classes_namespace>
        If <classes_namespace> does not exist, it is created by sorting the
        labels extracted by the <ground_truth_type> or <data>.
        Returns <ground_truth> vectors which
        holds indices that refer to the <classes_namespace>, which is also returned.
        If <ret_names>, then the corresponging names for each index are also
        returned. Any item with no class has corresponding ground truth NaN
        '''
        if isinstance(data, basestring):
            if not os.path.exists(data):
                raise Exception(data + ' is a non existent path')
            paths = []
            for root, _, files in os.walk(data):
                root_separ = root.split(os.path.sep)
                if root_separ[-2] != CONST['hnd_mk_fold_name']:
                    paths += [os.path.join(root, filename)
                              for filename in sorted(files) if
                              filename.endswith('.png')]
            files = [os.path.basename(path) for path in paths]
            files = sorted(files)
            if not paths:
                raise Exception(data + ' does not include any png file')
        elif data is not None:
            data = np.array(data)
            if np.max(data) != 1:
                raise Exception('data should be boolean')
            else:
                data = data.astype(bool)
        ground_truth_init = {}
        max_ind = 0
        if ground_truth_type[-4::] == '.csv':
            try:
                with open(ground_truth_type, 'r') as inp:
                    for line in inp:
                        if '\n' in line:
                            line = line.replace('\n', '')
                        items = line.split(':')
                        if len(items) <= 1:
                            continue
                        try:
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[1].split(',')])
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[2].split(',')])
                        except (AttributeError, KeyError):
                            ground_truth_init[items[0]] = []
                            try:
                                ground_truth_init[items[0]].append([
                                    int(item) for item in items[1].split(',')])
                                ground_truth_init[items[0]].append([
                                    int(item) for item in items[2].split(',')])
                            except:
                                print items
                                raise
                        max_ind = max(max_ind, max(
                            ground_truth_init[items[0]][1]))
            except (EOFError, IOError):
                raise Exception('Invalid csv file given\n' +
                                self.construct_ground_truth.__doc__)
            keys = ground_truth_init.keys()
            class_match = {}
            if classes_namespace is None:
                classes_namespace = sorted(ground_truth_init.keys())
            for key in keys:
                try:
                    class_match[key] = classes_namespace.index(key)
                except ValueError:
                    ground_truth_init.pop(key, None)
            if not ground_truth_init:
                raise Exception(
                    'No classes found matching with training data ones')
            if data is None:
                ground_truth = np.zeros(max_ind + 1)
            elif not isinstance(data, basestring):
                if length is None:
                    ground_truth = np.zeros(len(data))
                else:
                    ground_truth = np.zeros(length)
            else:
                ground_truth = np.zeros(max([int(filter(str.isdigit,
                                                        os.path.basename(filename))) for
                                             filename in files]) + 1)
            ground_truth[:] = np.NaN
            all_bounds = [map(list, zip(*ground_truth_init[key])) for key in
                          ground_truth_init.keys()]
            if isinstance(data, basestring):
                iterat = [int(filter(str.isdigit,
                                     os.path.basename(filename)))
                          for filename in files]
            elif data is not None:
                iterat = np.where(data)[0]
            else:
                iterat = range(len(ground_truth))
            for count, ind in enumerate(iterat):
                for key, bounds in zip(ground_truth_init, all_bounds):
                    for bound in bounds:
                        if ind <= bound[1] and ind >= bound[0]:
                            ground_truth[ind] = class_match[key]
                            break
        elif ground_truth_type == 'filename':
            ground_truth_vecs = [filename.split('-') for filename
                                 in files]
            classes = []
            ground_truth = np.zeros(max([int(filter(str.isdigit, vec[1])) for
                                         vec in ground_truth_vecs]) + 1)
            ground_truth[:] = np.NaN
            inval_format = True
            for count, item in enumerate(ground_truth_vecs):
                if len(item) > 2:
                    inval_format = True
                    break
                if len(item) == 2:
                    inval_format = False
                    if all_actions:
                        if item[1] not in classes_namespace:
                            continue
                    if item[1] not in classes:
                        classes.append(item[1])
                    ground_truth[count] = classes.index(items[1])
            if inval_format:
                LOG.error('Invalid format')
                raise Exception(self.construct_ground_truth.__doc__)
        elif ground_truth_type == 'datapath':
            ground_truth_init = {}
            for path, filename in zip(paths, files):
                ground_truth_init[os.path.normpath(path)
                                  .split(os.path.sep)[-3]] = int(
                                      filter(str.isdigit, os.path.basename(
                                          filename)))
            keys = ground_truth_init.keys()
            if all_actions:
                class_match = {}
                for key in keys:
                    try:
                        class_match[key] = classes_namespace.index(key)
                    except ValueError:
                        ground_truth_init.pop(key, None)
                if not ground_truth_init:
                    raise Exception(
                        'No classes found matching with training data ones'
                        + '. The keys of the testing data are ' + str(keys))
            else:
                class_match = {}
                for count, key in enumerate(keys):
                    class_match[key] = count
                ground_truth = np.zeros(max([int(filter(str.isdigit,
                                                        os.path.basename(
                                                            filename))) for
                                             filename in files]) + 1)
                ground_truth[:] = np.NaN
                for key in ground_truth_init:
                    ground_truth[
                        np.array(ground_truth_init[key])] = class_match[key]
        elif ground_truth_type.split('-')[0] == 'constant':
            action_cand = ground_truth_type.split('-')[1]
            if all_actions:
                class_match = {}
                class_match[action_cand] = classes_namespace.index(
                    action_cand)
            else:
                class_match[action_cand] = 0
            if action_cand in classes_namespace:
                ground_val = classes_namespace.index(action_cand)
            else:
                raise Exception('Invalid action name, it must exists in '
                                + 'classes_namespace')
            ground_truth = np.zeros(max([int(filter(str.isdigit,
                                                    os.path.basename(filename))) for
                                         filename in files]) + 1)
            ground_truth[:] = np.NaN
            for fil in sorted(files):
                ground_truth[int(filter(
                    str.isdigit, os.path.basename(fil)))] = ground_val

        else:
            raise Exception('Invalid ground_truth_type\n' +
                            self.construct_ground_truth.__doc__)
        ret = (ground_truth,)
        if ret_breakpoints:
            ret += (ground_truth_init,)
        if ret_labs:
            ret += (classes_namespace,)
        return ret


class Hull(object):
    '''Convex Hulls of contours'''

    def __init__(self):
        self.hand = None


class Interp(object):
    '''Interpolated Contour variables'''

    def __init__(self):
        self.vecs_starting_ind = None
        self.vecs_ending_ind = None
        self.points = None
        self.final_segments = None


class KalmanFilter:

    def __init__(self):
        self.prev_objects_mask = np.zeros(0)
        self.cur_objects_mask = np.zeros(0)
        self.can_exist = np.zeros(0)


class Latex(object):
    '''
    Basic transriptions to latex
    '''


    def wrap_entry(self, str_to_analyse, ignore_num=True):
        '''
        found_strings = [m.group(1) for m in
                          re.finditer('\\\\(.*){(.*)}', str_to_analyse)]
        '''
        import re
        from textwrap import TextWrapper
        wrapper = TextWrapper(width=8, break_long_words=False,
                              break_on_hyphens=False, replace_whitespace=False)
        try:
            to_app, found_strings = zip(*[[m.group(1), m.group(2)] for m in
                                      re.finditer('(\\\\.*)\{([^{}]+)\}',
                                                  str_to_analyse)])
            to_app = list(to_app)
            found_strings = list(found_strings)
        except:
            found_strings = []
        if not found_strings:
            found_strings = [str_to_analyse]
            to_app = [0]

        for string in str_to_analyse.split(' '):
            if not string.startswith('\\'):
                found_strings.append(string)
                to_app.append(0)
        for string,to_ap in zip(found_strings, to_app):
            remove = [m.group(1) for m in re.finditer('(\\\\([a-zA-Z0-9]*) )', string)]
            for rem in remove:
                string = string.replace(rem,'')
            if ignore_num:
                perform_wrapping = False
                try:
                    float(string)
                    continue
                except:
                    pass
            wrapped = wrapper.fill(string)
            if wrapped.replace(' ',
                               '') != string.replace(' ',''):
                if to_ap:
                    wrapped = '\n'.join([to_ap+'{' +el + '}' for el in
                               wrapped.split('\n')])
                    str_to_analyse = str_to_analyse.replace(to_ap+'{'+
                                                            string+'}',
                                   '\\aspecialcell{' +
                                   wrapped.replace('\n','\\\\')
                                           + '}')
                else:
                    str_to_analyse = str_to_analyse.replace(string,
                                   '\\aspecialcell{' +
                                   wrapped.replace('\n','\\\\')
                                           + '}')

        return str_to_analyse

    def wrap_latex_table_entries(self,latex_text, width=8,
                                 ignore_num=True):
        '''
        gets a latex table and wraps its entries,
        to have a certain <width>, ignoring numeric
        values if <ignore_num>
        '''
        import re


        def wrap_table(latex_table):
            latex_table = latex_table.replace('\n',' ')
            latex_table = ' '.join(latex_table.split())
            inds = [[m.start(),(1 if m.group(1) is not None
                                else 2)] for m in
                                re.finditer('(&)|(\\\\\\\\)',
                                            latex_table)]
            for (start, start_off), (end, end_off) in zip(
                inds[:-1][::-1],inds[1:][::-1]):
                latex_table = (latex_table[:start+start_off]+
                    self.wrap_entry(latex_table[
                                         start+start_off:
                                         end])
                               +latex_table[end:])
            return latex_table
        tables_start_inds = [m.end() for m in
                       re.finditer('\\\\begin\{tabular\}',
                                   latex_text)]
        tables_end_inds = [m.start()-1 for m in re.
                           finditer('\\end{tabular}',
                           latex_text)]
        for tab_st, tab_en in zip(tables_start_inds[::-1],
                                  tables_end_inds[::-1]):
            latex_table = latex_text[tab_st:tab_en]
            latex_table = wrap_table(latex_table)
            latex_text = (latex_text[:tab_st] +
                          latex_table +
                          latex_text[tab_en:])
        preamble = ('\\newcommand{\\aspecialcell}[2][c]{ \n'+
                 '\\begin{tabular}[#1]{@{}c@{}}#2\\end{tabular}}\n ')
        if preamble not in latex_text:
            beg_ind = latex_text.find('\\begin{document}')
            if beg_ind == -1:
                latex_text = preamble + latex_text
            else:
                latex_text = (latex_text[:beg_ind] +
                                    preamble +
                                    latex_text[beg_ind:])
        return latex_text



    def compile(self, path, name):
        '''
        <path>:save path
        <name>:tex name
        '''
        import subprocess
        proc = subprocess.Popen(['pdflatex',
                                 '-output-directory',
                                 path,
                                 os.path.join(path, name)])
        proc.communicate()

    def add_package(self, data, package, options=None):
        if not package in data:
            import re
            try:
                pack_ind = [m.end()
                            for m in re.finditer('usepackage', data)][-1]
                pack_ind += data[pack_ind:].find('}') + 2
            except IndexError:
                try:
                    pack_ind = [m.end() for m in
                                re.finditer('documentclass', data)][0]
                    pack_ind += data[pack_ind:].find('}') + 3
                except:
                    pack_ind = 0
            pack_data = '\\usepackage'
            if options is not None:
                pack_data += '[' + options + ']'
            pack_data += '{' + package + '} \n'
            if pack_ind:
                data = (data[:pack_ind] + pack_data +
                        data[pack_ind:])
            else:
                data = pack_data + data
        return data

    def add_graphics(self, files, tex_path=None, captions=None,
                     labels=None, 
                     options=None, nomargins=False,
                     shrink_to_fit_only=True):
        if not isinstance(files, list):
            files = [files]
        if captions is None:
            captions = [None] * len(files)
        if not isinstance(captions, list):
            files = [captions]
        if labels is None:
            labels = [None] * len(files)
        if not isinstance(labels, list):
            labels = [labels]
        if len(files) != len(captions) or len(labels) != len(captions):
            print ('Error:Captions and labels should have the same length as' +
                   ' files if defined')
            exit()
        data = ''
        try:
            with open(tex_path, 'r') as inp:
                data += inp.read()
            tex_exists = True
        except (IOError, EOFError, TypeError):
            data = '\\documentclass[12pt,a4paper]{article} \n'
            tex_exists = False
        data = self.add_package(data, 'graphicx')
        data = self.add_package(data, 'float')
        data = self.add_package(data, 'grffile', 'space')
        if shrink_to_fit_only:
            data = self.add_package(data, 'adjustbox')
        if data.find('\\begin{document}') == -1:
            data += '\n\\begin{document}\n'

        if not tex_exists:
            data += '\\end{document}\n'
        data_ind = data.find('\\end{document}')
        data2add = ''
        for fil, caption, label in zip(files, captions, labels):
            if label is not None:
                if label.startswith('tab'):
                    data2add += '\\begin{table}[H] \n \\centering \n'
                else:
                    data2add += '\\begin{figure}[H] \n \\centering \n'
            if nomargins:
                data2add += '\\centerline{'
            if shrink_to_fit_only and options is not None:
                data2add += '\\includegraphics'
                data2add += '[' + options + ']'
            else:
                data2add += '\\includegraphics'
                if options is not None:
                    data2add += '[' + options + ']'
            data2add += '{' + fil + '}'
            if nomargins:
                data2add += '}'
            data2add += '\n'
            if caption is not None:
                data2add += '\\caption{' + caption + '}\n'
            if label is not None:
                data2add += '\\label{' + label + '}\n'
                if label.startswith('tab'):
                    data2add += '\\end{table}\n'
                else:
                    data2add += '\\end{figure}\n'
        data = data[:data_ind] + data2add + data[data_ind:]
        return data

    def array_transcribe(self, arr, xlabels=None, ylabels=None,
                         sup_x_label=None, sup_y_label=None,
                         extra_locs='bot', boldlabels=True,
                         title=None, isdataframe=False,
                         wrap=True, wrap_size=8,
                         round_nums=True, max_float=3):
        '''
        <arr> is the input array, <xlabels> are the labels along x axis,
        <ylabels> are the labels along the y axis, <sup_x_label> and
        <sup_y_label> are corresponding labels description. <arr> can be also a
        list of numpy arrays when <extra_locs> is a list of 'right' and 'bot' with
        the same length as the <arr[1:]> list (or a string if uniform structure
        is wanted). If this is the case, then starting
        by the first array in the list, each next array is concatenated to it,
        while adding either a double line or a double column separating them.
        The dimensions of the arrays and the labels should be coherent, or an
        exception will be thrown.
        '''
        doublerows = []
        doublecols = []
        whole_arr = None
        if isdataframe:
            if xlabels is None:
                xlabels = arr.keys()
            if ylabels is None:
                try:
                    ylabels = arr.index()
                except:
                    ylabels = arr.index.levels[0]
            arr = arr.values

        if isinstance(arr, list):
            if isinstance(extra_locs, basestring):
                extra_locs = [extra_locs] * (len(arr) - 1)
            if len(arr) != len(extra_locs) + 1:
                raise Exception('<extra_locs> should have the'
                                + ' same length as <arr> -1\n' +
                                self.array_transcribe.__doc__)
            if not isinstance(arr[0], np.ndarray) or len(arr[0].shape) == 1:
                arr[0] = np.atleast_2d(arr[0])
            whole_arr = arr[0]
            for array, loc in zip(arr[1:], extra_locs):
                if not isinstance(array, np.ndarray) or len(array.shape) == 1:
                    array = np.atleast_2d(array)
                if loc == 'right':
                    if whole_arr.shape[0] != array.shape[0]:
                        raise Exception('The dimensions are not coeherent\n' +
                                        self.array_transcribe.__doc__)
                    doublecols.append(whole_arr.shape[1])
                    whole_arr = np.concatenate((whole_arr, array), axis=1)
                elif loc == 'bot':
                    if whole_arr.shape[1] != array.shape[1]:
                        raise Exception('The dimensions are not coeherent\n' +
                                        self.array_transcribe.__doc__)
                    doublerows.append(whole_arr.shape[0])
                    whole_arr = np.concatenate((whole_arr, array), axis=0)
        elif len(arr.shape) == 1:
            whole_arr = np.atleast_2d(arr)
        else:
            whole_arr = arr
        if xlabels is not None:
            if boldlabels:
                xlabels = [r'\textbf{' + lab + r'}' for lab in xlabels]
            xlabels = np.array(xlabels)
            xlabels = xlabels.astype(list)
        if ylabels is not None:
            if boldlabels:
                ylabels = [r'\textbf{' + lab + r'}' for lab in ylabels]
            ylabels = np.array(ylabels)
            ylabels = ylabels.astype(list)
        y_size, x_size = whole_arr.shape
        y_mat, x_mat = whole_arr.shape
        ex_x = xlabels is not None
        ex_y = ylabels is not None
        ex_xs = sup_x_label is not None
        ex_ys = sup_y_label is not None
        x_mat = x_size + ex_y + ex_ys
        y_mat = y_size + ex_x + ex_xs
        init = '\\documentclass{standalone} \n'
        needed_packages = '\\usepackage{array, multirow, hhline, rotating}\n'
        cols_space = []
        if len(doublecols) != 0:
            doublecols = np.array(doublecols)
            doublecols += ex_y + ex_ys - 1
            for cnt in range(x_mat):
                if cnt in doublecols:
                    cols_space.append('c ||')
                else:
                    cols_space.append('c|')
        else:
            cols_space = ['c |'] * x_mat
        begin = '\\begin{document} \n'
        begin += '\\begin{tabular}{|' + \
            ''.join(cols_space) + '}\n'
        small_hor_line = '\\cline{' + \
            str(1 + ex_ys + ex_y) + '-' + str(x_mat) + '}'
        double_big_hor_line = ('\\hhline{' + (ex_ys) * '|~'
                               + (x_size + ex_y) * '|=' + '|}')
        big_hor_line = '\\cline{' + str(1 + ex_ys) + '-' + str(x_mat) + '}'
        whole_hor_line = '\\cline{1-' + str(x_mat) + '}'
        if title is not None:
            begin += '\\multicolumn'
            begin += '{'+str(x_mat)+'}{c}'
            begin += '{\\textbf{' + title + '}}\\\\[2ex] \n'
        if sup_x_label is not None:
            if boldlabels:
                sup_x_label = r'\textbf{' + sup_x_label + r'}'
            if ex_ys or ex_y:
                multicolumn = ('\\multicolumn{' + str(ex_ys + ex_y) + '}{c|}{} & ' +
                               '\\multicolumn{' + str(x_size) +
                               '}{c|}{' + sup_x_label + '} \\\\ \n')
            else:
                multicolumn = ('\\multicolumn{' + str(x_size) +
                               '}{|c|}{' + sup_x_label + '} \\\\ \n')

        else:
            multicolumn = ''
        if ex_ys:
            if boldlabels:
                sup_y_label = r'\textbf{' + sup_y_label + r'}'
            multirow = whole_hor_line + \
                '\\multirow{' + str(y_size) + '}{*}{\\rotatebox[origin=c]{90}{'\
                + sup_y_label + '}}'
        else:
            multirow = ''

        end = '\\hline \\end{tabular}\n \\end{document}'
        y = []
        for row in whole_arr.astype(str):
            y.append([])
            for elem in row:
                try:
                    int(elem)
                    y[-1].append(elem)
                except:
                    try:
                        float(elem)
                        if round_nums:
                            y[-1].append(
                                "%.*f" % (max_float,float(elem)))
                        else:
                            y[-1].append(elem)
                    except:
                        y[-1].append(elem)
        str_arr = y
        str_rows = [' & '.join(row) + '\\\\ \n ' for row in str_arr]
        if ex_y:
            str_rows = ["%s & %s" % (ylabel, row) for (row, ylabel) in
                        zip(str_rows, ylabels)]
        if ex_ys:
            str_rows = [" & " + str_row for str_row in str_rows]
        xlabels_row = ''
        if ex_x:
            if ex_ys or ex_y:
                xlabels_row = (' \\multicolumn{' + str(x_mat - x_size) +
                               '}{c |}{ } & ' + ' & '.
                               join(xlabels.astype(list)) + '\\\\ \n')
            else:
                xlabels_row = (' & '.join(xlabels.astype(list)) + '\\\\ \n')

        xlabels_row += multirow
        if not ex_ys:
            str_rows = [xlabels_row] + str_rows
        else:
            str_rows[0] = xlabels_row + str_rows[0]

        str_mat = (small_hor_line + multicolumn + small_hor_line)
        for cnt in range(len(str_rows)):
            str_mat += str_rows[cnt]
            if cnt in doublerows:
                str_mat += double_big_hor_line
            else:
                str_mat += big_hor_line
        str_mat = init + needed_packages + begin + str_mat + end
        if wrap:
            str_mat = self.wrap_latex_table_entries(str_mat)
        return str_mat


class Lim(object):
    '''limits for for-loops'''

    def __init__(self):
        self.max_im_num_to_save = 0
        self.init_n = 0


class LoggingOperations(object):
    '''
    operations to a logger or a logging module
    '''
    import logging

    class MaxLevelFilter(logging.Filter):
        '''Filters (lets through) all messages with level < LEVEL'''

        def __init__(self, level):
            self.level = level

        def filter(self, record):
            # "<" instead of "<=": since logger.setLevel is inclusive, this should be exclusive
            return record.levelno < self.level

    class SingleLevelFilter(logging.Filter):
        '''Filters (or filters out) messages of a specific level'''

        def __init__(self, passlevel, reject):
            self.passlevel = passlevel
            self.reject = reject

        def filter(self, record):
            if self.reject:
                return (record.levelno != self.passlevel)
            else:
                return (record.levelno == self.passlevel)

    def add_level(self, log_name, custom_log_module=None, log_num=None,
                  log_call=None,
                  lower_than=None, higher_than=None, same_as=None,
                  verbose=True):
        '''
        Function to dynamically add a new log level to a given custom logging module.
        <custom_log_module>: the logging module. If not provided, then a copy of
            <logging> module is used
        <log_name>: the logging level name
        <log_num>: the logging level num. If not provided, then function checks
            <lower_than>,<higher_than> and <same_as>, at the order mentioned.
            One of those three parameters must hold a string of an already existent
            logging level name.
        In case a level is overwritten and <verbose> is True, then a message in WARNING
            level of the custom logging module is established.
        '''
        if custom_log_module is None:
            import imp
            custom_log_module = imp.load_module('custom_log_module',
                                                *imp.find_module('logging'))
        log_name = log_name.upper()

        def cust_log(par, message, *args, **kws):
            # Yes, logger takes its '*args' as 'args'.
            if par.isEnabledFor(log_num):
                par._log(log_num, message, args, **kws)
        available_level_nums = [key for key in custom_log_module._levelNames
                                if isinstance(key, int)]

        available_levels = {key: custom_log_module._levelNames[key]
                            for key in custom_log_module._levelNames
                            if isinstance(key, str)}
        if log_num is None:
            try:
                if lower_than is not None:
                    log_num = available_levels[lower_than] - 1
                elif higher_than is not None:
                    log_num = available_levels[higher_than] + 1
                elif same_as is not None:
                    log_num = available_levels[higher_than]
                else:
                    raise Exception('Infomation about the ' +
                                    'log_num should be provided')
            except KeyError:
                raise Exception('Non existent logging level name')
        if log_num in available_level_nums and verbose:
            custom_log_module.warn('Changing ' +
                                   custom_log_module._levelNames[log_num] +
                                   ' to ' + log_name)
        custom_log_module.addLevelName(log_num, log_name)

        if log_call is None:
            log_call = log_name.lower()
        exec(
            'custom_log_module.Logger.' +
            eval('log_call') +
            ' = cust_log',
            None,
            locals())
        return custom_log_module


class Mask(object):
    '''binary masks'''

    def __init__(self):
        self.rgb_final = None
        self.background = None
        self.final_mask = None
        self.color_mask = None
        self.memory = None
        self.anamneses = None
        self.bounds = None


class Memory(object):
    '''
    class to keep memory
    '''

    def __init__(self):
        self.memory = None
        self.image = None
        self.bounds = None
        self.bounds_mask = None

    def live(self, anamnesis, importance=1):
        '''
        add anamnesis to memory
        '''
        if not isinstance(anamnesis[0, 0], np.bool_):
            anamnesis = anamnesis > 0
        if self.image is None:
            self.image = np.zeros(anamnesis.shape)
        else:
            self.image *= (1 - CONST['memory_fade_exp'])
        self.image += (CONST['memory_power'] * anamnesis * importance -
                       CONST['memory_fade_const'])
        self.image[self.image < 0] = 0

    def erase(self):
        '''
        erase memory
        '''
        self.memory = None
        self.image = None

    def remember(self, find_bounds=False):
        '''
        remember as many anamneses as possible
        '''
        if self.image is None:
            return None, None, None, None
        self.memory = self.image > CONST['memory_strength']
        if np.sum(self.memory) == 0:
            return None, None, None, None
        if find_bounds:
            self.anamneses = np.atleast_2d(cv2.findNonZero(self.memory.
                                                           astype(np.uint8)).squeeze())
            x, y, w, h = cv2.boundingRect(self.anamneses)
            self.bounds_mask = np.zeros_like(self.image, np.uint8)
            cv2.rectangle(self.bounds_mask, (x, y), (x + w, y + h), 1, -1)
            self.bounds = (x, y, w, h)
            return self.memory, self.anamneses, self.bounds, self.bounds_mask
        return self.memory, None, None, None

# class BackProjectionFilter(object):


class CamShift(object):
    '''
    Implementation of camshift algorithm for depth images
    '''

    def __init__(self):
        self.track_window = None
        self.rect = None

    def calculate(self, frame, mask1, mask2=None):
        '''
        frame and mask are 2d arrays
        '''
        _min = np.min(frame[frame != 0])
        inp = ((frame.copy() - _min) /
               float(np.max(frame) - _min)).astype(np.float32)
        inp[inp < 0] = 0
        hist = cv2.calcHist([inp], [0],
                            (mask1 > 0).astype(np.uint8),
                            [1000],
                            [0, 1])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        hist = hist.reshape(-1)
        if self.track_window and self.track_window[
                2] > 0 and self.track_window[3] > 0:
            if mask2 is None:
                mask2 = np.ones(frame.shape)
            prob_tmp = cv2.calcBackProject([inp[mask2 > 0]], [0], hist, [0, 1],
                                           1).squeeze()
            prob = np.zeros(frame.shape)
            prob[mask2 > 0] = prob_tmp
            '''
            term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                         CONST['max_count'],
                         CONST['par_eps'])
            self.rect, self.track_window = cv2.CamShift(prob,
                                                             self.track_window,
                                                             term_crit)
            '''
        return prob

    def reset(self):
        '''
        reset algorithm
        '''
        self.track_window = None
        self.rect = None

class MacroMetricsCalculation(object):
    def construct_vectors(self, true_classes, pred_classes, utterances_inds, threshold=0.5):
        '''
        <pred_classes> are the predicted classes
        <true_classes> are the ground truth classes
        <utterances_inds> is a dictionary of vectors, holding the indices of the utterances,
        taken from the ground truth
        <threshold> is the least rate of same category samples inside an
        utterance, so as to is assumed to be
        actually recognized.
        '''
        predicted_cats = []
        actual_cats = []
        try:
            utterances_inds[0][0]
        except:
            utterances_inds = [utterances_inds]
        for utt_row in utterances_inds:
            uni_utt = np.unique(utt_row)
            predicted_cats.append([])
            actual_cats.append([])
            for utt_ind in uni_utt:
                if utt_ind == -1:
                    predicted_cats[-1].append(-1)
                    actual_cats[-1].append(-1)
                    continue
                selection = np.array(utt_row) == utt_ind
                predicted = pred_classes[selection]
                actual = true_classes[selection]
                uni_actual, counts = np.unique(actual, return_counts=True)
                actual_cats[-1].append(uni_actual[np.argmax(np.unique(actual))])
                uni_pred, counts = np.unique(predicted, return_counts=True)
                if np.max(counts) > threshold * np.sum(selection):
                    predicted_cats[-1].append(uni_pred[
                        np.argmax(counts)])
                else:
                    predicted_cats[-1].append(-1)
        rav_actual_cats = np.array([item for sublist in actual_cats for item in sublist])
        rav_predicted_cats = np.array([item for sublist in predicted_cats for item in
                                       sublist])
        select = rav_actual_cats != -1
        res = rav_actual_cats[select], rav_predicted_cats[select]
        return res



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
        # lims: l->t->r->b
        self.nonconvex_edges_lims = []
        self.convex_edges_lims = []
        self.edges_positions_indices = []
        self.edges_positions = []
        # trusty_pixels is 1 for the pixels that remained nonzero during
        # initialisation
        self.trusty_pixels = np.zeros(1)
        # valid_values hold the last seen nonzero value of an image pixel
        # during initialisation
        self.valid_values = np.zeros(1)
        # changed all_positions to cart_positions
        self.cart_positions = None
        self.polar_positions = None
        self.background = np.zeros(1)
        self.found_objects_mask = np.zeros(0)
        self.hand_patch = None
        self.hand_patch_pos = None

    def construct_positions(self, img, polar=False):
        self.cart_positions = np.transpose(np.nonzero(np.ones_like(
            img))).reshape(img.shape + (2,))
        if polar:
            cmplx_positions = (self.cart_positions[:, :, 0] * 1j +
                               self.cart_positions[:, :, 1])
            ang = np.angle(cmplx_positions)
            ang[ang < -pi] += 2 * pi
            ang[ang > pi] -= 2 * pi
            self.polar_positions = np.concatenate(
                (np.absolute(cmplx_positions)[..., None], ang[..., None]),
                axis=2)
            return self.cart_positions, self. polar_positions
        else:
            return self.cart_positions


class Model(object):
    '''variables for modelling nonlinearities'''

    def __init__(self):
        self.noise_model = (0, 0)
        self.med = None
        self.var = None


class NoiseRemoval(object):

    def remove_noise(self, thresh=None, norm=True, img=None, in_place=False):
        if img is None:
            img = data.depth_im
        elif id(img) == id(data.depth_im):
            in_place = True
        if thresh is None:
            thresh = CONST['noise_thres']
        mask = img < thresh
        img = img * mask
        if norm:
            img = img / float(thresh)
        if in_place:
            data.depth_im = img
        return img

    def masked_mean(self, data, win_size):
        mask = np.isnan(data)
        K = np.ones(win_size, dtype=int)
        return np.convolve(np.where(mask, 0, data), K) / np.convolve(~mask, K)

    def masked_filter(self, data, win_size):
        '''
        Mean filter data with missing values, along axis 1
        '''
        if len(data.shape) == 1:
            inp = np.atleast_2d(data).T
        else:
            inp = data
        return np.apply_along_axis(self.masked_mean,
                                   0, inp, win_size)[:-win_size + 1]


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

# pylint:disable=no-self-use


class PolarOperations(object):
    '''
    Class to hold all used  operations on polar coordinates
    '''

    def derotate_points(self, img, points, angle, center):
        '''
        <points> should have dimension Nx2
        '''
        points = np.atleast_2d(points).T
        angle = - angle
        _cos = np.cos(angle)
        _sin = np.sin(angle)
        _x1 = img.shape[1] / 2.0
        _y1 = img.shape[0] / 2.0
        _x0 = center[1]
        _y0 = center[0]
        # Due to the non orthogonal system of the image, it is not
        #  [[_cos, _sin],[-_sin ,_cos]]
        M = np.array([[_cos, -_sin, -_x0 * _cos + _y0 * _sin + (_x1)],
                      [_sin, _cos, - _x0 * _sin - _y0 * _cos + (_y1)]])
        return np.dot(M,
                      np.concatenate((points,
                                      np.ones((1, points.shape[1]))),
                                     axis=0))

    def derotate(self, img, angle, center, in_rads=True):
        angle = - angle
        _cos = np.cos(angle)
        _sin = np.sin(angle)
        _x1 = img.shape[1] / 2.0
        _y1 = img.shape[0] / 2.0
        _x0 = center[1]
        _y0 = center[0]
        # Due to the non orthogonal system of the image, it is not
        #  [[_cos, _sin],[-_sin ,_cos]]
        M = np.array([[_cos, -_sin, -_x0 * _cos + _y0 * _sin + (_x1)],
                      [_sin, _cos, - _x0 * _sin - _y0 * _cos + (_y1)]])
        img = cv2.warpAffine(img, M, (img.shape[1],
                                      img.shape[0]))
        return img

    def find_cocircular_points(
            self, polar_points, radius, resolution=np.sqrt(2) / 2.0):
        '''
        Find cocircular points given radius and suggested resolution
        '''
        return polar_points[
            np.abs(polar_points[:, 0] - radius) <= resolution, :]

    def change_origin(self, old_polar, old_ref_angle,
                      old_ref_point, new_ref_point):
        '''Caution: old_polar is changed'''
        old_polar[:, 1] += old_ref_angle
        complex_diff = (new_ref_point[0] - old_ref_point[0]) * \
            1j + new_ref_point[1] - old_ref_point[1]
        polar_diff = [np.absolute(complex_diff), np.angle(complex_diff)]
        _radius = np.sqrt(polar_diff[0]**2 + old_polar[:, 0] * old_polar[:, 0] - 2 *
                          polar_diff[0] * old_polar[:, 0] * np.cos(old_polar[:, 1] - polar_diff[1]))
        _sin = old_polar[:, 0] * np.sin(old_polar[:, 1]) - \
            polar_diff[0] * np.sin(polar_diff[1])
        _cos = old_polar[:, 0] * np.cos(old_polar[:, 1]) - \
            polar_diff[0] * np.cos(polar_diff[1])
        _angle = np.arctan2(_sin, _cos)
        return np.concatenate((_radius[:, None], _angle[:, None]), axis=1)

    def polar_to_cart(self, polar, center, ref_angle):
        '''
        Polar coordinates to cartesians,
        given center and reference angle
        '''
        return np.concatenate(
            ((polar[:, 0] * np.sin(ref_angle + polar[:, 1]) +
              center[0])[:, None].astype(int),
             (polar[:, 0] * np.cos(ref_angle + polar[:, 1]) +
              center[1])[:, None].astype(int)), axis=1)

    def mod_correct(self, polar):
        '''
        Correct polar points, subjects to
        modular (-pi,pi). polar is changed.
        '''
        polar[polar[:, 1] > pi, 1] -= 2 * pi
        polar[polar[:, 1] < -pi, 1] += 2 * pi

    def fix_angle(self, angle):
        '''
        Same with mod_correct, for single angles
        '''
        if angle < -pi:
            angle += 2 * pi
        elif angle > pi:
            angle -= 2 * pi
        return angle

    def mod_between_vals(self, angles, min_bound, max_bound):
        '''
        Find angles between bounds, using modular (-pi,pi) logic
        '''
        if max_bound == min_bound:
            return np.zeros((0))
        res = self.mod_diff(max_bound, min_bound, 1)[1]
        if res == 0:
            return (angles <= max_bound) * (angles >= min_bound)
        else:
            return ((angles >= max_bound) * (angles <= pi)
                    + (angles >= -pi) * (angles <= min_bound))

    def mod_diff(self, angles1, angles2, ret_argmin=0):
        '''
        Angle substraction using modulo in (-pi,pi)
        '''
        sgn = -1 + 2 * (angles1 > angles2)
        if len(angles1.shape) == 0:

            diff1 = np.abs(angles1 - angles2)
            diff2 = 2 * pi - diff1
            if ret_argmin:
                return sgn * min([diff1, diff2]), np.argmin([diff1, diff2])
            else:
                return sgn * min([diff1, diff2])
        diff = np.empty((2, angles1.shape[0]))
        diff[0, :] = np.abs(angles1 - angles2)
        diff[1, :] = 2 * pi - diff[0, :]
        if ret_argmin:
            return sgn * np.min(diff, axis=0), np.argmin(diff, axis=0)
        else:
            return sgn * np.min(diff, axis=0)
# pylint:enable=no-self-use


class PlotOperations(object):

    def put_legend_outside_plot(self, axes, already_reshaped=False):
        '''
        Remove legend from the insides of the plots
        '''
        # Shrink current axis by 20%
        box = axes.get_position()
        if not already_reshaped:
            axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        lgd = axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return lgd

class PreprocessingOperations(object):
    def equalize_samples(self,
                         samples,
                         utterance_indices=None,
                         mode='random'):
        '''
        Equalizes dimensions of n_i x m arrays, with i=1:k the array index,
        in the <samples> list, cutting off random samples. 
        If <utterance_indices> are given
        (list of n_i vectors, describing samples utterances indices),
        then the cutting off is performed
        in utterances indices level.
        <mode> can be either 'random'(default) or 'serial'
        '''
        if utterance_indices is not None:
            samples_uniq_inds = []
            inv_inds = []
            for category in utterance_indices:
                uniq_samples, inv_indices = np.unique(category, return_inverse=True)
                samples_uniq_inds.append(uniq_samples[uniq_samples!=-1])
                try:
                    inv_indices = inv_indices[inv_indices!= np.where(uniq_samples
                                                                 == -1)[0]]
                except:
                    pass
                inv_inds.append(inv_indices)
        else:
            samples_uniq_inds = samples
            inv_inds = [np.arange(len(samples))]*len(samples)
        num_indices = min([len(cat) for cat in samples_uniq_inds])
        eq_samples = []
        inds_to_return = []
        for category_inds, category, inv_indices in zip(
            samples_uniq_inds, samples, inv_inds):
            if mode == 'random':
                sel_indices = np.random.choice(len(category_inds),num_indices,
                                           replace=False)
            elif mode == 'serial':
                sel_indices = np.arange(num_indices)
            mask =  np.any(np.any(inv_indices[None,:]==sel_indices[:,None],axis=0),axis=0)
            eq_samples.append(np.array(category)[mask])
        return eq_samples
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
            return 1
        shapes = [(im_shape[0], im_shape[1], c, len(im_shape))
                  for (c, im_shape) in enumerate([im.shape for im in self.images])]
        isrgb = 1 if sum(zip(*shapes)[3]) != 2 * len(shapes) else 0
        sorted_images = [self.images[i] for i in list(
            zip(*sorted(shapes, key=lambda x: x[0], reverse=True))[2])]
        imy, imx, _, _ = tuple([max(coord) for
                                coord in zip(*shapes)])
        yaxis = (len(self.images) - 1) / self.maxdim + 1
        xaxis = min(self.maxdim, len(self.images))
        if not isrgb:
            montage = 255 *\
                np.ones((imy * yaxis, imx * xaxis), dtype=np.uint8)
        elif isrgb:
            montage = 255 *\
                np.ones((imy * yaxis, imx * xaxis, 3), dtype=np.uint8)
        x_init = 0
        for count, image in enumerate(sorted_images):
            _max = np.max(image)
            _min = np.min(image)
            if _max != _min:
                image = ((image - _min) / float(_max - _min) *
                         255).astype(np.uint8)
            if isrgb:
                if len(image.shape) == 2:
                    image = np.tile(image[:, :, None], (1, 1, 3))
            if not isrgb:
                montage[(count / self.maxdim) * imy: (count / self.maxdim + 1)
                        * imy, x_init: x_init + image.shape[1]] = image
            else:
                montage[(count / self.maxdim) * imy: (count / self.maxdim + 1)
                        * imy, x_init: x_init + image.shape[1], :] = image
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
                raise(e)
        else:
            cv2.imwrite(var2, montage)
        return

    def print_results(self, filename):
        '''class to print data to file'''
        with open(filename, 'w') as fil:
            for line in self.data:
                fil.write(line + '\n')


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
        if len(points) <= 1:
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
                    self.locs[:locs.shape[0], count] = locs[
                        :, 0] + locs[:, 1] * 1j
                except BaseException:
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
        '''cv2.imshow('test',self.uint8_depth_im)
        cv2.waitKey(0)
        '''
        for count in np.arange(self.count + 1)[self.centers_to_calculate]:
            xcoords = self.locs[
                :self.pixsize[count], count].real.astype(int)
            ycoords = self.locs[
                :self.pixsize[count], count].imag.astype(int)
            self.masses[count] = np.sum(data.uint8_depth_im[xcoords, ycoords])
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
                    count, :self.pixsize[count]] = data.uint8_depth_im[xcoords, ycoords]
        if first_time:
            self.initial_center = self.center.copy()


class Segmentation(object):
    ''' Objects used for background segmentation '''

    def __init__(self):
        self.bounding_box = []
        self.needs_segmentation = 1
        self.check_if_segmented_1 = 0
        self.check_if_segmented_2 = 0
        self.prev_im = np.zeros(0)
        self.exists_previous_segmentation = 0
        self.initialised_centers = 0
        self.z_objects = SceneObjects()
        self.nz_objects = SceneObjects()
        self.neighborhood_existence = np.zeros(0)
        self.proximity_table = np.zeros(0)
        self.filled_neighborhoods = []
        self.found_objects = np.zeros(0)
        self.total_obj_num = 0
        self.fgbg = None

    def flush_previous_segmentation(self):
        self.bounding_box = []
        self.z_objects = SceneObjects()
        self.nz_objects = SceneObjects()
        self.proximity_table = np.zeros(0)
        self.filled_neighborhoods = []
        self.found_objects = np.zeros(0)
        self.total_obj_num = 0

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
            self.neighborhood_existence[count, :] = np.sum(self.proximity_table
                                                           == count, axis=0)

    def find_objects(self):
        time1 = time.clock()
        check_atoms = []
        # nz_objects.center at the beginning of center list so following is
        # valid
        for count1, vec in\
                enumerate(self.nz_objects.center_displacement):
            if (abs(vec[0]) > CONST['min_displacement'] or abs(vec[1]) >
                    CONST['min_displacement']):
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
                    if neighbor not in self.filled_neighborhoods[
                            neighborhood_id]:
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
                            CONST['depth_tolerance']) * (vals > 0)
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
                            CONST['depth_tolerance']) * (last_vals > 0)
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
        # euclid_dists = np.abs(np.transpose(complex_inters) - complex_inters)
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


class TableOperations(object):

    def __init__(self, usetex=False):
        self.usetex = usetex

    def construct(self, axes, cellText, colLabels=None,
                  rowLabels=None, cellColours=None,
                  cellLoc='center', loc='center',
                  boldLabels=True, usetex=None):
        if usetex is not None:
            self.usetex = usetex
        with plt.rc_context({'text.usetex': self.usetex,
                             'text.latex.unicode': self.usetex}):

            if boldLabels and self.usetex:
                if colLabels is not None:
                    colLabels = [('\n').join([r'\textbf{' + el + r'}' for el in
                                              row_el.split('\n')]) for row_el in
                                 colLabels]
                if rowLabels is not None:
                    rowLabels = [('\n').join([r'\textbf{' + el + r'}' for el in
                                              row_el.split('\n')]) for row_el in
                                 rowLabels]
                table = axes.table(cellText=cellText,
                                   colLabels=colLabels,
                                   rowLabels=rowLabels,
                                   cellColours=cellColours,
                                   cellLoc=cellLoc,
                                   loc=loc)
            if boldLabels and not self.usetex:
                if colLabels is not None:
                    [table.properties()['celld'][element].get_text().set_fontweight(1000)
                     for element in
                     table.properties()['celld'] if element[0] == 0]
                if colLabels is not None:
                    [table.properties()['celld'][element].get_text().set_fontweight(1000)
                     for element in
                     table.properties()['celld'] if element[1] == 0]
        return table

    def fit_cells_to_content(self, fig,
                             the_table, inc_by=0.1, equal_height=False,
                             equal_width=False, change_height=True,
                             change_width=True):
        table_prop = the_table.properties()
        # fill the transpose or not, if we need col height or row width
        # respectively.
        rows_heights = [[] for i in range(len([cells for cells in
                                               table_prop['celld'] if cells[1] == 0]))]
        cols_widths = [[] for i in range(len([cells for cells in
                                              table_prop['celld'] if cells[0] == 0]))]
        renderer = fig.canvas.get_renderer()
        for cell in table_prop['celld']:
            text = table_prop['celld'][(0, 0)]._text._text

            bounds = table_prop['celld'][cell].get_text_bounds(renderer)
            cols_widths[cell[1]].append(bounds[2])
            rows_heights[cell[0]].append(bounds[3])
        cols_width = [max(widths) for widths in cols_widths]
        if equal_width:
            cols_width = [max(cols_width)] * len(cols_width)
        rows_height = [max(heights) for heights in rows_heights]
        if equal_height:
            rows_height = [max(rows_height)] * len(rows_height)
        if not isinstance(inc_by, list):
            inc_by = [inc_by, inc_by]
        for cell in table_prop['celld']:
            bounds = table_prop['celld'][cell].get_text_bounds(renderer)
            new_width = ((1 + inc_by[0]) * cols_width[cell[1]] if change_width else
                         bounds[2])
            new_height = ((1 + inc_by[1]) * rows_height[cell[0]] if change_height
                          else bounds[3])
            table_prop['celld'][cell].set_bounds(*(bounds[:2] + (new_width,
                                                                 new_height,
                                                                 )))


class Threshold(object):
    '''necessary threshold variables'''

    def __init__(self):
        self.lap_thres = 0
        self.depth_thres = 0


class TimeOperations(object):

    def __init__(self):
        self.times_mat = []
        self.labels = []
        self.convert_to_ms = True
        self.meas = 's'

    def compute_stats(self, time_list, label='', print_out=True,
                      convert_to_ms=True):
        '''
        <time_list> is a list or list of lists or a numpy array
        <label> can be a string or a list of strings.
        '''
        time_array = np.atleast_2d(np.array(time_list).squeeze())
        if isinstance(label, basestring):
            label = [label] * time_array.shape[0]
        print time_array.shape
        if len(label) != min(time_array.shape):
            raise Exception('Invalid number of labels given')
        self.convert_to_ms = convert_to_ms
        if self.convert_to_ms:
            self.meas = 'ms'
            time_array = time_array * 1000
        stats = np.array([np.mean(time_array, axis=1),
                          np.max(time_array, axis=1),
                          np.min(time_array, axis=1),
                          np.median(time_array, axis=1)])
        for count in range(time_array.shape[0]):
            if print_out:
                print('Mean ' + label[count] + ' time ' +
                      str(stats[0, count]) + ' ' + self.meas)
                print('Max ' + label[count] + ' time ' +
                      str(stats[1, count]) + ' ' + self.meas)
                print('Min ' + label[count] + ' time ' +
                      str(stats[2, count]) + ' ' + self.meas)
                print('Median ' + label[count] + ' time ' +
                      str(stats[3, count]) + ' ' + self.meas)
        self.times_mat.append(stats)
        self.labels.append([lab.title() for lab in label])

    def extract_to_latex(self, path, stats_on_xaxis=True):
        '''
        extract total stats to a latex table.
        '''
        stats_labels = [
            'Mean(' + self.meas + ')', 'Max(' + self.meas + ')',
            'Min(' + self.meas + ')', 'Median(' + self.meas + ')']
        if stats_on_xaxis:
            time_array = latex.array_transcribe(self.times_mat, stats_labels,
                                                ylabels=self.labels,
                                                extra_locs='right')
        else:
            time_array = latex.array_transcribe(np.transpose(
                self.times_mat), self.labels,
                ylabels=stats_labels,
                extra_locs='bot')
        with open(os.path.splitext(path)[0] + '.tex',
                  'w') as out:
            out.write(time_array)


with open(CONST_LOC + "/config.yaml", 'r') as stream:
    try:
        CONST = yaml.load(stream)
    except yaml.YAMLError as exc:
        print exc
# pylint: disable=C0103
circ_oper = CircularOperations()
contours = Contour()
counters = Counter()
chhm = CountHandHitMisses()
data = Data()
dict_oper = DictionaryOperations()
draw_oper = DrawingOperations()
edges = Edges()
file_oper = FileOperations()
gd_oper = GroundTruthOperations()
imfold_oper = ImagesFolderOperations()
latex = Latex()
lims = Lim()
log_oper = LoggingOperations()
macro_metrics = MacroMetricsCalculation()
masks = Mask()
meas = Measure()
models = Model()
paths = Path()
points = Point()
plot_oper = PlotOperations()
pol_oper = PolarOperations()  # depends on Measure class
preproc_oper = PreprocessingOperations()
table_oper = TableOperations()
thres = Threshold()
type_conv = TypeConversions()
im_results = Result()
segclass = Segmentation()
existence = ExistenceProbability()
interpolated = contours.interpolated
noise_proc = NoiseRemoval()
