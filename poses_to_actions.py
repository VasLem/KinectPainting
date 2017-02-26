import os, sys, errno
import numpy as np
import classifiers as clfs
import class_objects as co
import cPickle as pickle
import logging



def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
class MixedClassifier(clfs.Classifier):
    '''
    CLassifier that combines a specified SVMs classifier <svms_classifier>,
    that classifies all actions in timespace, and a Random Forest one, <rf_classifier>,
    that classifies static actions. During training phase, a confusion matrix
    between random forest predictions and provided ground truth of actions
    (static and dynamic) is constructed, named 'coherence matrix'.
    The combination is done using a mixed Bayesian-deduced model, which is
    explained in detail in <run_testing> . It basically uses both classifiers,
    to produce a better estimation, which is extremely useful in realtime,
    where the SVMs classifier can not cope well in actions boundaries.
    '''
    def __init__(self, svms_classifier=None, rf_classifier=None,
                 log_lev='INFO',visualize=False,
                 add_info='without sparse coding'):
        # matrix rows are poses labels`
        # matrix columns are actions labels
        # matrix entries are coherence probabilities
        if svms_classifier is None or rf_classifier is None:
            raise Exception(self.__doc__)
        self.poses_classifier = rf_classifier  # poses classifier
        self.actions_classifier = svms_classifier  # actions classifier
        clfs.Classifier.__init__(self,log_lev=log_lev, visualize=visualize,
                                 masks_needed=False,
                 buffer_size = self.actions_classifier.buffer_size,
                 ispassive = False, use='mixed', name='actions',
                                add_info=add_info)
        self.poses_classifier_test = None
        self.actions_classifier_test = None
        self.coherence_matrix = None
        self.actions = self.actions_classifier.train_classes
        self.poses = self.poses_classifier.train_classes
        _max = len(self.poses_classifier.train_classes)
        match = [-1] * (len(self.actions))
        for cnt,pose in enumerate(self.poses):
            act_ind = self.actions.index(pose)
            match[cnt] = act_ind
        left = list(set(range(len(match)))-set(match))
        for cnt in range(len(match)):
            if match[cnt] == -1:
                match[cnt] = left[0]
                left = left[1:]
        self.a2p_match = np.array(match)
        self.train_classes = np.array(
                self.actions_classifier.train_classes)[self.a2p_match].tolist()
        # actions ground truth
        self.poses_predictions = None
        self.actions_ground_truth = None

    def extract_pred_and_gt(self):
        '''
        This function loads from a <csv_pathname> the actions ground truth and
        tests <self.poses_classifier> using images inside <set_pathname>,
        extracting preficted poses.
        <set_pathname> is the pathname of the folder of png images refering
        to the <csv_pathname>, which holds the corresponding actions
        ground truth. So both <set_pathname> and <csv_pathname> should be
        correlated.
        The actions ground truth contains only actions that exist already
        inside <self.actions_classifier>. If this is not desired, set
        <keep_only_valid_actions> to False.
        '''
        ground_truths =[os.path.splitext(fil)[0] for fil
                        in os.listdir(co.CONST['ground_truth_fold'])
                        if fil.endswith('.csv')]
        rosbags = [os.path.splitext(fil)[0] for fil in
                   os.listdir(co.CONST['rosbag_location'])
                   if fil.endswith('.bag')]
        to_process = [rosbag for rosbag in rosbags if rosbag in ground_truths]
        ground_truths = [os.path.join(
            co.CONST['ground_truth_fold'],name+'.csv') for name in to_process]
        rosbags = [os.path.join(
            co.CONST['rosbag_location'],name+'.bag') for name in to_process]
        self.poses_predictions = []
        self.actions_ground_truth = []
        data = {}
        prev_root = ''
        prev_action = ''
        for root, dirs, filenames in os.walk(co.CONST['actions_path']):
            for filename in sorted(filenames):
                if root != prev_root and str.isdigit(
                    os.path.normpath(
                        root).split(
                            os.path.sep)[-1]):
                    prev_root = root
                    action = os.path.normpath(
                            root).split(os.path.sep)[-2]
                    if action != prev_action:
                        LOG.info('Processing action: ' + action)
                        prev_action = action
                    if action not in self.actions:
                        raise Exception('Action ' + action +
                                        'not in trained actions superset')
                    self.poses_classifier.reset_offline_test()
                    actions_ground_truth = (
                        self.actions_classifier.construct_ground_truth(
                        data=root, ground_truth_type='constant-' +action))
                    self.poses_classifier.run_testing(
                        data=root, online=False,
                        construct_gt=False,
                        save=False,load=False,display_scores=False)
                    poses_pred = self.poses_classifier.recognized_classes
                    poses_predictions_expanded = np.zeros(
                        np.max(self.poses_classifier.test_sync) + 1)
                    poses_predictions_expanded[:] = None
                    poses_predictions_expanded[self.poses_classifier.test_sync
                                       ] = poses_pred
                    poses_pred = poses_predictions_expanded
                    fmask = np.isfinite(
                                poses_pred)*np.isfinite(actions_ground_truth)
                    self.poses_predictions += (poses_pred[
                        fmask].astype(int)).tolist()
                    self.actions_ground_truth += (
                        actions_ground_truth[fmask].astype(int)).tolist()
        self.poses_classifier.reset_offline_test()

    def construct_coherence(self):
        '''
        Coherence matrix is a matrix which has as many rows as the poses
        labels and as many columns as the actions labels. Each row sums to 1
        and describes the coherence between the corresponding pose and the
        action labels.
        This function constructs actions to pose <coherence_matrix>.
        <extract_variables> should be run first
        '''
        if self.actions is None:
            raise Exception(self.construct_coherence.__doc__)
        self.coherence_matrix = np.zeros((len(self.poses),
                                          len(self.actions)))
        
        for action_truth, pose_pred in zip(self.actions_ground_truth,
                                           self.poses_predictions):
            self.coherence_matrix[pose_pred,
                                  action_truth] += 1
        self.coherence_matrix = self.coherence_matrix[
            :, self.a2p_match]
        #self.actions = [self.actions[cnt] for cnt in self.a2p_match]
        self.coherence_matrix = self.coherence_matrix / np.sum(
            self.coherence_matrix, axis=0)
        return self.coherence_matrix

    def run_training(self, save=True, load=True,
                     classifiers_savepath=None):
        '''
        Computes coherence matrix
        Overrides <clfs.run_training>
        '''
        if classifiers_savepath is None:
            classifiers_savepath = 'trained_'
            classifiers_savepath += self.full_name.replace(' ','_').lower()
            classifiers_savepath += '.pkl'
        if load and os.path.isfile(classifiers_savepath):
            with open(classifiers_savepath, 'r') as inp:
                (self.coherence_matrix,
                 self.poses,
                 self.actions) = pickle.load(inp)
        else:
            self.extract_pred_and_gt()
            self.construct_coherence()
            if save:
                with open(classifiers_savepath,'w') as out:
                    pickle.dump((self.coherence_matrix,self.poses,self.actions), out)

        return self.coherence_matrix, self.poses, self.actions

    def plot_coherence(self, save=True):
        '''
        Plots the coherence matrix
        '''
        from matplotlib import pyplot as plt
        import matplotlib
        matplotlib.rc('text', usetex=True)
        matplotlib.rcParams['text.latex.unicode'] = True
        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        #pylint: disable=no-member
        res = ax.imshow(self.coherence_matrix, cmap=plt.cm.jet,
                        interpolation='nearest')
        #pylint: enable=no-member
        for c1 in xrange(len(self.poses)):
            for c2 in xrange(len(self.actions)):
                ax.annotate(str('$%.2f$' % self.coherence_matrix[c1,c2]), xy=(c2,
                                                                            c1),
                            horizontalalignment='center',
                            verticalalignment='center',fontsize=16)

        cb = fig.colorbar(res)
        ax.set_xlabel(r'Actions', fontsize=16)
        ax.set_ylabel(r'Poses', fontsize=16)
        ax.set_title(r'Coherence', fontsize=18)
        plt.xticks(range(len(self.actions)), [r'%s' % action for action in
                                              self.actions], rotation = 45)
        plt.yticks(range(len(self.poses)), [r'%s' % pose for pose in
                                            self.poses])
        save_fold = os.path.join(
            co.CONST['results_fold'], 'Classification', 'Poses RF',)
        makedir(save_fold)
        if save:
            plt.savefig(os.path.join(
                save_fold, 'Coherence Matrix.pdf'))


    def run_mixer(self, rf_probs, svms_scores=None, img_count=None, save=False,
                  online=True,*args,**kwargs):
        if not self.testing_initialized:
            if not online:
                self.reset_offline_test()
            else:
                self.reset_online_test()
        if isinstance(rf_probs, tuple):
            svms_scores= rf_probs[1]
            rf_probs = rf_probs[0]
        if img_count is not None:
            self.img_count=img_count
        fmask_svms = np.isfinite(np.array(svms_scores[:, 0]))
        fmask_rf = np.isfinite(np.array(rf_probs[:,0]))
        partial_lack = np.logical_xor(fmask_svms, fmask_rf)
        partial_lack_svms = np.logical_and(partial_lack,
                                           np.logical_not(fmask_svms))
        partial_lack_rf = np.logical_and(partial_lack,
                                         np.logical_not(fmask_rf))
        fin_svms_probs = svms_scores[fmask_svms, :]
        _mins = np.min(fin_svms_probs, axis=1)[:, None]
        _maxs = np.max(fin_svms_probs, axis=1)[:, None]
        below_z = fin_svms_probs < 0
        thres = 1/float(len(self.train_classes))
        fin_svms_probs = (thres * (below_z*fin_svms_probs - _mins)/
                          (- _mins).astype(float) +
                          (1-thres) *((1-below_z)*fin_svms_probs/
                                      _maxs.astype(float)))
        fin_svms_probs = fin_svms_probs/np.sum(
            fin_svms_probs,axis=1)[:,None]
        exp_svms_probs = np.zeros_like(svms_scores)
        exp_svms_probs[partial_lack_svms, :] = thres
        exp_svms_probs[fmask_svms, :] = fin_svms_probs

        total = np.logical_or(fmask_svms, fmask_rf)
        fin_svms_probs = exp_svms_probs[total, :]
        fin_svms_probs = fin_svms_probs.T
        if rf_probs[total,:].shape[0] == 0:
            fin_inv_rf_probs = np.zeros_like(rf_probs) + 1/float(
                thres)
        else:
            fin_rf_probs = rf_probs[fmask_rf,:]
            fin_inv_rf_probs = np.zeros_like(fin_rf_probs)
            fin_inv_rf_probs[fin_rf_probs != 0] = 1/ fin_rf_probs[
                fin_rf_probs != 0]
            inv_rf_probs = np.zeros_like(rf_probs)
            inv_rf_probs[fmask_rf,:] = fin_inv_rf_probs
            inv_rf_probs[partial_lack_rf, :] = 1/float(thres)
            fin_inv_rf_probs = inv_rf_probs[total, :]
        fin_inv_rf_probs = fin_inv_rf_probs.T

        self.scores = np.zeros_like(svms_scores)
        self.scores[:] = np.NaN
        fin_scores = np.dot(self.coherence_matrix.T,
                              np.dot(self.coherence_matrix,
                              fin_svms_probs)
                              * fin_inv_rf_probs) * fin_svms_probs
        fin_scores = fin_scores.T
        fin_scores = np.minimum(fin_scores, 10)
        self.scores[total, :] = fin_scores
        _sum = np.sum(
            self.scores[total, :], axis=1)[:, None]
        _sum[_sum == 0] = 1
        self.scores[total, :] = self.scores[total, :] / _sum
        #pylint: disable=no-member
        if not online:
            self.recognized_classes = self.classify_offline(display=True)
            self.display_scores_and_time(save=save)
        else:
            self.classify_online(self.scores.ravel(),
                                 self.img_count,
                                 self.actions_classifier.mean_from)
        #pylint: enable=no-member

        self.img_count += 1

        return self.recognized_classes, self.scores



    def run_testing(self, data=None, online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=co.CONST['test_actions_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=False, testname=None, display_scores=True,
                    derot_angle=None, derot_center=None,
                    construct_gt=True, just_scores=False):
        '''
        Mixed bayesian model, meant to provide unified action scores.
        P(p_i|a_j) = c_ij in Coherence Map C
        P(a_j|t) = probabilities produced by svms scores
        P(p_i|t) = poses RF probability scores
        Combined Prediction = Sum{i}{c[i,j]*P[a_j]/P[p_i]*Sum{k}{c[i,k]*P[a_j]}}
        If S is the matrix [P(a_j|t)[j,t]], j=0:n-1, t=0:T-1
        and R is the matrix [1/P[p_i|t][i,t]], i=0:m, t=0:T-1
        then the combined prediction becomes S'= S x (C.T * (R x (C*S)))
        where '*' is the dot product and 'x' is the Hadamard product.
        If S[:,t] is missing, it is replaced by a uniform hypothesis of
        probability.

        Overrides <clfs.Classifier.run_testing>, but the input arguments are
        the same, so for help consult <classifiers.Classifiers>
        '''
        if online:
            if img_count is not None:
                self.scores_exist += ((img_count - self.img_count) * [False])
                if img_count is not None:
                    self.img_count = img_count
        self.init_testing(data=data,
                          online=online,
                          save=save,
                          load=load,
                          testname=testname,
                          scores_savepath=scores_savepath,
                          scores_filter_shape=5,
                          std_small_filter_shape=co.CONST[
                              'STD_small_filt_window'],
                          std_big_filter_shape=co.CONST[
                              'STD_big_filt_window'])
        rf_exist,_ = self.poses_classifier.run_testing(data=data,
                                          online=online,
                                          construct_gt=False,
                                          ground_truth_type=ground_truth_type,
                                         save=True,
                                         load=True,
                                         display_scores=True,
                                         testname = testname,
                                         just_scores=True,
                                         derot_angle=derot_angle,
                                             derot_center=derot_center,
                                            img_count=self.img_count)
        svms_scores_exist, _ = self.actions_classifier.run_testing(data=data,
                                            derot_angle=derot_angle,
                                               derot_center=derot_center,
                                            online=online,
                                            construct_gt=False,
                                            ground_truth_type=ground_truth_type,
                                            save=True,
                                            load=True,
                                            display_scores=True,
                                            testname = testname,
                                            just_scores=True,
                                              img_count=self.img_count)
        if svms_scores_exist:
            if online:
                self.scores_exist.append(True)
                svms_scores = self.actions_classifier.scores[-1]
                svms_scores = svms_scores[:, self.a2p_match]
            else:
                svms_scores = self.actions_classifier.scores
                svms_scores = svms_scores[:, self.a2p_match]
        else: #only in online mode
            '''
            if rf_exist:
                self.scores_exist.append(True)
                svms_scores = np.zeros(
                        (1,len(self.train_classes))) + 1 / float(
                            len(self.train_classes))
            else:
            '''
            self.scores_exist.append(False)
            return False
        if online:
            rf_probs = self.poses_classifier.scores[-1]
        else:
            rf_probs = self.poses_classifier.scores
            probs_expanded = np.zeros((1+self.poses_classifier.test_sync[-1],
                                        len(self.poses)))
            probs_expanded[:] = np.nan
            probs_expanded[self.poses_classifier.test_sync, :
                                        ] = rf_probs
            rf_probs = probs_expanded
        if not online:
            self.test_ground_truth = self.construct_ground_truth(data, ground_truth_type)
        self.run_mixer(rf_probs, svms_scores, save=save, online=online)

def main():
    from matplotlib import pyplot as plt
    mixedclassifier_simple = MixedClassifier(clfs.ACTIONS_CLASSIFIER_SIMPLE,
                                             clfs.POSES_CLASSIFIER,
                                             add_info='without sparse coding')
    mixedclassifier_sparse = MixedClassifier(clfs.ACTIONS_CLASSIFIER_SPARSE,
                                             clfs.POSES_CLASSIFIER,
                                             add_info='with sparse coding')
    testname = 'actions'
    coherence = mixedclassifier_simple.run_training(load=True)
    #mixedclassifier.reset_online_test()
    #clfs.fake_online_testing(mixedclassifier,testname)

    mixedclassifier_simple.run_testing(co.CONST['test_' + testname],
                            ground_truth_type=co.CONST[
        'test_' + testname + '_ground_truth'],
        online=False, load=True)

    mixedclassifier_simple.plot_coherence()
    mixedclassifier_simple.visualize_scores()
    plt.show()

LOG = logging.getLogger('__name__')
CH = logging.StreamHandler(sys.stderr)
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)

if __name__ == '__main__':
    # signal.signal(signal.SIGINT, signal_handler)
    main()
