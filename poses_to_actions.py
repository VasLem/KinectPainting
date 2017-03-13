import os, sys, errno
import numpy as np
import classifiers as clfs
import class_objects as co
import cPickle as pickle
import logging
from numpy.linalg import inv


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class MixedClassifier(clfs.Classifier):
    def __init__(self, svms_classifier=None,
                 rf_classifier=None,log_lev='INFO',
                 visualize=False, add_info=None, *args,**kwargs):
        clfs.Classifier.__init__(self,log_lev=log_lev, visualize=visualize,
                                 masks_needed=False,
                 ispassive = False, use='Mixed', name='',
                                add_info=add_info, *args, **kwargs)
        from sklearn.ensemble import RandomForestClassifier
        self.classifier = RandomForestClassifier(self.num_of_estimators)
        self.enhanced_dyn = EnhancedDynamicClassifier(svms_classifier,
                                                      rf_classifier)
        self.train_classes = np.hstack((self.passive_actions,
                                        self.dynamic_actions))
        self.classifier_savename = 'trained_'
        self.classifier_savename += self.full_info.replace(' ', '_').lower()
    def single_run_training(self,action_path, action):
            traindata = np.concatenate(
                self.enhanced_dyn.run_testing(action_path,
                                              online=False,
                                              just_scores=True),axis=1)
            ground_truth,_ = (self.construct_ground_truth(
                data=action_path,
                ground_truth_type='constant-' + action))
            return traindata, ground_truth

    def run_training(self, save=True, load=True,
                     classifier_savename=None):
        if classifier_savename is not None:
            self.classifier_savename = classifier_savename
        if (self.classifier_savename not in
            self.trained_classifiers or not load):
            traindata,ground_truth = self.apply_to_training(self.single_run_training)
            fmask = np.isfinite(traindata[:,0])
            self.classifier = self.classifier.fit(np.concatenate(traindata[fmask,:],axis=0),
                                np.concatenate(ground_truth[fmask,:],axis=0))
            self.trained_classifiers[self.classifier_savename] = self.classifier
            with open('trained_classifiers.pkl','w') as out:
                pickle.dump(self.trained_classifiers, out)
        else:
            self.classifier = self.trained_classifiers[
                self.classifier_savename]

    def run_testing(self, data=None, online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=co.CONST['test_actions_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=False, testname=None, display_scores=True,
                    derot_angle=None, derot_center=None,
                    construct_gt=True, just_scores=False):
        testdata =np.hstack(
            self.enhanced_dyn.run_testing(data,online=online,just_scores=True))
        self.scores = self.classifier.predict_proba(testdata)
        recognized_classes = []
        for score in self.scores:
            if np.sum(score) is None:
                recognized_classes.append(None)
                continue
            if (np.max(score) >= 0 or len(
                recognized_classes) == 0 or
                recognized_classes[-1] is None):
                recognized_classes.append(score.argmax())
            else:
                recognized_classes.append(
                    recognized_classes[-1])
        if not online:
            self.test_ground_truth = self.construct_ground_truth(
                data=data,
                ground_truth_type=ground_truth_type)
        self.correlate_with_ground_truth(save=save,
                                         display=display_scores)



class EnhancedDynamicClassifier(clfs.Classifier):
    '''
    CLassifier that combines a specified SVMs classifier <svms_classifier>,
    that classifies all actions in timespace, and a Random Forest one, <rf_classifier>,
    that classifies static actions. During training phase, a confusion matrix
    between random forest predictions and provided ground truth of actions
    (static and dynamic) is constructed, named 'coherence matrix'.
     The combination is done using a mixture Bayesian-deduced model, which is
    explained in detail in <run_testing> .It uses both classifiers,
    to produce a better estimation, which is useful in realtime,
    where the SVMs classifier can not cope well due to the increased
    features dimensionality.
    '''
    def __init__(self, svms_classifier=None, rf_classifier=None,
                 log_lev='INFO',visualize=False, separated_actions=True,
                 add_info=None, *args, **kwargs):
        # matrix rows are poses labels`
        # matrix columns are actions labels
        # matrix entries are coherence probabilities
        if svms_classifier is None or rf_classifier is None:
            raise Exception(self.__doc__)
        self.passive_actions_classifier = rf_classifier  # poses classifier
        self.dynamic_actions_classifier = svms_classifier  # actions classifier
        clfs.Classifier.__init__(self,log_lev=log_lev, visualize=visualize,
                                 masks_needed=False,
                 buffer_size = self.dynamic_actions_classifier.buffer_size,
                 ispassive = False, use='Double', name='actions',
                                add_info=add_info, *args, **kwargs)
        self.add_train_classes(co.CONST['actions_path'])
        self.passive_actions_classifier_test = None
        self.dynamic_actions_classifier_test = None
        self.coherence_matrix = None
        self.dynamic_actions = self.dynamic_actions_classifier.train_classes
        self.passive_actions = self.passive_actions_classifier.train_classes
        _max = len(self.passive_actions_classifier.train_classes)
        if not separated_actions:
            match = [-1] * (len(self.dynamic_actions))
            for cnt,pose in enumerate(self.passive_actions):
                act_ind = self.dynamic_actions.index(pose)
                match[cnt] = act_ind
            left = list(set(range(len(match)))-set(match))
            for cnt in range(len(match)):
                if match[cnt] == -1:
                    match[cnt] = left[0]
                    left = left[1:]
            self.a2p_match = np.array(match)
            self.dynamic_actions = np.array(
                self.dynamic_actions)[self.a2p_match].tolist()
        else:
            self.a2p_match = None
        # actions ground truth2
        self.passive_actions_predictions = None
        self.dynamic_actions_ground_truth = None
        self.classifier_savename = 'trained_'
        self.classifier_savename += self.full_info.replace(' ', '_').lower()


    def single_extract_pred_and_gt(self,action_path, action):
        self.passive_actions_classifier.reset_offline_test()
        actions_ground_truth,_ = (
            self.dynamic_actions_classifier.construct_ground_truth(
                data=action_path,
                ground_truth_type='constant-' + action))
        self.dynamic_actions_ground_truth += (
            actions_ground_truth.tolist())
        self.passive_actions_classifier.run_testing(
            data=action_path,
            online=False,
            construct_gt=False,
            save=False,
            load=False,
            display_scores=False)
        poses_pred = self.passive_actions_classifier.recognized_classes
        poses_predictions_expanded = np.zeros(
            np.max(self.passive_actions_classifier.test_sync) + 1)
        poses_predictions_expanded[:] = None
        poses_predictions_expanded[self.passive_actions_classifier.test_sync
                           ] = poses_pred
        poses_pred = poses_predictions_expanded
        self.passive_actions_predictions += poses_pred.tolist()
        self.dynamic_actions_classifier.run_testing(
            data=action_path,
            online=False,
            construct_gt=False,
            save=False,
            load=False,
            display_scores=False)
        self.dynamic_scores.append(self.dynamic_actions_classifier.scores)
        return None

    def extract_pred_and_gt(self):
        '''
        This function loads from a <csv_pathname> the actions ground truth and
        tests <self.passive_actions_classifier> using images inside <set_pathname>,
        extracting preficted poses.
        <set_pathname> is the pathname of the folder of png images refering
        to the <csv_pathname>, which holds the corresponding actions
        ground truth. So both <set_pathname> and <csv_pathname> should be
        correlated.
        The actions ground truth contains only actions that exist already
        inside <self.dynamic_actions_classifier>. If this is not desired, set
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
        self.passive_actions_predictions = []
        self.dynamic_actions_ground_truth = []
        self.dynamic_scores = []
        self.apply_to_training(self.single_extract_pred_and_gt,
                               excluded_actions=self.passive_actions)
        self.dynamic_scores = np.concatenate(tuple(self.dynamic_scores),axis=0)
        self.dynamic_scores = self.dynamic_scores[
            np.prod(np.isfinite(self.dynamic_scores).astype(int),axis=1).astype(bool),:]
        self.max = np.max(self.dynamic_scores, axis=0)
        self.min = np.min(self.dynamic_scores, axis=0)
        self.passive_actions_predictions = np.array(self.passive_actions_predictions)
        self.dynamic_actions_ground_truth = np.array(self.dynamic_actions_ground_truth)
        fmask = (np.isfinite(self.passive_actions_predictions) *
                 np.isfinite(self.dynamic_actions_ground_truth))>0
        self.passive_actions_predictions = self.passive_actions_predictions[
            fmask].astype(int)
        self.dynamic_actions_ground_truth = self.dynamic_actions_ground_truth[
            fmask].astype(int)
        self.passive_actions_classifier.reset_offline_test()

    def construct_coherence(self):
        '''
        Coherence matrix is a matrix which has as many rows as the poses
        labels and as many columns as the actions labels. Each row sums to 1
        and describes the coherence between the corresponding pose and the
        action labels.
        This function constructs actions to pose <coherence_matrix>.
        <extract_variables> should be run first
        '''
        if self.dynamic_actions is None:
            raise Exception(self.construct_coherence.__doc__)
        self.coherence_matrix = np.zeros((len(self.passive_actions),
                                          len(self.dynamic_actions)))
        for action_truth, pose_pred in zip(self.dynamic_actions_ground_truth,
                                           self.passive_actions_predictions):
            self.coherence_matrix[pose_pred,
                                  action_truth] += 1
        if self.a2p_match is not None:
            self.coherence_matrix = self.coherence_matrix[
                :, self.a2p_match]
        #self.dynamic_actions = [self.dynamic_actions[cnt] for cnt in self.a2p_match]
        self.coherence_matrix = self.coherence_matrix / np.sum(
            self.coherence_matrix, axis=0)
        return self.coherence_matrix

    def run_training(self, save=True, load=True,
                     classifier_savename=None):
        '''
        Computes coherence matrix
        Overrides <clfs.run_training>
        '''
        if classifier_savename is not None:
            self.classifier_savename = classifier_savename
        if not self.classifier_savename in self.trained_classifiers or not load:
            LOG.info('Training ' + self.classifier_savename)
            self.extract_pred_and_gt()
            self.construct_coherence()
            self.plot_coherence()
            if save:
                self.trained_classifiers[self.classifier_savename] = (
                    self.coherence_matrix, self.min, self.max,
                    self.passive_actions,self.dynamic_actions)
            with open('trained_classifiers.pkl', 'w') as out:
                pickle.dump(self.trained_classifiers, out)
        else:
            (self.coherence_matrix,
             self.min,
             self.max,
             self.passive_actions,
             self.dynamic_actions) = self.trained_classifiers[self.classifier_savename]
        if self.classifier_savename not in self.classifiers_list:
            with open('trained_classifiers_list.yaml','a') as out:
                out.write(self.classifier_savename+': '+
                          str(len(self.classifiers_list))+'\n')
            self.classifiers_list[self.classifier_savename] = str(len(self.classifiers_list))

        return self.coherence_matrix, self.passive_actions, self.dynamic_actions

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
        for c1 in xrange(len(self.passive_actions)):
            for c2 in xrange(len(self.dynamic_actions)):
                ax.annotate(str('$%.2f$' % self.coherence_matrix[c1,c2]), xy=(c2,
                                                                            c1),
                            horizontalalignment='center',
                            verticalalignment='center',fontsize=16)

        cb = fig.colorbar(res)
        ax.set_xlabel(r'Actions', fontsize=16)
        ax.set_ylabel(r'Poses', fontsize=16)
        ax.set_title(r'Coherence', fontsize=18)
        plt.xticks(range(len(self.dynamic_actions)), [r'%s' % action for action in
                                              self.dynamic_actions], rotation = 45)
        plt.yticks(range(len(self.passive_actions)), [r'%s' % pose for pose in
                                            self.passive_actions])
        save_fold = os.path.join(
            co.CONST['results_fold'], 'Classification', 'Total',)
        makedir(save_fold)
        if save:
            plt.savefig(os.path.join(
                save_fold, 'Coherence Matrix.pdf'))


    def run_mixer(self, rf_probs, svms_scores=None, img_count=None, save=False,
                  online=True,scores_only=False,*args,**kwargs):
        self.rf_probs = rf_probs.copy()
        if isinstance(rf_probs, tuple):
            svms_scores= rf_probs[1]
            rf_probs = rf_probs[0]
        svms_scores = svms_scores.reshape(svms_scores.shape[0],-1)
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
        thres = 1 / float(len(self.dynamic_actions))
        fin_svms_probs = (thres * (below_z * fin_svms_probs - _mins) /
                          (- _mins).astype(float) +
                          (1 - thres) *((1 - below_z)*fin_svms_probs /
                                      _maxs.astype(float)))
        '''
        fin_svms_probs = fin_svms_probs/np.sum(
            fin_svms_probs,axis=1)[:,None]
        '''
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

        fin_scores = np.dot(self.coherence_matrix.T,
                              np.dot(self.coherence_matrix,
                              fin_svms_probs)
                              * fin_inv_rf_probs) * fin_svms_probs
        fin_scores = fin_scores.T
        fin_scores = np.minimum(fin_scores, 10)
        _sum = np.sum(
            fin_scores, axis=1)[:, None]
        _sum[_sum == 0] = 1
        fin_scores /= _sum.astype(float)
        #join passive predictions with updated dynamic predictions
        #the way they are joined depends on the way dynamic and passive actions
        #were exposed to ground truth constructor
        self.scores = np.zeros((svms_scores.shape))
        self.scores[:] = np.NaN
        self.scores[total, :] = fin_scores
        #pylint: disable=no-member
        if not scores_only:
            if not online:
                self.recognized_classes = self.classify_offline(display=True)
                self.display_scores_and_time(save=save)
            else:
                self.classify_online(self.scores.ravel(),
                                     self.img_count,
                                     self.dynamic_actions_classifier.mean_from)
        #pylint: enable=no-member

            self.img_count += 1

            return self.recognized_classes, self.scores
        else:
            return self.rf_probs, self.scores



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
                self.img_count = img_count
            else:
                self.img_count += 1
        if not self.testing_initialized or not online:
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
        rf_exist,_ = self.passive_actions_classifier.run_testing(data=data,
                                          online=online,
                                          construct_gt=False,
                                          ground_truth_type=ground_truth_type,
                                         save=True,
                                         load=True,
                                         display_scores=True,
                                         testname=testname,
                                         just_scores=True,
                                         derot_angle=derot_angle,
                                             derot_center=derot_center,
                                            img_count=self.img_count)
        svms_scores_exist, _ = self.dynamic_actions_classifier.run_testing(data=data,
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
                svms_scores = self.dynamic_actions_classifier.scores[-1]
                if self.a2p_match is not None:
                    svms_scores = svms_scores[:, self.a2p_match]
            else:
                svms_scores = self.dynamic_actions_classifier.scores
                if self.a2p_match is not None:
                    svms_scores = svms_scores[:, self.a2p_match]
        else: #only in online mode
            self.scores_exist.append(False)
            return False
        if online:
            rf_probs = self.passive_actions_classifier.scores[-1]
        else:
            rf_probs = self.passive_actions_classifier.scores
            probs_expanded = np.zeros((1+self.passive_actions_classifier.test_sync[-1],
                                        len(self.passive_actions)))
            probs_expanded[:] = np.nan
            probs_expanded[self.passive_actions_classifier.test_sync, :
                                        ] = rf_probs
            rf_probs = probs_expanded
        if not online:
            self.test_ground_truth,_ = self.construct_ground_truth(data, ground_truth_type)
        if just_scores:
            rf_probs, svms_scores = self.run_mixer(rf_probs, svms_scores,
                                                   save=save, online=online,
                                                   just_scores=just_scores)
            return rf_probs, svms_scores
        else:
            recognized_classes, scores = self.run_mixer(rf_probs, svms_scores,
                                                   save=save, online=online,
                                                   just_scores=just_scores)
            return recognized_classes, scores

def main():
    from matplotlib import pyplot as plt
    construct_enhanced_dynamic_actions_classifier(
        train=True,
        test=False,
        visualize=False,
        test_against_all=True)
    construct_mixed_classifier(
        train=True,
        test=True,
        visualize=True,
        test_against_all=True)

    plt.show()


def construct_mixed_classifier(testname='actions', train=False,
                                 test=True, visualize=True,
                                 dicts_retrain=False, hog_num=None,
                                 name='actions', use_dicts=False,
                                 des_dim=None, test_against_all=False):
    '''
    Constructs a enhanced classifier
    '''
    svms_classifier = clfs.construct_dynamic_actions_classifier(
        train=False, test=False, visualize=False, test_against_all=False,
        features=['GHOG','ZHOF'])
    rf_classifier = clfs.construct_passive_actions_classifier(train=False,
                                                            test=False,
                                                            visualize=False,
                                                            test_against_all=False)

    mixed = MixedClassifier(
        svms_classifier=svms_classifier,
        rf_classifier=rf_classifier)
    mixed.run_training(load=not train)
    if test or visualize:
        if test_against_all:
            iterat = mixed.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                mixed.run_testing(os.path.join(
                    co.CONST['test_save_path'],name),
                    ground_truth_type=os.path.join(
                        co.CONST['ground_truth_fold'],
                        name+'.csv'),
                    online=False, load=False)
            else:
                mixed.run_testing(os.path.join(
                    co.CONST['test_save_path'],name),
                    ground_truth_type=os.path.join(
                        co.CONST['ground_truth_fold'],
                        name+'.csv'),
                    online=False, load=False)
    return mixed


def construct_enhanced_dynamic_actions_classifier(testname='actions', train=False,
                                 test=True, visualize=True,
                                 dicts_retrain=False, hog_num=None,
                                 name='actions', use_dicts=False,
                                 des_dim=None, test_against_all=False):
    '''
    Constructs a enhanced classifier
    '''
    svms_classifier = clfs.construct_dynamic_actions_classifier(
        train=False, test=False, visualize=False, test_against_all=False,
        features=['GHOG','ZHOF'])
    rf_classifier = clfs.construct_passive_actions_classifier(train=False,
                                                            test=False,
                                                            visualize=False,
                                                            test_against_all=False)

    enhanced = EnhancedDynamicClassifier(
        svms_classifier=svms_classifier,
        rf_classifier=rf_classifier)
    enhanced.run_training(load=not train)
    if test or visualize:
        if test_against_all:
            iterat = enhanced.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                enhanced.run_testing(os.path.join(
                    co.CONST['test_save_path'],name),
                    ground_truth_type=os.path.join(
                        co.CONST['ground_truth_fold'],
                        name+'.csv'),
                    online=False, load=False)
            else:
                enhanced.run_testing(os.path.join(
                    co.CONST['test_save_path'],name),
                    ground_truth_type=os.path.join(
                        co.CONST['ground_truth_fold'],
                        name+'.csv'),
                    online=False, load=False)
    return enhanced

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
