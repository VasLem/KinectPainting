
import os
import sys
import errno
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



class EnhancedDynamicClassifier(clfs.Classifier):
    '''
    CLassifier that combines a specified SVMs classifier <dynamic_classifier>,
    that classifies dynamic actions in timespace, and a Random Forest one, <passive_classifier>,
    that classifies passive actions. During training phase, a confusion matrix
    between random forest predictions and provided ground truth of actions
    (static and dynamic) is constructed, named 'coherence matrix'.
     The combination is done using a mixture Bayesian-deduced model, which is
    explained in detail in <run_testing> .It uses both classifiers,
    to produce a better estimation, which is useful in realtime,
    where the SVMs classifier can not cope well due to the increased
    features dimensionality.
    '''

    def __init__(self, dynamic_classifier=None, passive_classifier=None,
                 log_lev='INFO', visualize=False, separated_actions=True,
                 add_info=None, show_all_actions=True, *args, **kwargs):
        # matrix rows are poses labels`
        # matrix columns are actions labels
        # matrix entries are coherence probabilities
        if dynamic_classifier is None or passive_classifier is None:
            raise Exception(self.__doc__)
        self.passive_actions_classifier = passive_classifier  # passive gestures classifier
        self.dynamic_actions_classifier = dynamic_classifier  # dynamic gestures classifier
        if show_all_actions:
            use = 'In Sync'
            action_type = 'All'
        else:
            use = 'CDBIMM'
            action_type = 'Dynamic'
        clfs.Classifier.__init__(self, log_lev=log_lev, visualize=visualize,
                                 features=str(co.dict_oper.create_sorted_dict_view({'Cl_{pas}':
                                            str(passive_classifier.classifier_id),
                                            'Cl_{dyn}':
                                           str(dynamic_classifier.classifier_id)})),
                                 masks_needed=False,
                                 buffer_size=self.dynamic_actions_classifier.buffer_size,
                                 action_type=action_type, use=use, name='actions',
                                 add_info=add_info, *args, **kwargs)
        self.parameters['sub_classifiers'] = [
            'Cl$_{\\textup{pas}}$',
            'CDBIMM']
        self.add_train_classes(co.CONST['actions_path'])
        self.passive_actions_classifier_test = None
        self.dynamic_actions_classifier_test = None
        self.coherence_matrix = None
        self.dynamic_actions = self.dynamic_actions_classifier.train_classes
        self.passive_actions = self.passive_actions_classifier.train_classes
        _max = len(self.passive_actions_classifier.train_classes)
        if not separated_actions:
            match = [-1] * (len(self.dynamic_actions))
            for cnt, pose in enumerate(self.passive_actions):
                act_ind = self.dynamic_actions.index(pose)
                match[cnt] = act_ind
            left = list(set(range(len(match))) - set(match))
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

    def single_extract_pred_and_gt(self, action_path, action_name):
        self.passive_actions_classifier.reset_offline_test()
        actions_ground_truth = (
            self.dynamic_actions_classifier.construct_ground_truth(
                data=action_path,
                ground_truth_type='constant-' + action_name,
                classes_namespace=self.train_classes['Dynamic']))
        self.dynamic_actions_ground_truth += (
            actions_ground_truth.tolist())
        self.passive_actions_classifier.run_testing(
            data=action_path,
            online=False,
            construct_gt=False,
            save=False,
            load=not self.train_all,
            display_scores=False,
            save_results=False)
        poses_pred = self.passive_actions_classifier.recognized_classes
        self.passive_actions_predictions += poses_pred
        self.dynamic_actions_classifier.run_testing(
            data=action_path,
            online=False,
            construct_gt=False,
            save=False,
            load=not self.train_all,
            display_scores=False,
            just_scores=True,
            compute_perform=False,
            save_results=False)
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
        ground_truths = [os.path.splitext(fil)[0] for fil
                         in os.listdir(co.CONST['ground_truth_fold'])
                         if fil.endswith('.csv')]
        rosbags = [os.path.splitext(fil)[0] for fil in
                   os.listdir(co.CONST['rosbag_location'])
                   if fil.endswith('.bag')]
        to_process = [rosbag for rosbag in rosbags if rosbag in ground_truths]
        ground_truths = [os.path.join(
            co.CONST['ground_truth_fold'], name + '.csv') for name in to_process]
        rosbags = [os.path.join(
            co.CONST['rosbag_location'], name + '.bag') for name in to_process]
        self.passive_actions_predictions = []
        self.dynamic_actions_ground_truth = []
        self.dynamic_scores = []
        self.apply_to_training(self.single_extract_pred_and_gt,
                               excluded_actions=self.passive_actions)
        self.dynamic_scores = np.concatenate(
            tuple(self.dynamic_scores), axis=0)
        self.dynamic_scores = self.dynamic_scores[
            np.prod(np.isfinite(self.dynamic_scores).astype(int), axis=1).astype(bool), :]
        self.max = np.max(self.dynamic_scores, axis=0)
        self.min = np.min(self.dynamic_scores, axis=0)
        self.passive_actions_predictions = np.array(
            self.passive_actions_predictions)
        self.dynamic_actions_ground_truth = np.array(
            self.dynamic_actions_ground_truth)
        fmask = (np.isfinite(self.passive_actions_predictions) *
                 np.isfinite(self.dynamic_actions_ground_truth)) > 0
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
                     classifier_savename=None,
                     train_all=False):
        '''
        Computes coherence matrix
        Overrides <clfs.run_training>
        '''
        self.train_all = train_all
        if classifier_savename is not None:
            self.classifier_savename = classifier_savename
        if self.unified_classifier is None or not load:
            LOG.info('Training ' + self.classifier_savename)
            LOG.info('Gathering passive actions classifier traindata..')
            self.extract_pred_and_gt()
            LOG.info('Constructing coherence matrix...')
            self.construct_coherence()
            self.plot_coherence()
            if save:
                scores_params = {}
                scores_params['min'] = self.min
                scores_params['max'] = self.max
                self.parameters['scores_params'] = scores_params
                co.file_oper.save_labeled_data(['Classifier'] + self.classifier_id,
                                               ['Combined Classifier',
                                                (self.training_parameters,
                                                (self.coherence_matrix,
                                                self.min,
                                                self.max))])
        else:
             self.coherence_matrix = self.additional_params[0][0]
        if self.classifier_savename not in self.classifiers_list:
            with open('trained_classifiers_list.yaml', 'a') as out:
                out.write(self.classifier_savename + ': ' +
                          str(len(self.classifiers_list)) + '\n')
            self.classifiers_list[self.classifier_savename] = str(
                len(self.classifiers_list))

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
        # pylint: disable=no-member
        res = ax.imshow(self.coherence_matrix, cmap=plt.cm.jet,
                        interpolation='nearest')
        # pylint: enable=no-member
        for c1 in xrange(len(self.passive_actions)):
            for c2 in xrange(len(self.dynamic_actions)):
                ax.annotate(str('$%.2f$' % self.coherence_matrix[c1, c2]), xy=(c2,
                                                                               c1),
                            horizontalalignment='center',
                            verticalalignment='center', fontsize=16)

        cb = fig.colorbar(res)
        ax.set_xlabel(r'Dynamic', fontsize=16)
        ax.set_ylabel(r'Passive', fontsize=16)
        ax.set_title(r'Coherence', fontsize=18)
        plt.xticks(range(len(self.dynamic_actions)), [r'%s' % action for action in
                                                      self.dynamic_actions], rotation=45)
        plt.yticks(range(len(self.passive_actions)), [r'%s' % pose for pose in
                                                      self.passive_actions])
        save_fold = os.path.join(
            co.CONST['results_fold'], 'Classification', 'Total',)
        makedir(save_fold)
        if save:
            plt.savefig(os.path.join(
                save_fold, 'Coherence Matrix.pdf'))

    def run_mixer(self, passive_scores, dynamic_scores=None, img_count=None, save=False,
                  online=True, just_scores=False, compute_perform=True,
                  display=True,
                  *args, **kwargs):
        self.passive_scores = passive_scores.copy()
        if isinstance(passive_scores, tuple):
            dynamic_scores = passive_scores[1]
            passive_scores = passive_scores[0]
        dynamic_scores = dynamic_scores.reshape(dynamic_scores.shape[0], -1)
        if img_count is not None:
            self.img_count = img_count
        fmask_dynamic = np.prod(
            np.isfinite(np.array(dynamic_scores)),axis=1).astype(bool)
        fmask_passive = np.prod(
            np.isfinite(np.array(passive_scores)),axis=1).astype(bool)
        partial_lack = np.logical_xor(fmask_dynamic, fmask_passive)
        partial_lack_dynamic = np.logical_and(partial_lack,
                                              np.logical_not(fmask_dynamic))
        partial_lack_passive = np.logical_and(partial_lack,
                                              np.logical_not(fmask_passive))
        total = np.logical_or(fmask_dynamic, fmask_passive)
        fin_dynamic_probs = dynamic_scores[fmask_dynamic, :]
        thres = 1 / float(len(self.dynamic_actions))
        if 'SVM' in self.dynamic_actions_classifier.parameters['classifier']:
            _mins = np.min(fin_dynamic_probs, axis=1)[:, None]
            _maxs = np.max(fin_dynamic_probs, axis=1)[:, None]
            below_z = fin_dynamic_probs < 0
            fin_dynamic_probs = (thres * (below_z * fin_dynamic_probs - _mins) /
                                 (- _mins).astype(float) +
                                 (1 - thres) * ((1 - below_z) * fin_dynamic_probs /
                                                _maxs.astype(float)))
        exp_dynamic_probs = np.zeros_like(dynamic_scores)
        exp_dynamic_probs[:] = np.nan
        exp_dynamic_probs[partial_lack_dynamic, :] = thres
        exp_dynamic_probs[fmask_dynamic, :] = fin_dynamic_probs
        fin_dynamic_probs = exp_dynamic_probs[total, :]
        fin_dynamic_probs = fin_dynamic_probs.T

        if passive_scores[total, :].shape[0] == 0:
            fin_inv_passive_scores = np.zeros_like(passive_scores) + 1 / float(
                thres)
        else:
            fin_passive_scores = passive_scores[fmask_passive, :]
            fin_inv_passive_scores = np.zeros_like(fin_passive_scores)
            fin_inv_passive_scores[fin_passive_scores != 0] = 1 / fin_passive_scores[
                fin_passive_scores != 0]
            inv_passive_scores = np.zeros_like(passive_scores)
            inv_passive_scores[fmask_passive, :] = fin_inv_passive_scores
            inv_passive_scores[partial_lack_passive, :] = 1 / float(thres)
            fin_inv_passive_scores = inv_passive_scores[total, :]
        fin_inv_passive_scores = fin_inv_passive_scores.T
        fin_scores= []
        for (dyn_probs,inv_pas_probs) in zip(
            fin_dynamic_probs.T,fin_inv_passive_scores.T):
            fin_scores.append([])
            for j in range(len(self.dynamic_actions)):
                p_aj_t = dyn_probs[j]*np.sum(self.coherence_matrix[:,j]*
                                     inv_pas_probs*
                                     np.sum(self.coherence_matrix[
                                         :,:]*dyn_probs[None,:],axis=1),
                                     axis=0)

                fin_scores[-1].append(p_aj_t)
        fin_scores = np.array(fin_scores).T
        '''
        fin_scores = np.dot(self.coherence_matrix.T,
                            np.dot(self.coherence_matrix,
                                   fin_dynamic_probs)
                            * fin_inv_passive_scores) * fin_dynamic_probs
        '''
        _sum = np.sum(
            fin_scores, axis=0)
        _sum[_sum == 0] = 1
        fin_scores /= _sum.astype(float)
        '''
        from matplotlib import pyplot as plt
        fig = plt.figure()
        for c,score in enumerate(fin_scores):
            plt.plot(range(score.shape[0]),score,
                     label=self.dynamic_actions[c])
        plt.legend()
        plt.show()
        '''
        fin_scores = fin_scores.T
        # join passive predictions with updated dynamic predictions
        # the way they are joined depends on the way dynamic and passive actions
        # were exposed to ground truth constructor
        dyn_scores = np.zeros((dynamic_scores.shape))
        dyn_scores[:] = np.NaN
        dyn_scores[total, :] = fin_scores
        pas_scores = passive_scores
        self.scores = {'Passive': pas_scores, 'Dynamic': dyn_scores}
        return self.classify(just_scores, online,
                             compute_perform, display,
                             save)

    def classify(self, just_scores, online, compute_perform, display, save):
        # pylint: disable=no-member
        if not just_scores:
            if not online:
                self.recognized_classes = self.classify_offline(display=display,
                                                                compute_perform=compute_perform,
                                                                extraction_method='std_check',
                                                                tol=0.7)
                self.correlate_with_ground_truth(save=save,
                                                 display=display,
                                                 compute_perform=compute_perform)
                self.display_scores_and_time(save=save)
            else:
                self.classify_online(self.scores.ravel(),
                                     self.img_count,
                                     self.dynamic_actions_classifier.mean_from)
        # pylint: enable=no-member

            self.img_count += 1

            return self.recognized_classes, self.scores
        else:
            return self.passive_scores, self.scores

    def run_testing(self, data=None, online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=co.CONST['test_actions_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=True, testname=None, display_scores=True,
                    derot_angle=None, derot_center=None,
                    construct_gt=True, just_scores=False,
                    compute_perform=True):
        '''
        Mixed bayesian model, meant to provide unified action scores.
        P(p_i|a_j) = c_ij in Coherence Map C
        P(a_j|t) = probabilities produced by dynamic scores
        P(p_i|t) = passive gestures RF probability scores
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
        loaded = False
        if not online:
            LOG.info('Testing:' + data)
            try:
                self.test_ind = self.available_tests.index(data)
                test_name = data
                self.test_name = data
            except:
                if data.split(os.sep)[-1] in self.available_tests:
                    self.test_ind =(
                    self.available_tests.index(data.split(os.sep)[-1]))
                    test_name =data.split(os.sep)[-1]
                    self.test_name = data.split(os.sep)[-1]
                elif data in self.dynamic_actions or data in self.passive_actions:
                    self.test_ind = None
                elif data.split(os.sep)[-1] in self.dynamic_actions or \
                        data.split(os.sep)[-1] in self.passive_actions:
                    self.test_ind = None
                else:
                    raise Exception('test data must be inside test_save_path,'+
                                ' check config.yaml')
            if construct_gt and ground_truth_type is None:
                ground_truth_type=os.path.join(
                        co.CONST['ground_truth_fold'],
                        self.test_name + '.csv')
        elif isinstance(data, tuple):
            derot_angle = data[1]
            derot_center = data[2]
            data = data[0]
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
        if not online:
            if (load and self.testdata[self.available_tests.index(data)] is not None):
                LOG.info('Loading saved scores, created by '
                         + 'testing \'' + self.full_info + '\' with \'' +
                         self.testdataname + '\'')
                (passive_scores, dynamic_scores) = self.testdata[
                    self.available_tests.index( data)]
                loaded = True
        if not loaded:
            passive_exist, _ = self.passive_actions_classifier.run_testing(data=data,
                                                                           online=online,
                                                                           construct_gt=False,
                                                                           ground_truth_type=ground_truth_type,
                                                                           save=True,
                                                                           load=load,
                                                                           display_scores=False,
                                                                           testname=testname,
                                                                           just_scores=True,
                                                                           derot_angle=derot_angle,
                                                                           derot_center=derot_center,
                                                                           img_count=self.img_count)
            dynamic_exist, _ = self.dynamic_actions_classifier.run_testing(data=data,
                                                                           derot_angle=derot_angle,
                                                                           derot_center=derot_center,
                                                                           online=online,
                                                                           construct_gt=False,
                                                                           ground_truth_type=ground_truth_type,
                                                                           save=True,
                                                                           load=load,
                                                                           display_scores=False,
                                                                           testname=testname,
                                                                           just_scores=True,
                                                                           img_count=self.img_count)
            if dynamic_exist:
                if online:
                    self.scores_exist.append(True)
                    dynamic_scores = self.dynamic_actions_classifier.scores[-1]
                    if self.a2p_match is not None:
                        dynamic_scores = dynamic_scores[:, self.a2p_match]
                else:
                    dynamic_scores = self.dynamic_actions_classifier.scores
                    if self.a2p_match is not None:
                        dynamic_scores = dynamic_scores[:, self.a2p_match]
            else:  # only in online mode
                self.scores_exist.append(False)
                return None, None
            if online:
                passive_scores = self.passive_actions_classifier.scores[-1]
            else:
                passive_scores = self.passive_actions_classifier.scores
            if not online:
                if dynamic_scores.shape[0] < passive_scores.shape[0]:
                    addnan = np.zeros((-dynamic_scores.shape[0] + passive_scores.shape[0],
                                       dynamic_scores.shape[1]))
                    addnan[:] = None
                    dynamic_scores = np.concatenate(
                        (dynamic_scores, addnan), axis=0)
                elif dynamic_scores.shape[0] > passive_scores.shape[0]:
                    addnan = np.zeros((dynamic_scores.shape[0] - passive_scores.shape[0],
                                       passive_scores.shape[1]))
                    addnan[:] = None
                    passive_scores = np.concatenate(
                        (passive_scores, addnan), axis=0)
        if not online and construct_gt:
            self.test_ground_truth = {'Passive': self.construct_ground_truth(
                data, ground_truth_type=ground_truth_type,
                classes_namespace=self.train_classes['Passive']),
                'Dynamic': self.construct_ground_truth(
                    data,
                    ground_truth_type=ground_truth_type,
                    classes_namespace=
                    self.train_classes['Dynamic'])}
        if just_scores:
            if loaded:
                output = (passive_scores, dynamic_scores)
            else:
                output = self.run_mixer(passive_scores, dynamic_scores,
                                        save=save, online=online,
                                        just_scores=just_scores,
                                        compute_perform=compute_perform,
                                        display=display_scores)
        else:
            if loaded:
                output = self.classify(just_scores, online, compute_perform,
                                       display=display_scores, save=save)
            else:
                output = self.run_mixer(passive_scores, dynamic_scores,
                                        save=save, online=online,
                                        just_scores=just_scores,
                                        compute_perform=compute_perform,
                                        display=display_scores)
        if self.test_ind is not None:
            co.file_oper.save_labeled_data(['Testing']+self.tests_ids[
                self.test_ind],self.testdata[self.test_ind])
        return output
def construct_enhanced_dynamic_actions_classifier(testname='actions', train=False,
                                                  test=True, visualize=True,
                                                  dicts_retrain=False, hog_num=None,
                                                  name='actions', use_dicts=False,
                                                  des_dim=None,
                                                  test_against_all=False, train_all=False):
    '''
    Constructs a enhanced classifier
    '''
    dynamic_classifier = clfs.construct_dynamic_actions_classifier(
        train=False, test=False, visualize=False, test_against_all=False,
        features=['GHOG', '3DHOF'],use=
    'RDF',post_scores_processing_method='std_check' )
    passive_classifier = clfs.construct_passive_actions_classifier(train=False,
                                                                   test=False,
                                                                   visualize=False,
                                                                   test_against_all=False,
                                                                   features=['GHOG'],
                                                                   post_scores_processing_method=
                                                                    'prob_check')

    enhanced = EnhancedDynamicClassifier(
        dynamic_classifier=dynamic_classifier,
        passive_classifier=passive_classifier)
    enhanced.run_training(load=not train, train_all=train_all)
    if test or visualize:
        if test_against_all:
            iterat = enhanced.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                enhanced.testing_initialized = False
            enhanced.run_testing(os.path.join(
                co.CONST['test_save_path'], name),
                ground_truth_type=os.path.join(
                    co.CONST['ground_truth_fold'],
                    name + '.csv'),
                online=False, load=False)
    return enhanced
def main():
    from matplotlib import pyplot as plt
    construct_enhanced_dynamic_actions_classifier(
        train=False,
        test=False,
        visualize=True,
        test_against_all=True,
        train_all=False)
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
