'''
Implementation of Classifier Training, partly described inside Fanello et al.
'''
import sys
import signal
import errno
import glob
import numpy as np
import class_objects as co
import action_recognition_alg as ara
import cv2
import os.path
import cPickle as pickle
import logging
import yaml
import time
from OptGridSearchCV import optGridSearchCV
# pylint: disable=no-member,R0902,too-many-public-methods,too-many-arguments
# pylint: disable=too-many-locals, too-many-branches, too-many-statements


class Classifier(object):
    '''
    Class to hold all Classifier specific methods.
    <features>:['pca','ghog','3dhof']
    <action_type>:True if no buffers are used
    <use_sparse> is True if sparse coding is used
    Classifier Parameters, for example <n_estimators> and <kernel> can be
    a list, which will be reduced using optimized grid search with cross
    validation.
    '''

    def __init__(self, log_lev='INFO',
                 visualize=False, masks_needed=True,
                 buffer_size=co.CONST['buffer_size'],
                 sparse_dim=co.CONST['sparse_dim'],
                 features=None,
                 post_pca=False,
                 post_pca_components=1,
                 action_type='Dynamic',
                 use='SVM', num_of_cores=4, name='',
                 n_estimators=None,
                 add_info=None,
                 use_sparse=False,
                 kernel=None):
        # General configuration
        if features is None:
            features = ['3DXYPCA', 'GHOG', 'ZHOF', '3DHOF']
        if not isinstance(features, list):
            features = [features]
        features_params = {}
        for descriptor in features:
            features_params[descriptor] = {attrib.replace(descriptor,''):
                                           co.CONST[attrib] for
                                           attrib in co.CONST if
                                           attrib.startswith(descriptor)}
            features_params[descriptor]['sparsecoded'] = use_sparse
            if not use_sparse:
                features_params[descriptor]['sparse_params'] = None
            else:
                features_params[descriptor]['sparse_params'] = {
                    attrib.replace('sparse',''):
                    co.CONST[attrib] for
                    attrib in co.CONST if
                    attrib.startswith('sparse')}
        self.kernel = kernel
        self.n_estimators = n_estimators
        self.sparse_dim = sparse_dim
        if 'SVM' in use and kernel is None:
            self.kernel = 'linear'
        elif 'RDF' in use and n_estimators is None:
            self.n_estimators = co.CONST['RDF_trees']
        classifier_params = {'SVM_kernel': self.kernel,
                             'RDF_n_estimators': self.n_estimators}
        dynamic_params = {'buffer_size': buffer_size,
                          'buffer_confidence_tol': co.CONST['buffer_confidence_tol'],
                          'filter_window_size': co.CONST['STD_big_filt_window'],
                          'filter_window_confidence_tol':
                          co.CONST['filt_window_confidence_tol'],
                          'post_PCA': post_pca,
                          'post_PCA_components': post_pca_components}
        features_params['dynamic_params'] = dynamic_params
        if use_sparse:
            if not isinstance(sparse_dim, list):
                sparse_dim = [sparse_dim] * len(features)
            if len(list(sparse_dim)) != len(features):
                raise Exception('<sparse_dim> should be either an integer/None or' +
                                ' a list with same length with <features>')
            sparse_params = dict(zip(features, sparse_dim))
        else:
            sparse_params = None
        testing_params = {'online': None}
        fil = os.path.join(co.CONST['rosbag_location'],
                           'gestures_type.csv')
        self.passive_actions = None
        self.dynamic_actions = None
        if os.path.exists(fil):
            with open(fil, 'r') as inp:
                for line in inp:
                    if line.split(':')[0] == 'Passive':
                        self.passive_actions = line.split(
                            ':')[1].rstrip('\n').split(',')
                    elif line.split(':')[0] == 'Dynamic':
                        self.dynamic_actions = line.split(
                            ':')[1].rstrip('\n').split(',')
        LOG.debug('Extracting: ' + str(features))
        self.parameters = {'classifier': use,
                           'features': features,
                           'features_params':features_params,
                           'dynamic_params': dynamic_params,
                           'classifier_params': classifier_params,
                           'sparse_params': sparse_params,
                           'action_type': action_type,
                           'sparsecoded': use_sparse,
                           'testing': False,
                           'testing_params': testing_params,
                           'dynamic_actions': self.dynamic_actions,
                           'passive_actions': self.passive_actions}
        self.features = features
        self.add_info = add_info
        self.log_lev = log_lev
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.masks_needed = masks_needed
        self.action_type = action_type
        self.use = use
        self.num_of_cores = num_of_cores
        self.name = name
        self.post_pca = post_pca
        self.update_experiment_info()
        if 'SVM' in self.use:
            if self.kernel != 'linear':
                from sklearn.svm import SVC
                from sklearn.multiclass import OneVsRestClassifier
                self.classifier_type = OneVsRestClassifier(SVC(kernel=self.kernel),
                                                           self.num_of_cores)
            else:
                from sklearn.svm import LinearSVC
                from sklearn.multiclass import OneVsRestClassifier
                self.classifier_type = OneVsRestClassifier(LinearSVC(),
                                                           self.num_of_cores)
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.classifier_type =\
                RandomForestClassifier(10)
        self.debug = False
        self.unified_classifier = None
        self.sparsecoded = use_sparse
        self.sparse_coders = None  # is loaded from memory
        self.decide = None
        self.predict = None
        # Training variables
        self.one_v_all_traindata = None
        self.train_ground_truth = None  # is loaded from memory after training
        self.train_classes = None  # is loaded from memory after training
        self.train_inds = None
        # Testing general variables
        self.accuracy = None
        self.f1_scores = None
        self.confusion_matrix = None
        self.tester = None
        self.scores = None
        self.scores_savepath = None
        self.filtered_scores_std = []
        self.filtered_scores_std_mean = []
        self.filtered_scores = None
        self.scores_filter_shape = None
        self.std_big_filter_shape = None
        self.std_small_filter_shape = None
        self.recognized_classes = []
        self.crossings = None
        self.testname = ''
        self.save_fold = None
        self.online = False
        self.fss_max_iter = None
        # Testing offline variables
        self.testdataname = ''
        self.test_sync = None

        # Testing online variables
        self.frame_prev = None
        self.count_prev = None
        self.buffer_exists = None
        self.frame_exists = None
        self.scores_exist = None
        self.img_count = -1
        self._buffer = []
        self.scores_running_mean_vec = []
        self.big_std_running_mean_vec = []
        self.small_std_running_mean_vec = []
        self.saved_buffers_scores = []
        self.new_action_starts_count = 0
        self.test_ground_truth = None
        self.test_classes = None
        self.mean_from = -1
        self.on_action = False
        self.act_inds = []
        self.max_filtered_score = 0
        self.less_filtered_scores_std = None
        self.high_filtered_scores_std = None
        self.testdata = None
        self.classifier_folder = None
        self.classifier_savename = None
        self.testing_initialized = False
        self.classifiers_list = {}
        try:
            with open('trained_classifiers.pkl', 'r') as inp:
                self.trained_classifiers = pickle.load(inp)
        except (IOError, EOFError):
            LOG.warning('Non existent trained classifiers file')
            self.trained_classifiers = {}
        self.available_tests = os.listdir(co.CONST['test_save_path'])
        try:
            with open('all_test_scores.pkl', 'r') as inp:
                self.all_test_scores = pickle.load(inp)
        except (IOError, EOFError):
            LOG.warning('Non existent testing scores file')
            self.all_test_scores = {}
        try:
            with open('classified_dict.pkl', 'r') as inp:
                self.classified_dict = pickle.load(inp)
        except (IOError, EOFError):
            LOG.warning('Non existent classified samples file')
            self.classified_dict = {}
        try:
            with open('trained_classifiers_list.yaml', 'r') as inp:
                try:
                    self.classifiers_list = yaml.load(inp)
                except yaml.YAMLError as exc:
                    raise exc
        except (IOError, EOFError):
            pass
        # DEBUGGING
        '''
        for classifier in self.trained_classifiers.keys():
            if classifier not in self.classifiers_list:
                self.trained_classifiers.pop(classifier)
        with open('trained_classifiers.pkl','w') as out:
            pickle.dump(self.trained_classifiers, out)
        print self.trained_classifiers.keys()
        '''
        self.classifier_savename = 'trained_'
        self.classifier_savename += self.full_info.replace(' ', '_').lower()
        try:
            self.classifier_folder = str(self.classifiers_list[
                self.classifier_savename])
        except KeyError:
            self.classifier_folder = str(len(self.classifiers_list))

        self.coders_to_train = []
        if self.sparsecoded:
            self.sparse_coders = [None] * len(
                self.parameters['features'])
            # DEBUGGING
            '''
            LOG.info('Fixing names: ')
            with open('all_sparse_coders.pkl', 'r') as inp:
                self.all_sparse_coders = pickle.load(inp)
                for feat_count,feature in enumerate(
                    self.parameters['features']):
                    try:
                        self.all_sparse_coders[
                                            feature + ' ' +
                                            str(self.parameters['sparse_params'][
                                                feature]) + ' PCA ' +
                                                str(post_pca_components)] =\
                            self.all_sparse_coders.pop(
                                            feature + ' ' +
                                            str(self.parameters['sparse_params'][
                                                feature]) + ' with post pca')
                    except KeyError:
                        pass
                print self.all_sparse_coders.keys()
            with open('all_sparse_coders.pkl', 'w') as out:
                pickle.dump(self.all_sparse_coders,out)
            '''
            try:
                with open('all_sparse_coders.pkl', 'r') as inp:
                    LOG.info('Loading coders from: ' +
                             'all_sparse_coders.pkl')
                    self.all_sparse_coders = pickle.load(inp)
                    for coder in self.all_sparse_coders.keys():
                        try:
                            if (self.all_sparse_coders[coder] is None
                                or (self.all_sparse_coders[coder].codebook is
                                None and
                                self.all_sparse_coders[coder].bmat is None)):
                                self.all_sparse_coders.pop(coder)
                        except:
                            pass
                    LOG.info('Available sparse coders:' +
                             str(self.all_sparse_coders.keys()))
                    for feat_count, feature in enumerate(
                            self.parameters['features']):
                        try:
                            if self.post_pca and not self.action_type=='Passive':
                                self.sparse_coders[feat_count] =\
                                    self.all_sparse_coders[
                                        feature + ' ' +
                                        str(self.parameters['sparse_params'][
                                            feature]) + ' PCA ' +
                                        str(post_pca_components)]
                            else:
                                self.sparse_coders[feat_count] =\
                                    self.all_sparse_coders[
                                        feature + ' ' +
                                        str(self.parameters['sparse_params'][
                                            feature])]
                        except KeyError:
                            self.coders_to_train.append(feat_count)
            except (IOError, EOFError):
                LOG.warning('Non existent trained sparse coders file')
                self.all_sparse_coders = {}
                self.coders_to_train = range(len(self.features))
            self.parameters['sparse_params']['trained_coders'] = len(
                self.coders_to_train) == 0
        else:
            self.sparse_coders = None
        # parameters bound variables
        self.features_extraction = ara.FeatureExtraction(self.parameters)
        self.action_recog = ara.ActionRecognition(
            self.parameters,
            coders=self.sparse_coders,
            log_lev=log_lev)

    def update_experiment_info(self):
        if self.action_type=='Passive':
            info = 'passive '
        else:
            info = 'dynamic '
        info = info + self.name + ' ' + self.use + ' '
        info += 'using'
        if self.parameters['sparsecoded']:
            info += ' sparsecoded'
        for feature in self.parameters['features']:
            info += ' ' + feature
        info += ' features '
        if 'SVM' in self.parameters['classifier']:
            info += 'with ' + self.parameters[
                'classifier_params']['SVM_kernel'] + ' kernel'
        elif 'RDF' in self.parameters['classifier']:
            info += ('with ' + str(self.parameters['classifier_params'][
                'RDF_n_estimators']) + ' estimators')
        if self.parameters['action_type']=='Dynamic':
            info += ' with buffer size ' + str(self.buffer_size)
        if self.parameters['sparsecoded']:
            info += ' sparse dimension(s) ' + str(self.sparse_dim)
        if self.post_pca:
            info += ' with post time-pca'
        self.full_info = info.title()
        if self.add_info:
            info += self.add_info


    def apply_to_training(
            self, method, excluded_actions=None, *args, **kwargs):
        '''
        Apply a method to training data
        '''
        prev_root = ''
        prev_action = ''
        res = []
        actions = (self.passive_actions +
                   self.dynamic_actions)
        if excluded_actions is not None:
            for action in excluded_actions:
                actions.remove(action)
        paths = os.listdir(co.CONST['actions_path'])
        for action in actions:
            if action not in paths:
                actions.remove(action)
        if not actions:
            raise Exception('Badly given actions_path in config.yaml')
        dirs = [os.path.join(co.CONST['actions_path'], action) for action in
                actions]
        for direc in dirs:
            for root, dirs, _ in os.walk(direc):
                separated_root = os.path.normpath(
                    root).split(
                        os.path.sep)
                if root != prev_root and str.isdigit(
                        separated_root[-1]) and separated_root[
                            -2] != co.CONST['hnd_mk_fold_name']:
                    prev_root = root
                    if separated_root[-2] == co.CONST['mv_obj_fold_name']:
                        action = separated_root[-3]
                        action_path = (os.path.sep).join(separated_root[:-2])
                    else:
                        action = separated_root[-2]
                    if excluded_actions is not None:
                        if action in excluded_actions:
                            continue
                    if action != prev_action:
                        LOG.info('Processing action: ' + action)
                        res.append(method(action_path=action_path,
                                          action=action,
                                          *args, **kwargs))
                        prev_action = action
        try:
            return map(list, zip(*res))
        except TypeError:
            return res

    def initialize_classifier(self, classifier):
        '''
        Add type to classifier and set methods
        '''
        self.unified_classifier = classifier
        if 'SVM' in self.use:
            self.decide = self.unified_classifier.decision_function
            self.predict = self.unified_classifier.predict
        elif 'RDF' in self.use:
            self.decide = self.unified_classifier.predict_proba
            self.predict = self.unified_classifier.predict

    def reset_offline_test(self):
        '''
        Reset offline testing variables
        '''
        # Testing general variables
        self.scores = None
        self.filtered_scores_std = []
        self.filtered_scores_std_mean = []
        self.filtered_scores = None
        self.recognized_classes = []
        self.crossings = None
        self.save_fold = None
        self.testing_initialized = True
        # Testing offline variables
        self.test_sync = None

    def reset_online_test(self):
        '''
        Reset online testing variables
        '''
        # Testing general variables
        self.scores = None
        self.filtered_scores_std = []
        self.filtered_scores_std_mean = []
        self.filtered_scores = []
        self.recognized_classes = []
        self.crossings = []
        self.save_fold = None

        # Testing online variables
        self.frame_prev = None
        self.count_prev = None
        self.buffer_exists = []
        self.frame_exists = []
        self.scores_exist = []
        self.img_count = -1
        self._buffer = []
        self.scores_running_mean_vec = []
        self.big_std_running_mean_vec = []
        self.small_std_running_mean_vec = []
        self.saved_buffers_scores = []
        self.new_action_starts_count = 0
        self.test_ground_truth = None
        self.test_classes = None
        self.mean_from = -1
        self.on_action = False
        self.act_inds = []
        self.max_filtered_score = 0
        self.less_filtered_scores_std = None
        self.high_filtered_scores_std = None
        self.testdata = None
        self.testing_initialized = True

    def add_train_classes(self, training_datapath):
        '''
        Set the training classes of the classifier
        '''
        self.train_classes = [name for name in os.listdir(training_datapath)
                              if os.path.isdir(os.path.join(training_datapath, name))][::-1]
        classes = []
        if self.action_type=='Passive' or self.action_type=='All':
            if self.passive_actions is not None:
                classes += [(clas, ind) for (ind, clas) in
                            enumerate(self.passive_actions) if clas
                            in self.train_classes]
        if self.action_type=='Dynamic' or self.action_type=='All':
            if self.dynamic_actions is not None:
                classes += [(clas, ind) for (ind, clas) in
                            enumerate(self.dynamic_actions) if clas
                            in self.train_classes]
        self.train_classes, self.train_inds = map(list, zip(*classes))
        self.train_inds = np.array(self.train_inds)
        if 'Double' in self.use:
            self.train_inds[-len(self.dynamic_actions) +
                            1:] += len(self.passive_actions)

    def run_training(self, coders_retrain=False,
                     classifiers_retrain=False, test_against_training=False,
                     training_datapath=None, classifier_savename=None,
                     num_of_cores=4, buffer_size=None, classifier_save=True,
                     max_act_samples=None,
                     min_dict_iterations=5,
                     visualize_feat=False, just_sparse=False,
                     init_sc_traindata_num=200):
        '''
        <Arguments>
        For testing:
            If a testing using training data is to be used, then
            <test_against_training> is to be True
        For coders training:
            Do not train coders if coder already exists or <coders_retrain>
            is False. <min_dict_iterations> denote the minimum training iterations to
            take place after the whole data has been processed from the trainer
            of the coder.<init_dict_traindata_num> denotes how many samples
            will be used in the first iteration of the sparse coder training
        For svm training:
            Train ClassifierS with <num_of_cores> and buffer size <buffer_size>.
            Save them if <classifier_save> is True to <classifiers_savepath>. Do not train
            if <classifiers_savepath> already exists and <classifiers_retrain> is False.
        '''
        self.parameters['testing'] = False
        LOG.info(self.full_info + ':')
        if classifier_savename is not None:
            self.classifier_savename = classifier_savename
        if training_datapath is None:
            training_datapath = co.CONST['actions_path']
        self.add_train_classes(training_datapath)

        if self.sparsecoded:
            if coders_retrain:
                if isinstance(coders_retrain, list):
                    self.coders_to_train = coders_retrain
                else:
                    self.coders_to_train = range(
                        len(self.parameters['features']))
                self.parameters['sparse_params']['trained_coders'] = False
        if self.classifier_savename not in self.trained_classifiers:
            LOG.info('Missing trained classifier:' +
                     self.full_info)
            LOG.info('Classifier will be retrained')
            classifiers_retrain = True
        if buffer_size is not None:
            self.buffer_size = buffer_size
        if classifiers_retrain or self.coders_to_train or test_against_training:
            if isinstance(classifiers_retrain,list):
                self.coders_to_train = sorted(list(set(classifiers_retrain+
                                           self.coders_to_train)))
            if self.coders_to_train:
                max_act_samples = None
            self.prepare_training_data(training_datapath, max_act_samples,
                                       visualize_feat=visualize_feat)
        if self.sparsecoded:
            self.process_coders(min_iterations=min_dict_iterations,
                                sp_opt_max_iter=200,
                                init_traindata_num=init_sc_traindata_num,
                                incr_rate=2)
        if just_sparse:
            return

        if self.sparsecoded and self.coders_to_train and (
                classifiers_retrain or test_against_training):
            # Enters only if coders were not initially trained or had to be
            # retrained. Otherwise, sparse features are computed when
            #<Action.add_features> is called
            LOG.info('Making Sparse Features..')
            self.action_recog.actions.update_sparse_features(
                self.sparse_coders,
                max_act_samples=max_act_samples,
                fss_max_iter=self.fss_max_iter,
                last_only=False)
        self.process_training(num_of_cores, classifiers_retrain,
                              self.classifier_savename, classifier_save,
                              test_against_training)

    def prepare_training_data(self, path=None, max_act_samples=None,
                              fss_max_iter=100, visualize_feat=False):
        '''
        Read actions from the <path> and name them according to their parent
        folder name
        '''
        loaded = False
        if self.debug:
            LOG.warning('Debug is on, loading debug_train.pkl..')
            try:
                with open('debug_train.pkl','r') as inp:
                    self.action_recog = pickle.load(inp)
                loaded = True
            except (IOError, EOFError):
                LOG.warning('Non existent debug_train.pkl')
        self.fss_max_iter = fss_max_iter
        if not loaded:
            LOG.info('Adding actions..')
            for action in self.train_classes:
                if not isinstance(visualize_feat, bool):
                    try:
                        visualize = action.startswith(visualize_feat)
                    except TypeError:
                        visualize = action in visualize_feat
                else:
                    visualize = visualize_feat
                LOG.info('Action:' + action)
                self.action_recog.add_action(name=action,
                                             data=os.path.join(path, action),
                                             use_dexter=False,
                                             action_type=self.action_type,
                                             max_act_samples=max_act_samples,
                                             fss_max_iter=fss_max_iter,
                                             visualize_=visualize)
            if self.debug:
                LOG.warning('Debug is on, so the training dataset will be saved'+
                            'inside debug_train.pkl')
                with open('debug_train.pkl', 'w') as out:
                    pickle.dump(self.action_recog, out)

    def process_coders(self, coders_savepath=None, min_iterations=10,
                       sp_opt_max_iter=200,
                       init_traindata_num=200, incr_rate=2):
        '''
        Train coders using <train_act_num>-th action or load them if
            <retrain> is False and save file exists. <max_act_samples> is the
            number of samples to be sparse coded after the completion of the
            training/loading phase and defines the training data size
            of each action.
        '''
        if coders_savepath is None:
            coders_savepath = 'all_sparse_coders.pkl'
        if self.coders_to_train is not None and self.coders_to_train:
            LOG.info('Training coders..')
            self.action_recog.train_sparse_coders(
                trained_coders_list=self.sparse_coders,
                coders_to_train=self.coders_to_train,
                codebooks_dict=self.all_sparse_coders,
                min_iterations=min_iterations,
                sp_opt_max_iter=sp_opt_max_iter,
                init_traindata_num=init_traindata_num,
                coders_savepath=coders_savepath,
                incr_rate=incr_rate,
                debug=self.debug)
            self.parameters['sparse_params']['trained_coders'] = True
            for coder_ind in self.coders_to_train:
                self.sparse_coders[
                    coder_ind] = (
                        self.action_recog.sparse_helper.sparse_coders[
                            coder_ind])
        self.action_recog.sparse_helper.sparse_coders = (
            self.sparse_coders)

    def process_training(self, num_of_cores=4, retrain=False,
                         savepath=None, save=True,
                         against_training=False):
        '''
        Train (or load trained) Classifiers with number of cores num_of_cores, with buffer size (stride
            is 1) <self.buffer_size>. If <retrain> is True, Classifiers are retrained, even if
            <save_path> exists. If against_training, the training data is
            prepared, assuming that it will be used after that for testing.
        '''
        loaded = 0
        if save and savepath is None:
            raise Exception('savepath needed')
        if (against_training or retrain or
                not savepath in self.trained_classifiers):
            if against_training and (not retrain or savepath in
                                     self.trained_classifiers):
                LOG.info('Preparing Classifiers Train Data for testing..')
            else:
                LOG.info('Preparing Classifiers Train Data..')
            if not self.action_type=='Passive':
                acts_buffers = [action.retrieve_features()
                                for action in self.action_recog.actions.actions]

                acts_buffers = [np.swapaxes(buffers, 1, 2).reshape(
                    buffers.shape[0], -1) for buffers in acts_buffers]
                LOG.info('Train Data has ' + str(len(acts_buffers)) +
                         ' buffer lists. First buffer list has length ' +
                         str(len(acts_buffers[0])) +
                         ' and last buffer has shape ' +
                         str(acts_buffers[0][-1].shape))
                multiclass_traindata = acts_buffers
            else:
                multiclass_traindata = [action.retrieve_features().squeeze() for
                                        action in
                                        self.action_recog.actions.actions]

            LOG.info('Train Data has ' + str(len(multiclass_traindata)) +
                     ' training datasets for each action. Shape of first dataset is ' +
                     str(multiclass_traindata[0].shape))
            LOG.info(
                'Creating ground truth vector and concatenating remaining data..')
            self.train_ground_truth = []
            self.one_v_all_traindata = np.concatenate(
                tuple(multiclass_traindata), axis=0)
            for count, data in enumerate(multiclass_traindata):
                self.train_ground_truth.append(count *
                                               np.ones(data.shape[0]))
            self.train_ground_truth = np.concatenate(
                tuple(self.train_ground_truth), axis=0)
            LOG.info('Final TrainData to be used as input to Classifier' +
                     ' has shape ' + str(self.one_v_all_traindata.shape))
            LOG.info('Ground Truth vector to be used as input to Classifier' +
                     ' had shape ' + str(self.train_ground_truth.shape))
        if not retrain and savepath in self.trained_classifiers:
            LOG.info('Loading trained classifier: ' +
                     self.full_info)
            (self.unified_classifier,
             _,
             _, _) = self.trained_classifiers[savepath]
            self.initialize_classifier(self.unified_classifier)
            loaded = 1
        else:
            if retrain and not os.path.exists(savepath):
                LOG.info('retrain switch is True, so the Classifiers ' +
                         'are retrained')
            LOG.info(
                'Training Classifiers using ' +
                str(num_of_cores) +
                ' cores..')
            classifier_params = {elem.replace(self.use+'_',''):
                                 self.parameters['classifier_params'][elem]
                                 for elem in
                                 self.parameters['classifier_params']
                                 if elem.startswith(self.use)}
            if any([isinstance(classifier_params[elem],list)
                    for elem in classifier_params]):
                grid_search_params = classifier_params.copy()
                from sklearn.multiclass import OneVsRestClassifier
                if isinstance(self.classifier_type, OneVsRestClassifier):
                    grid_search_params={('estimator__'+key):classifier_params[key]
                                        for key in classifier_params}
                best_params, best_scores, best_estimators = optGridSearchCV(
                    self.classifier_type, self.one_v_all_traindata,
                    self.train_ground_truth, grid_search_params,n_jobs=4,
                    fold_num=3)
                best_params = best_params[-1]
                best_scores = best_scores[-1]
                best_estimator = best_estimators[-1]
                if isinstance(self.classifier_type, OneVsRestClassifier):
                    best_params={key.replace('estimator__',''):
                                        classifier_params[
                                            key.replace('estimator__','')]
                                        for key in best_params}
                classifier_params= {self.use+'_'+key:best_params[key] for key
                                  in best_params}
                print classifier_params
                self.parameters['classifier_params'].update(classifier_params)
                self.classifier_type = best_estimator
                self.update_experiment_info()

            self.initialize_classifier(self.classifier_type.fit(self.one_v_all_traindata,
                                                                self.train_ground_truth))
        if save and not loaded:
            LOG.info('Saving trained Classifiers to trained' +
                     ' classifiers dictionary with name: ' + savepath)
            if savepath not in self.classifiers_list:
                with open('trained_classifiers_list.yaml', 'a') as out:
                    out.write(savepath + ': ' +
                              str(len(self.classifiers_list)) + '\n')
                self.classifiers_list[savepath] = str(
                    len(self.classifiers_list))

            self.trained_classifiers[savepath] = (self.unified_classifier,
                                                  self.train_classes,
                                                  self.parameters,
                                                  self.train_inds)
            with open('trained_classifiers.pkl', 'w') as out:
                pickle.dump(self.trained_classifiers, out)

    def offline_testdata_processing(self, datapath):
        '''
        Offline testing data processing, using data in <datapath>.
        '''
        LOG.info('Processing test data..')
        LOG.info('Extracting features..')
        features, frame_inds = self.action_recog.add_action(
            name='test',
            data=datapath,
            for_testing=True,
            action_type=self.action_type,
            fss_max_iter=100)
        if self.action_type=='Dynamic':
            self.test_sync = frame_inds[0]
            test_buffers_start_inds = frame_inds[1]
            test_buffers_end_inds = frame_inds[2]
            testdata = np.swapaxes(features, 1, 2).reshape(
                features.shape[0], -1)
            return testdata, test_buffers_start_inds, test_buffers_end_inds
        else:
            self.test_sync = frame_inds
            testdata = features
            return testdata

    def construct_ground_truth(self, data=None, ground_truth_type=None,
                               all_actions=True):
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
                                of the directory the filenames,
                                which were saved inside
                                <..../class/num1/num2.png> with <num1>
                                and <num2> integers
                            'constant-*'(wildcard) to add the same ground
                                truth label foro all .png files inside data
                                if * is a valid action and exists
                                inside <self.train_classes>
        <data> should be either a string refering to the datapath
        of the numbered frames or a boolean list/array.
        if <testing> is True, compare ground truth classes with
            <self.train_classes> and remove classes that do not exist
            inside <self.train_classes>
        Returns <ground_truth> vectors which holds indices that refer
            to <classes> vector, which is also returned. Any item with
            no class has corresponding ground truth NaN
        '''
        if isinstance(data, basestring):
            if not os.path.exists(data):
                raise Exception(data + ' is a non existent path')
            paths = []
            for root, _, files in os.walk(data):
                root_separ = root.split(os.path.sep)
                if root_separ[-2] != co.CONST['hnd_mk_fold_name']:
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
        if ground_truth_type[-4::] == '.csv':
            try:
                with open(ground_truth_type, 'r') as inp:
                    for line in inp:
                        if '\n' in line:
                            line.replace('\n', '')
                        items = line.split(':')
                        try:
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[1].split(',')])
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[2].split(',')])
                        except (AttributeError, KeyError):
                            ground_truth_init[items[0]] = []
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[1].split(',')])
                            ground_truth_init[items[0]].append([
                                int(item) for item in items[2].split(',')])
            except (EOFError, IOError):
                raise Exception('Invalid csv file given\n' +
                                self.construct_ground_truth.__doc__)
            keys = ground_truth_init.keys()
            if not all_actions or 'Double' in self.use:
                class_match = {}
                for key in keys:
                    try:
                        class_match[key] = self.train_classes.index(key)
                    except ValueError:
                        ground_truth_init.pop(key, None)
            else:

                class_match = {}
                for key in keys:
                    try:
                        class_match[key] = self.train_inds[
                            self.train_classes.index(key)]
                    except ValueError:
                        ground_truth_init.pop(key, None)

            if not ground_truth_init:
                raise Exception(
                    'No classes found matching with training data ones')
            if not isinstance(data, basestring):
                ground_truth = np.zeros(len(data))
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
                        if item[1] not in self.train_classes:
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
                        class_match[key] = self.train_classes.index(key)
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
                class_match[action_cand] = self.train_classes.index(
                    action_cand)
            else:
                class_match[action_cand] = 0
            if action_cand in self.train_classes:
                ground_val = self.train_classes.index(action_cand)
            else:
                raise Exception('Invalid action name, it must exists in '
                                + 'self.train_classes')
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
        clas = [clas for clas in class_match]
        clas_ind = [class_match[clas] for clas in class_match]
        test_classes = [x for (_, x) in sorted(zip(clas_ind, clas), key=lambda
                        pair: pair[0])]
        return ground_truth, test_classes

    def plot_result(self, data, info=None, save=True, xlabel='Frames', ylabel='',
                    labels=None, colors=None, linewidths=None, alphas=None,
                    xticks_names=None, yticks_names=None, xticks_locs=None,
                    yticks_locs=None, markers=None, ylim=None, xlim=None,
                    display_all=False):
        '''
        <data> is a numpy array dims (n_points, n_plots),
        <labels> is a string list of dimension (n_plots)
        <colors> ditto
        '''
        import matplotlib
        from matplotlib import pyplot as plt
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams['text.latex.unicode'] = True
        # plt.style.use('seaborn-ticks')
        if len(data.shape) == 1:
            data = np.atleast_2d(data).T
        fig = plt.figure()
        axes = fig.add_subplot(111)
        if xticks_locs is not None:
            axes.set_xticks(xticks_locs, minor=True)
            axes.xaxis.grid(True, which='minor')
        if yticks_locs is not None:
            axes.set_yticks(yticks_locs, minor=True)
            axes.yaxis.grid(True, which='minor')
        if xticks_names is not None:
            plt.xticks(range(len(xticks_names)), xticks_names)
        if yticks_names is not None:
            plt.yticks(range(len(yticks_names)), yticks_names)
        if markers is None:
            markers = [','] * data.shape[1]
        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        if alphas is None:
            alphas = data.shape[1] * [1]
        while len(colors) < data.shape[1]:
            colors += [tuple(np.random.random(3))]
        if linewidths is None:
            linewidths = [1] * data.shape[1]
        if labels is not None:
            for count in range(data.shape[1]):
                axes.plot(data[:, count], label='%s' % labels[count],
                          color=colors[count],
                          linewidth=linewidths[count],
                          marker=markers[count], alpha=alphas[count])
            lgd = co.plot_oper.put_legend_outside_plot(axes)
        else:
            for count in range(data.shape[1]):
                axes.plot(data[:, count],
                          color=colors[count],
                          linewidth=linewidths[count],
                          marker=markers[count], alpha=alphas[count])
        if info is not None:
            plt.title(self.testname +
                      '\n Dataset: ' + self.testdataname +
                      '\n' + info.title())
        else:
            plt.title(self.testname +
                      '\n Dataset ' + self.testdataname)
            info = ''
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if ylim is not None:
            plt.ylim(ylim)
        if xlim is not None:
            plt.xlim(xlim)
        if save:
            if display_all:
                testname = self.action_type.lower()
                filename = os.path.join(*self.save_fold.split(os.sep)[:-1] +
                                        ['Total', testname + '.pdf'])
            else:
                if self.testname is None:
                    self.testname = (self.name + ' ' + self.use).title()
                if self.save_fold is None:
                    fold_name = str(
                        self.classifiers_list[self.classifier_savename])
                    self.save_fold = os.path.join(
                        co.CONST['results_fold'], 'Classification', fold_name)
                    if self.add_info is not None:
                        self.save_fold = os.path.join(
                            self.save_fold, self.add_info.replace(' ', '_').lower())
                    self.save_fold = os.path.join(
                        self.save_fold, self.testdataname)
                    co.makedir(self.save_fold)

                if info is not None:
                    filename = os.path.join(
                        self.save_fold, self.testname + ' ' + info + '.pdf')
                else:
                    filename = os.path.join(
                        self.save_fold, self.testname + '.pdf')
            if labels is None:
                plt.savefig(filename)
            else:
                plt.savefig(filename,
                            bbox_extra_artists=(lgd,), bbox_inches='tight')

    def init_testing(self, data=None, online=True, save=True, load=True,
                     testname=None, scores_savepath=None,
                     scores_filter_shape=5,
                     std_small_filter_shape=co.CONST['STD_small_filt_window'],
                     std_big_filter_shape=co.CONST['STD_big_filt_window'],
                     testdatapath=None):
        '''
        Initializes paths and names used in testing to save, load and visualize
        data.
        Built as a convenience method, in case <self.run_testing> gets overriden.
        '''
        self.parameters['testing'] = True
        self.parameters['testing_params']['online'] = online
        self.tester = ara.Action(
            self.parameters,
            name='test',
            coders=self.sparse_coders)
        if online:
            self.reset_online_test()
        else:
            self.reset_offline_test()
        self.scores_filter_shape = scores_filter_shape
        self.std_small_filter_shape = std_small_filter_shape
        self.std_big_filter_shape = std_big_filter_shape
        self.online = online
        if testname is not None:
            self.testname = testname.title()
        else:
            self.testname = (self.name + ' ' + self.use).title()
        if self.add_info is not None:
            self.testname += ' ' + self.add_info.title()
        if online:
            if testdatapath is not None:
                self.testdataname = ('online (using '
                                     + os.path.basename(testdatapath) + ')')
            else:
                self.testdataname = 'online'
        else:
            self.testdataname = os.path.basename(data)
        if save or load:
            if self.classifier_savename not in self.trained_classifiers:
                LOG.warning('No trained classifier:' + self.classifier_savename +
                            ', not proceeding to testing')
                return False
            fold_name = self.classifier_folder
            self.save_fold = os.path.join(
                co.CONST['results_fold'], 'Classification', fold_name)
            if self.add_info is not None:
                self.save_fold = os.path.join(
                    self.save_fold, self.add_info.replace(' ', '_').lower())
            self.save_fold = os.path.join(self.save_fold, self.testdataname)
            co.makedir(self.save_fold)

            if scores_savepath is None:
                self.scores_savepath = self.testdataname + '_scores_for_'
                self.scores_savepath += self.full_info.replace(' ',
                                                               '_').lower()
                self.scores_savepath += '.pkl'
            else:
                self.scores_savepath = scores_savepath
        return True

    def run_testing(self, data=None, derot_angle=None, derot_center=None,
                    online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=co.CONST['test_actions_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=False, testname=None, display_scores=True,
                    construct_gt=True, just_scores=False, testdatapath=None,
                    compute_perform=True, features_given=False):
        '''
        Test Classifiers using data (.png files) located in <data>. If <online>, the
        testing is online, with <data> being a numpy array, which has been
        firstly processed by <hand_segmentation_alg>. If <against_training>,
        the testing happens using the concatenated training data, so <data> is
        not used. The scores retrieved from testing are filtered using a
        box filter of shape <box_filter_shape>. The running mean along a buffer
        of the data is computed with a running window of length
        <mean_filter_shape>. The ground truth for the testing data is given by
        <ground_truth_type> (for further info about the variable refer to
        <self.construct_ground_truth>). If the training is online, the count of
        the frame is passed by <img_count>. If <save> is True,
        testing results are saved to <scores_savepath>, or a path constructed
        by the configuration. <testname> overrides the first line of the plots.
        If <load> is True and <scores_save_path> exists, testing is bypassed and all the
        necessary results are loaded from memory. If <just_scores> is True, the
        classification stage is not done and only scores are computed. If
        <testdatapath> is not <None> and <online> is True, then it will be
        assumed that a pseudoonline testing is taking place
        '''
        loaded = False
        if not online:
            LOG.info('Testing:' + data)
        if isinstance(data, tuple):
            derot_angle = data[1]
            derot_center = data[2]
            data = data[0]
        if not self.testing_initialized or not online:
            if not self.init_testing(data=data,
                                     online=online,
                                     save=save,
                                     load=load,
                                     testname=testname,
                                     scores_savepath=scores_savepath,
                                     scores_filter_shape=scores_filter_shape,
                                     std_small_filter_shape=std_small_filter_shape,
                                     std_big_filter_shape=std_big_filter_shape,
                                     testdatapath=testdatapath):
                return False
        if not online:
            if (load and (self.classifier_savename in self.all_test_scores)
                    and (self.testdataname in
                         self.all_test_scores[self.classifier_savename])):
                LOG.info('Loading saved scores, created by '
                         + 'testing \'' + self.full_info + '\' with \'' +
                         self.testdataname + '\'')
                (self.scores, self.test_sync) = self.all_test_scores[self.classifier_savename][
                    self.testdataname]
                loaded = True
            if not loaded:
                if 'SVM' in self.use:
                    LOG.info('Classifier contains ' +
                             str(len(self.unified_classifier.estimators_)) + ' estimators')
                if against_training:
                    LOG.info('Testing Classifiers against training data..')
                    self.scores = self.decide(
                        self.one_v_all_traindata)
                else:
                    if not features_given:
                        if not self.action_type=='Passive':
                            (testdata,
                             test_buffers_start_inds,
                             test_buffers_end_inds) = self.offline_testdata_processing(
                                 data)
                        else:
                            testdata = self.offline_testdata_processing(
                                data)
                    else:
                        testdata = data
                    self.testdata = testdata
                    LOG.info(self.full_info + ':')
                    LOG.info('Testing Classifiers using testdata with size: '
                             + str(testdata.shape))
                    self.scores = self.decide(
                        testdata)
                    if not self.action_type=='Passive':
                        expanded_scores = np.zeros((self.test_sync[-1] + 1,
                                                    self.scores.shape[1]))
                        expanded_scores[:] = np.NaN
                        for score, start, end in zip(self.scores,
                                                     test_buffers_start_inds,
                                                     test_buffers_end_inds):
                            expanded_scores[start:end + 1, :] = score[None, :]
                        self.scores = expanded_scores
                    if save and not loaded:
                        LOG.info(
                            'Saving scores to all scores dictionary')
                        try:
                            self.all_test_scores[self.classifier_savename][
                                self.testdataname] = (self.scores,
                                                      self.test_sync)
                        except (KeyError, TypeError):
                            self.all_test_scores[self.classifier_savename] = {}
                            self.all_test_scores[self.classifier_savename][
                                self.testdataname] = (self.scores,
                                                      self.test_sync)
                        with open('all_test_scores.pkl', 'w') as out:
                            pickle.dump(self.all_test_scores, out)
            if construct_gt:
                LOG.info('Constructing ground truth vector..')
                self.test_ground_truth, self.test_classes = self.construct_ground_truth(
                    data, ground_truth_type)
            if not just_scores:
                self.classify_offline(save=save, display=display_scores,
                                      compute_perform=compute_perform)
                if display_scores:
                    self.display_scores_and_time(save=save)
            return True, self.scores
        else:
            '''
            input is processed from hand_segmentation_alg (any data
            processed in such way, that the result is the same with my processing,
            is acceptable, eg. Dexter)
            There must be a continuous data streaming (method called in every
            loop), even if the result of the previous algorithm is None
            '''
            scores_exist, score = self.process_online_data(data, img_count,
                                                           derot_angle,
                                                           derot_center,
                                                           just_scores=just_scores)
            return scores_exist, score

    def display_scores_and_time(self, save=False):
        '''
        Displays scores and elapsed time
        '''
        if self.online:
            self.plot_result(np.array(self.filtered_scores),
                             labels=self.train_classes,
                             xlabel='Frames',
                             save=save)
            LOG.info(self.name.title())
        time_oper = co.TimeOperations()
        if (self.action_recog.actions.
                features_extract is not None):
            if len(self.action_recog.actions.preproc_time) > 0:
                time_oper.compute_stats(self.action_recog.actions.preproc_time,
                                        'Preprocessing')
        try:
            if (len(
                    self.action_recog.actions.sparse_time) > 0 and
                    self.sparsecoded):
                sparse_time = np.array(self.action_recog.actions.sparse_time)
                if not (sparse_time.shape[0] <
                        sparse_time.shape[1]):
                    sparse_time = sparse_time.T

                time_oper.compute_stats(sparse_time,
                                        label=['Sparse Coding '
                                         + feature for feature in
                                         self.parameters['features']])
        except TypeError:
            pass
        if len(
                self.action_recog.actions.extract_time) > 0:
            time_oper.compute_stats(self.action_recog.actions.extract_time,
                                    'Feature Extraction')


    def process_online_data(self, data, img_count=None,
                            derot_angle=None, derot_center=None,
                            just_scores=False):
        '''
        <data> is the frame with frame number <img_count> or increasing by one
        relatively to the previous frame. Scores are filtered with a filter of
        length <self.scores_filter_shape>. <self.std_small_filter_shape> is the shape
        of the filter used to remove the temporal noise from the scores std.
        <self.std_big_filter_shape> is the shape of the filter to compute the mean
        of the scores std. Returns True if scores have been computed
        '''
        self.img_count += 1
        self.mean_from += 1
        # self.buffer_exists = self.buffer_exists[
        #    -self.std_big_filter_shape:]
        if not self.action_type=='Passive':
            if not self.img_count or (img_count == 0):
                self._buffer = []
                self.mean_from = 0
                self.buffer_exists = []
                self.frame_exists = []
                self.scores = []
                self.scores_exist = []
                self.filtered_scores = []
                self.filtered_scores_std_mean = []
                self.filtered_scores_std = []
                self.small_std_running_mean_vec = []
                self.big_std_running_mean_vec = []
                self.scores_running_mean_vec = []
                self.act_inds = []
                self.crossings = []
                self.count_prev = self.img_count - 1
            if img_count is not None:
                self.frame_exists += ((img_count - self.img_count) * [False])
                self.scores_exist += ((img_count - self.img_count) * [False])
                self.mean_from = img_count - self.img_count + self.mean_from
                self.img_count = img_count
        elif not self.img_count:
            self.scores_exist = []
            self.frame_exists = []
            self.scores = []
            self.filtered_scores = []
        if data is None:
            self.frame_exists.append(False)
            self.scores_exist.append(False)
            return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                                        np.float64)
        if self.frame_prev is None:
            self.frame_prev = data.copy()
        if not self.features_extraction.update(data,
                                               angle=derot_angle,
                                               center=derot_center,
                                               masks_needed=True,
                                               img_count=self.img_count,
                                               isderotated=False):
            return False, np.array([None] * len(self.train_classes))
        features = self.features_extraction.extract_features()
        valid = False
        if features is not None:
            valid, _ = self.tester.add_features_group(
                self.img_count, features=features)
        if not valid or features is None:
            self.scores_exist.append(False)
            return False, np.array([None] * len(self.train_classes))
        else:
            self.scores_exist.append(True)
        # needs fixing for not passive
        inp = self.tester.retrieve_features()
        inp = inp[0, ...]
        inp = inp.T.reshape(1, -1)
        score = (self.decide(inp))
        self.scores.append(score)
        if not just_scores:
            self.classify_online(score, self.img_count,
                                 self.mean_from)
        else:
            self.filtered_scores.append(score)
        return True, np.array(score).reshape(1, -1)

    def classify_online(self, score, img_count, mean_from):
        '''
        To be used after scores from <online_processing_data> have been
        computed. It is a convenience function to allow the modification of
        the scores, if this is wanted, before performing classification
        '''
        if self.action_type=='Passive':
            self.scores_filter_shape = 3
        if len(self.scores_running_mean_vec) < self.scores_filter_shape:
            self.scores_running_mean_vec.append(score.ravel())
        else:
            self.scores_running_mean_vec = (self.scores_running_mean_vec[1:]
                                            + [score.ravel()])
        self.filtered_scores.append(
            np.mean(np.array(self.scores_running_mean_vec), axis=0))
        if not self.action_type=='Passive':
            start_from = min(self.scores_filter_shape, mean_from)
            '''
            self.filtered_scores.append(
                np.mean(np.array(self.scores_running_mean_vec), axis=0))
            '''
            score_std = np.std(self.filtered_scores[-1])
            if len(self.small_std_running_mean_vec) < self.std_small_filter_shape:
                self.small_std_running_mean_vec.append(score_std)
            else:
                self.small_std_running_mean_vec = (
                    self.small_std_running_mean_vec[1:] +
                    [score_std])
            filtered_score_std = np.mean(self.small_std_running_mean_vec)
            self.filtered_scores_std.append(filtered_score_std)
            if len(self.big_std_running_mean_vec) < self.std_big_filter_shape:
                self.big_std_running_mean_vec.append(filtered_score_std)
            else:
                self.big_std_running_mean_vec = (self.big_std_running_mean_vec[1:]
                                                 + [filtered_score_std])
            if mean_from >= self.std_big_filter_shape:
                start_from = 0
            else:
                start_from = - mean_from
            self.filtered_scores_std_mean.append(
                np.mean(self.big_std_running_mean_vec[-start_from:]))
            std_mean_diff = self.filtered_scores_std_mean[
                -1] - self.filtered_scores_std[-1]
            if (np.min(std_mean_diff) > co.CONST['action_separation_thres'] and not
                    self.on_action) or not self.recognized_classes:
                self.crossings.append(img_count)
                self.on_action = True
                if self.recognized_classes:
                    self.recognized_classes[-1].add(length=img_count -
                                                    self.new_action_starts_count +
                                                    1)
                    LOG.info('Frame ' + str(img_count) + ': ' +
                             self.recognized_classes[-1].name +
                             ', starting from frame ' +
                             str(self.recognized_classes[-1].start) +
                             ' with length ' +
                             str(self.recognized_classes[-1].length))
                self.recognized_classes.append(ClassObject(self.train_classes))
                index = np.argmax(self.filtered_scores[-1])
                self.max_filtered_score = self.filtered_scores[-1][index]
                self.act_inds = [index]
                self.new_action_starts_count = img_count
                self.recognized_classes[-1].add(
                    index=index,
                    start=self.new_action_starts_count)
                LOG.info('Frame ' + str(img_count) + ': ' +
                         self.recognized_classes[-1].name)
                self.saved_buffers_scores = []
                if len(self.recognized_classes) >= 2:
                    return self.recognized_classes[-2]
                return None
            else:
                if len(self.recognized_classes) > 0:
                    _arg = np.argmax(self.filtered_scores[-1])
                    if self.max_filtered_score < self.filtered_scores[
                            -1][_arg]:
                        self.max_filtered_score = self.filtered_scores[
                            -1][_arg]
                        self.recognized_classes[-1].add(index=_arg)
                        LOG.info('Frame ' + str(img_count) + ': ' +
                                 self.recognized_classes[-1].name)
                    if std_mean_diff < co.CONST['action_separation_thres']:
                        self.on_action = False
                    self.saved_buffers_scores.append(score)
                    return None
                else:
                    return None
        else:
            if np.max(self.filtered_scores[-1]) >= 0.6 or len(
                    self.recognized_classes) == 0:
                self.recognized_classes.append(
                    self.filtered_scores[-1].argmax())
            else:
                self.recognized_classes.append(
                    self.recognized_classes[-1])
            LOG.info('Pose detected:'
                     + self.train_classes[self.recognized_classes[-1]])
            return self.train_classes[self.recognized_classes[-1]]

    def classify_offline(self, display=True,
                         save=True, compute_perform=True):
        '''
        To be used after offline have been computed. It is a convenience
        function to allow the modification of the scores, if this is wanted,
        before performing classification.

        Process scores using stds as proposed by the paper
        '''
        if 'Double' in self.use:
            self.filtered_scores = co.noise_proc.masked_filter(self.scores,
                                                                 self.scores_filter_shape)
            '''
            recognized_svms_classes = np.zeros(self.filtered_scores.shape[0])
            recognized_svms_classes[:] = np.NaN
            fmask_svm = np.isfinite(self.filtered_scores[:,0])
            recognized_svms_classes[fmask_svm] = np.argmax(
                self.filtered_scores[fmask_svm,:], axis=1) + len(self.passive_actions)
            '''
            recognized_svms_classes = []
            for score in self.scores:
                if score[0] is None:
                    recognized_svms_classes.append(None)
                    continue
                if (np.max(score) >= 0.7 or len(
                        recognized_svms_classes) == 0 or
                        recognized_svms_classes[-1] is None):
                    recognized_svms_classes.append(score.argmax())
                else:
                    recognized_svms_classes.append(
                        recognized_svms_classes[-1])

            recognized_rf_classes = []
            for score in self.rf_probs:
                if score[0] is None:
                    recognized_rf_classes.append(None)
                    continue
                if (np.max(score) >= 0.7 or len(
                        recognized_rf_classes) == 0 or
                        recognized_rf_classes[-1] is None):
                    recognized_rf_classes.append(score.argmax())
                else:
                    recognized_rf_classes.append(
                        recognized_rf_classes[-1])
            self.recognized_classes = [np.array(recognized_rf_classes),
                                       np.array(recognized_svms_classes)
                                       + len(self.passive_actions)]
        elif not self.action_type=='Passive':
            self.filtered_scores = co.noise_proc.masked_filter(self.scores,
                                                                 self.scores_filter_shape)
            #self.filtered_scores = self.scores
            fmask = np.prod(np.isfinite(self.scores), axis=1).astype(bool)
            self.filtered_scores_std = np.zeros(self.scores.shape[0])
            self.filtered_scores_std[:] = None
            self.filtered_scores_std[fmask] = np.std(self.scores[fmask, :],
                                                     axis=1)
            self.less_filtered_scores_std = co.noise_proc.masked_filter(
                self.filtered_scores_std, self.std_small_filter_shape)

            self.high_filtered_scores_std = co.noise_proc.masked_filter(self.filtered_scores_std,
                                                                          self.std_big_filter_shape)
            positive = np.zeros(self.scores.shape[0])
            positive[:] = None
            positive[fmask] = ((self.high_filtered_scores_std -
                                self.less_filtered_scores_std)[fmask] > 0).astype(int)
            # We are interested only in finding negative to positive zero crossings,
            # because this is where std falls below its mean
            neg_to_pos_zero_crossings = np.where(positive[1:] -
                                                 positive[:-1] ==
                                                 -1)[0]
            self.crossings = neg_to_pos_zero_crossings
            interesting_crossings = np.concatenate((np.array([0]),
                                                    neg_to_pos_zero_crossings,
                                                    np.array([self.scores.shape[0]])),
                                                   axis=0)
            self.recognized_classes = []
            for cross1, cross2 in zip(interesting_crossings[
                    :-1], interesting_crossings[1:]):
                act_scores = self.filtered_scores[cross1:cross2, :]
                mask = fmask[cross1:cross2]
                if not np.any(mask):
                    act = np.zeros(cross2 - cross1)
                    act[:] = None
                    self.recognized_classes.append(act)
                    continue
                index = np.mean(
                    act_scores[mask, :], axis=0).argmax()
                '''
                index = np.median(
                    act_scores[mask, :], axis=0).argmax()
                '''
                act = index * np.ones(cross2 - cross1)
                act[np.logical_not(mask)] = None
                self.recognized_classes.append(act)
            self.recognized_classes = np.concatenate(
                tuple(self.recognized_classes), axis=0)
            if display:
                self.plot_result(np.concatenate((
                    self.less_filtered_scores_std[:, None],
                    self.high_filtered_scores_std[:, None]), axis=1),
                                 info='Filtered Scores Statistics',
                                 xlabel='Frames',
                                 labels=['STD', 'STD Mean'],
                                 colors=['r', 'g'],
                                 save=save)
                self.plot_result(self.filtered_scores,
                                 labels=self.train_classes,
                                 info='Filtered Scores',
                                 xlabel='Frames',
                                 save=save)
                mean_diff = (np.array(self.high_filtered_scores_std) -
                             np.array(self.less_filtered_scores_std))
                mean_diff = (mean_diff) / float(np.max(np.abs(mean_diff[
                    np.isfinite(mean_diff)])))
                plots = [mean_diff]
                labels = ['Filtered scores\nnormalized\nmean difference']

                if self.test_ground_truth is not None:
                    plots += [((self.test_ground_truth - np.mean(self.test_ground_truth[
                        np.isfinite(self.test_ground_truth)])) / float(
                            np.max(self.test_ground_truth[
                                np.isfinite(self.test_ground_truth)])))[:, None]]
                    labels += ['Ground Truth']
                self.plot_result(np.concatenate(plots, axis=1), labels=labels,
                                 info='Metric of actions starting and ending ' +
                                 'points', xlabel='Frames', save=save)
        else:
            # self.filtered_scores = co.noise_proc.masked_filter(self.scores,
            #                                        3)
            self.filtered_scores = self.scores
            self.recognized_classes = []
            for score in self.filtered_scores:
                if np.max(score) >= 0.7 or len(
                        self.recognized_classes) == 0:
                    self.recognized_classes.append(score.argmax())
                else:
                    self.recognized_classes.append(
                        self.recognized_classes[-1])
            self.recognized_classes = np.array(self.recognized_classes)
            #self.recognized_classes = self.scores.argmax(axis=1)
        self.correlate_with_ground_truth(save=save, display=display,
                                         compute_perform=compute_perform)
        return self.recognized_classes

    def compute_performance_measures(
            self, recognized_classes, fmask, save=True):
        '''
        Extract confusion matrix, accuracy and f scores from the test
        '''
        from sklearn import metrics
        from scipy.linalg import block_diag
        LOG.info('Computing performance measures for ' +
                 self.classifier_savename + ' with dataset:' +
                 self.testdataname)
        if 'Double' in self.use:
            y_true_pas = np.array(self.test_ground_truth[fmask[0]])
            y_true_dyn = np.array(self.test_ground_truth[fmask[1]])
            pas_mask = y_true_pas < len(self.passive_actions)
            dyn_mask = y_true_dyn >= len(self.passive_actions)
            y_pred_pas = recognized_classes[0][fmask[0]][pas_mask]
            y_pred_dyn = recognized_classes[1][fmask[1]][dyn_mask]
            separated = [[y_true_pas[pas_mask], y_pred_pas],
                         [y_true_dyn[dyn_mask], y_pred_dyn]]
            self.f1_scores = np.concatenate(
                tuple([np.atleast_2d(metrics.f1_score(*tuple(inp), average=None))
                       for inp in separated]), axis=1)
            self.confusion_matrix = block_diag(*tuple([
                metrics.confusion_matrix(*tuple(inp)) for
                inp in separated]))
            self.accuracy = [metrics.accuracy_score(*tuple(inp))
                             for inp in separated]
            self.accuracy = ((self.accuracy[0] * len(self.passive_actions) +
                              self.accuracy[1] * len(self.dynamic_actions)) /
                             (len(self.passive_actions) +
                              len(self.dynamic_actions)))
        else:
            y_true = np.array(self.test_ground_truth[fmask]).astype(int)
            y_pred = np.array(recognized_classes[fmask]).astype(int)
            self.f1_scores = metrics.f1_score(y_true, y_pred, average=None)
            self.confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
            self.accuracy = metrics.accuracy_score(y_true, y_pred)
        existing_classes = np.array(self.train_classes)[
            np.array(list(
                set(np.unique(y_true)).union(
                    set(np.unique(y_pred)))))]
        LOG.info(existing_classes)
        LOG.info('F1 Scores: \n' + np.array2string(self.f1_scores))
        LOG.info(
            'Confusion Matrix: \n' +
            np.array2string(
                self.confusion_matrix))
        LOG.info('Accuracy: ' + str(self.accuracy))
        labels = existing_classes
        LOG.info(self.save_fold)
        if save and self.save_fold is not None:
            with open(os.path.join(self.save_fold, 'f1_scores.tex'), 'w') as out:
                out.write(co.latex.array_transcribe([self.f1_scores,
                                                     np.atleast_2d(self.accuracy)],
                                                    xlabels=np.concatenate((labels,
                                                                            ['Accuracy']), axis=0),
                                                    sup_x_label='F-Scores',
                                                    extra_locs=['right']))
            with open(os.path.join(self.save_fold,
                                   'Confusion_Matrix.tex'), 'w') as out:
                out.write(co.latex.array_transcribe(self.confusion_matrix,
                                                    ylabels=labels,
                                                    xlabels=labels,
                                                    sup_x_label='Predicted',
                                                    sup_y_label='Actual'))

    def construct_classifiers_matrix(self):
        '''
        Constructs a table which shows most parameters of the trained
        classifiers and saves it as a pdf inside Total Results folder
        '''
        from textwrap import TextWrapper
        wrapper = TextWrapper(width=15, break_long_words=False,
                              break_on_hyphens=False, replace_whitespace=False)
        all_parameters = [(self.trained_classifiers[name][2], self.classifiers_list[name])
                          for name in self.trained_classifiers if name in
                          self.classifiers_list]
        all_parameters = sorted(all_parameters, key=lambda pair: pair[1])
        params_rows = []
        for parameters in all_parameters:
            row = []
            row.append(parameters[1])
            row.append(parameters[0]['classifier'])
            row.append('\n'.join(
                parameters[0]['features']))
            row.append(parameters[0]['sparsecoded'])
            if parameters[0]['sparsecoded']:
                row.append('\n'.join(
                    ['%d' % parameters[0]['sparse_params'][feature]
                     for feature in parameters[0]['features']]))
            else:
                row.append('')
            row.append(parameters[0]['passive'])
            if (not parameters[0]['passive'] or
                    'Double' in parameters[0]['classifier']):
                '''
                if parameters[0]['classifier']=='Double':
                    row.append('%d'%parameters[0]['classifier_params'][
                    'RDF_n_estimators'])
                else:
                    row.append('')
                '''
                row.append('')
                row.append(str(parameters[0]['classifier_params']['SVM_kernel']))
                try:
                    row.append('%d' % parameters[0]['dynamic_params'][
                        'buffer_size'])
                except:
                    row.append('')
                try:
                    row.append('%d' % parameters[0]['dynamic_params'][
                        'filter_window_size'])
                except:
                    row.append('')

                try:
                    row.append(str(parameters[0]['dynamic_params'][
                        'post_PCA']))
                except:
                    row.append('')
                if row[-1] != '' and row[-1] == 'True':
                    try:
                        row.append('%d' % parameters[0]['dynamic_params'][
                            'post_PCA_components'])
                    except:
                        row.append('')

                else:
                    row.append('')
            else:
                row.append('%d' % parameters[0]['classifier_params'][
                    'RDF_n_estimators'])

                [row.append('') for count in range(5)]
            params_rows.append(row)
        params_valids = np.tile(
            (np.array(params_rows) != '')[..., None], (1, 1, 3))
        params_valids = params_valids * 0.5
        params_valids += 0.5
        params_rows = [[wrapper.fill(el) if
                        isinstance(el, basestring) else el
                        for el in row] for row in params_rows]
        params_col_names = ('Classifier', 'Type', 'Features', 'Sparse',
                            'Sparse Features\ndimension',
                            'Actions', 'Estimators', 'Kernel',
                            'Buffer size', 'Filter size', 'post PCA',
                            'post PCA\ncomponents')
        all_results = [(self.classified_dict[item],
                        self.classifiers_list[item])
                       for item in self.classified_dict if item in
                       self.classifiers_list]
        results_rows = []
        for results in all_results:
            row = []
            row.append(str(results[1]))
            mean = 0
            for test in self.available_tests:
                try:
                    row.append('%1.2f' % results[0][test][1][0])
                except KeyError:
                    row.append('')
                mean += results[0][test][1][0]
            row.append('%1.2f' % (mean / float(len(results[0]))))
            results_rows.append(row)
        if results_rows:
            results_col_names = ['Classifier'] + \
                self.available_tests + ['Mean']
            results_valids = np.tile(
                (np.array(results_rows) != '')[..., None], (1, 1, 3))
        results_rows = sorted(results_rows, key=lambda pair:
                              float(pair[-1]), reverse=True)
        from matplotlib.backends.backend_pdf import PdfPages
        from matplotlib import pyplot as plt
        import datetime
        #matplotlib.rcParams.update({'font.size': 22})
        if self.save_fold is None:
            save_fold = os.path.join(
                co.CONST['results_fold'], 'Classification', 'Total')
            co.makedir(save_fold)
            filename = os.path.join(save_fold,
                                    'matrices.pdf')
        else:

            filename = os.path.join(*self.save_fold.split(os.sep)[:-1] +
                                    ['Total', 'matrices.pdf'])
        with PdfPages(filename) as pdf:
            fig = plt.figure()
            axes = fig.add_subplot(111)
            params_table = co.table_oper.construct(axes, cellText=params_rows,
                                                   colLabels=params_col_names,
                                                   cellColours=params_valids,
                                                   cellLoc='center',
                                                   loc='center', usetex=True)
            # params_table.auto_set_font_size(False)
            params_table.scale(2, 2)
            # params_table.set_fontsize(10)
            co.table_oper.fit_cells_to_content(
                fig, params_table, equal_height=True)
            plt.axis('off')
            pdf.savefig(
                bbox_extra_artists=(
                    params_table,
                ),
                bbox_inches='tight')
            plt.close()
            if results_rows:
                fig = plt.figure()
                axes = fig.add_subplot(111)
                results_table = co.table_oper.construct(
                    axes,
                    cellText=results_rows,
                    colLabels=results_col_names,
                    cellColours=results_valids,
                    cellLoc='center',
                    loc='center')
                co.table_oper.fit_cells_to_content(fig,
                                                         results_table,
                                                         equal_height=True)
                results_table.scale(2, 2)
                plt.axis('off')
                pdf.savefig(
                    bbox_extra_artists=(
                        results_table,
                    ),
                    bbox_inches='tight')
                plt.close()
            idd = pdf.infodict()
            idd['Author'] = u'Vassilis Lemonidis'
            idd['Subject'] = 'Unified Comparative View'
            idd['Keywords'] = 'PdfPages multipage keywords author title subject'
            idd['CreationDate'] = datetime.datetime.today()

    def correlate_with_ground_truth(self, save=True, display=False,
                                    display_all=False, compute_perform=True):
        '''
        Plot results with title <title>
        <display_all> if a plot of all the classifiers results is wanted
        '''
        fmask = None
        if self.scores_exist is not None:
            self.scores_exist = np.array(self.scores_exist)
            expanded_recognized_classes = np.zeros(self.scores_exist.size)
            expanded_recognized_classes[:] = None
            for clas in self.recognized_classes:
                expanded_recognized_classes[clas.start:clas.start + clas.length + 1][
                    self.scores_exist[clas.start:clas.start + clas.length + 1]] = clas.index
            self.crossings = np.array(self.crossings)
            if self.test_ground_truth is not None:
                fmask = self.scores_exist * np.isfinite(self.test_ground_truth)
        elif (self.test_ground_truth is not None and self.recognized_classes is
              not None):

            if self.action_type=='Passive':
                recognized_classes_expanded = np.zeros_like(
                    self.test_ground_truth)
                recognized_classes_expanded[:] = np.nan
                recognized_classes_expanded[
                    self.test_sync] = self.recognized_classes[
                        np.isfinite(self.recognized_classes)]
                expanded_recognized_classes = recognized_classes_expanded
                fmask = \
                    np.isfinite(self.test_ground_truth) * np.isfinite(
                        recognized_classes_expanded)
            else:
                expanded_recognized_classes = self.recognized_classes
                fmask = (np.isfinite(self.recognized_classes) * np.isfinite(
                    self.test_ground_truth)).astype(bool)
        if fmask is not None and compute_perform:
            self.compute_performance_measures(expanded_recognized_classes,
                                              fmask, save=save)
            while True:
                try:
                    self.classified_dict[self.classifier_savename][
                        self.testdataname] = (expanded_recognized_classes,
                                              [self.accuracy, self.f1_scores,
                                               self.confusion_matrix], self.action_type)
                    break
                except (AttributeError, IndexError, KeyError):
                    self.classified_dict[self.classifier_savename] = {}
            with open('classified_dict.pkl', 'w') as out:
                pickle.dump(self.classified_dict, out)
        if save:
            display = True

        if display:
            if display_all:
                self.construct_classifiers_matrix()

                iterat = [self.classified_dict[name][self.testdataname][:-1]
                          for name in self.classifiers_list
                          if
                          self.classified_dict[name][self.testdataname][2]
                          == self.action_type and ('Double' not in name) and
                          ('Mixed' not in name)]
                name_iterat = [int(self.classifiers_list[name]) for name
                               in self.classifiers_list if
                               self.classified_dict[name][self.testdataname][2]
                               == self.action_type and ('Double' not in name) and
                               ('Mixed' not in name)]
                # sort iterat based on the index of classifier inside
                #   classifiers_list
                iterat = [x for (_, x) in sorted(zip(name_iterat, iterat),
                                                 key=lambda pair: pair[0])]
                # provide argsort using accuracy measures, to alter line width
                higher_acc = sorted(range(len(iterat)), key=lambda
                    l: l[1][0], reverse=True)
            else:
                try:
                    iterat = [self.classified_dict[self.classifier_savename][
                        self.testdataname][0:1]]
                    name_iterat = [
                        self.classifiers_list[
                            self.classifier_savename]]
                    higher_acc = [0]
                except KeyError as err:
                    LOG.warning(
                        str(err) + ' is missing from the tested datasets')
                    return False
            plots = []
            linewidths = []
            labels = []
            markers = []
            alphas = []
            xticks = None
            if self.test_ground_truth is not None:
                plots.append(self.test_ground_truth)
                labels.append('Ground Truth')
                linewidths.append(1.6)
                markers.append(',')
                alphas.append(1)
            if self.crossings is not None:
                xticks = self.crossings
                expanded_xticks = np.zeros_like(self.test_ground_truth)
                expanded_xticks[:] = None
                expanded_xticks[xticks] = 0
                plots.append(expanded_xticks)
                alphas.append(1)
                markers.append('o')
                labels.append('Actions\nbreak-\npoints')
                linewidths.append(1)
            width = 1
            dec_q = 0.3
            min_q = 0.2
            for count in range(len(higher_acc)):
                if 'Double' in self.use:
                    plots.append(iterat[count][0][0])
                    labels.append('RF Results')
                    markers.append(',')
                    linewidths.append(width)
                    alphas.append(0.8)
                    plots.append(iterat[count][0][1])
                    labels.append('SVM Results')
                else:
                    plots.append(iterat[count][0])
                    labels.append('Classifier ' + str(name_iterat[count]))
                markers.append(',')
                linewidths.append(width)
                alphas.append(0.8)
                width -= dec_q
                width = max(min_q, width)
            yticks = self.train_classes
            ylim = (-1, len(self.train_classes) + 1)
            self.plot_result(np.vstack(plots).T, labels=labels,
                             xticks_locs=xticks, ylim=ylim,
                             yticks_names=yticks,
                             info='Classification Results',
                             markers=markers,
                             linewidths=linewidths,
                             alphas=alphas,
                             xlabel='Frames', save=save)
            return True


class ClassObject(object):
    '''
    Class to hold classification classes
    '''

    def __init__(self, class_names):
        self.name = ''
        self.index = 0
        self.start = 0
        self.length = 0
        self.names = class_names

    def add(self, index=None, start=None, length=None):
        '''
        Add Class with name, corresponding index in self.train_classes,
            starting from start frame, with length length frames
        '''
        if index is not None:
            self.index = int(index)
            self.name = self.names[int(index)]
        if start is not None:
            self.start = start
        if length is not None:
            self.length = length


def fake_online_testing(classifier, testname='actions', path=None):
    '''
    Immitate online testing for performance testing and debugging reasons
    '''
    if path is None:
        path = os.path.join(co.CONST['test_' + testname], '0')
    filenames = glob.glob(os.path.join(path, '*.png'))
    sync = [int(filter(str.isdigit,
                       os.path.basename(filename)))
            for filename in filenames]
    [sync, filenames] = map(list, zip(*[[y, x] for (y, x) in
                                        sorted(zip(sync, filenames),
                                               key=lambda pair: pair[0])]))
    txts = glob.glob(os.path.join(path, '*.txt'))
    angles = []
    centers = []
    for txt in txts:
        if 'angles' in txt:
            with open(txt, 'r') as inpf:
                angles += map(float, inpf)
        elif 'centers' in txt:
            with open(txt, 'r') as inpf:
                for line in inpf:
                    center = [
                        float(num) for num
                        in line.split(' ')]
                    centers += [center]
    for count, filename in enumerate(filenames):
        img = cv2.imread(filename, -1)
        # DEBUGGING
        #cv2.imshow('test', (img % 255).astype(np.uint8))
        # cv2.waitKey(10)
        classifier.run_testing(
            data=img,
            img_count=sync[count],
            online=True,
            load=False,
            derot_angle=angles[count],
            derot_center=centers[count],
            testdatapath=co.CONST['test_' + testname])
    classifier.recognized_classes[-1].add(length=max(sync))
    if len(classifier.scores_exist) > 1:
        classifier.test_ground_truth = classifier.construct_ground_truth(
            classifier.scores_exist, ground_truth_type=co.CONST['test_'
                                                                + testname
                                                                + '_ground_truth'])
        classifier.filtered_scores = np.array(
            classifier.filtered_scores).squeeze()
        classifier.scores_exist = np.array(classifier.scores_exist)
        expanded_scores = np.zeros(
            (len(classifier.scores_exist), classifier.filtered_scores.shape[1]))
        expanded_scores[:] = np.NaN
        expanded_scores[
            classifier.scores_exist.astype(bool),
            :] = classifier.filtered_scores
    else:
        expanded_scores = np.array(classifier.filtered_scores)
    classifier.plot_result(expanded_scores,
                           labels=['%s' % classifier.train_classes[count]
                                   for count in
                                   range(expanded_scores.shape[1])],
                           info='Filtered Scores',
                           xlabel='Frames', save=True)
    classifier.plot_result(np.concatenate((
        classifier.filtered_scores_std,
        classifier.filtered_scores_std_mean), axis=0).T,
                           colors=['r', 'g'],
                           labels=['STD', 'STD Mean'],
                           info='Filtered Scores Statistics',
                           xlabel='Frames', save=True)


def signal_handler(sig, frame):
    '''
    Signal handler for CTRL-C interrupt (SIGINT)
    '''
    LOG.info('\nGot SIGINT')
    LOG.info('Exiting...')
    loc = locals()
    running_classifiers = [loc[key] for key in loc
                           if isinstance(loc[key], Classifier)
                           and loc[key].testing_initialized and
                           loc[key].online]
    for classifier in running_classifiers:
        classifier.display_scores_and_time()
    if running_classifiers:
        from matplotlib import pyplot as plt
        plt.show()
    sys.exit(0)




def construct_dynamic_actions_classifier(testname='test2', train=False,
                                         test=True, visualize=True,
                                         coders_retrain=False,
                                         name='actions', use_sparse=False,
                                         sparse_dim=None, test_against_all=False,
                                         visualize_feat=False, kernel=None,
                                         features='GHOG', post_pca=False,
                                         post_pca_components=1,
                                         just_sparse=False,
                                         debug=False,
                                         use='SVM',
                                         action_type='Dynamic'):
    '''
    Constructs an SVM classifier with input 3DHOF and GHOG features
    '''
    if use_sparse:
        if sparse_dim is None:
            sparse_dim = 256
    classifier = Classifier('INFO', action_type=action_type,
                            name=name, sparse_dim=sparse_dim,
                            use_sparse=use_sparse,
                            features=features,
                            kernel=kernel, post_pca=post_pca,
                            post_pca_components=post_pca_components,
                            use=use)
    if debug:
        classifier.debug = True
    classifier.run_training(classifiers_retrain=train,
                            coders_retrain=coders_retrain,
                            visualize_feat=visualize_feat,
                            just_sparse=just_sparse,
                            #init_sc_traindata_num=5000,
                            init_sc_traindata_num=15000,
                            min_dict_iterations=20)
    if test or visualize:
        if test_against_all:
            iterat = classifier.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                classifier.run_testing(os.path.join(
                    co.CONST['test_save_path'], name),
                                       ground_truth_type=os.path.join(
                                           co.CONST['ground_truth_fold'],
                                           name + '.csv'),
                                       online=False, load=False)
            else:
                classifier.run_testing(os.path.join(
                    co.CONST['test_save_path'], name),
                                       ground_truth_type=os.path.join(
                                           co.CONST['ground_truth_fold'],
                                           name + '.csv'),
                                       online=False, load=False)
    return classifier


def construct_passive_actions_classifier(testname='test2',
                                         train=True, test=True, visualize=True,
                                         test_against_all=False,
                                         features='3DXYPCA'):
    '''
    Constructs a random forests passive_actions classifier with input 3DXYPCA features
    '''
    classifier = Classifier('INFO', action_type='Passive',
                            name='actions', use='RDF',
                            use_sparse=False,
                            features=features)
    classifier.run_training(classifiers_retrain=train,
                            max_act_samples=2000)
    if test or visualize:
        if test_against_all:
            iterat = classifier.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                classifier.run_testing(os.path.join(
                    co.CONST['test_save_path'], name),
                                       ground_truth_type=os.path.join(
                                           co.CONST['ground_truth_fold'],
                                           name + '.csv'),
                                       online=False, load=False)
            else:
                classifier.run_testing(os.path.join(
                    co.CONST['test_save_path'], name),
                                       ground_truth_type=os.path.join(
                                           co.CONST['ground_truth_fold'],
                                           name + '.csv'),
                                       online=False, load=False)

    return classifier


def construct_total_statistics():
    '''
    Construct unified plots for all classifiers, tested on all tests
    '''
    classifier = Classifier()
    classifier.action_type = 'Dynamic'
    iterat = classifier.available_tests
    classifier.scores_exist = None
    classifier.recognized_classes = None
    for test in iterat:
        classifier.test_ground_truth = classifier.construct_ground_truth(
            ground_truth_type=os.path.join(co.CONST['test_save_path'], test))
        classifier.correlate_with_ground_truth(save=True,
                                               display=True,
                                               display_all=True,
                                               compute_perform=True)


def main():
    '''
    Example Usage
    '''
    from matplotlib import pyplot as plt
    testname = 'actions'
    # construct_passive_actions_classifier(test)
    # plt.show()
    '''
    TRAIN_ALL_SPARSE = construct_dynamic_actions_classifier(
        train=[0],features=['GHOG', 'ZHOF', '3DHOF', '3DXYPCA'],
        use_sparse=True,just_sparse=True, debug=True)
    '''
    '''
    TRAIN_ALL_SPARSE_WITH_POST_PCA = construct_dynamic_actions_classifier(
        train=False,features=['GHOG', 'ZHOF', '3DHOF', '3DXYPCA'],
        post_pca=True,
        use_sparse=True,just_sparse=True,post_pca_components=4)
    '''
    POSES_CLASSIFIER = construct_passive_actions_classifier(train=True,
                                                            test=True,
                                                            visualize=True,
                                                            test_against_all=True)
    ACTIONS_CLASSIFIER_SIMPLE = construct_dynamic_actions_classifier(
        train=True,
        test=True,
        visualize=True,
        test_against_all=True)
    ACTIONS_CLASSIFIER_SPARSE = construct_dynamic_actions_classifier(train=True,
                                                                     coders_retrain=False,
                                                                     test=True,
                                                                     visualize=True,
                                                                     test_against_all=True,
                                                                     use_sparse=True)
    ACTIONS_CLASSIFIER_SIMPLE_POST_PCA = construct_dynamic_actions_classifier(
        train=True,
        test=True,
        visualize=True,
        test_against_all=True,
        post_pca=True,
        post_pca_components=2)

    ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','3DHOF'], kernel='linear')
    ACTIONS_CLASSIFIER_SPARSE_WITH_3DHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','3DHOF'], coders_retrain=False, use_sparse=True,
        kernel='linear')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF_POST_PCA = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','3DHOF'],post_pca=True, post_pca_components=4)
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','3DHOF'],post_pca=True,use_sparse=True,
        post_pca_components=4)
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=False, test=True, visualize=True, test_against_all=True,
        features=['GHOG','ZHOF'])
    #Let's try RDF for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','3DHOF'],post_pca=False,use_sparse=False,
        use='RDF')

    #Let's try RDF with all features for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','3DHOF','3DXYPCA'],post_pca=False,use_sparse=False,
        use='RDF')

    # Let's try RDF for all features for all actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','3DHOF','3DXYPCA'], action_type='All', post_pca=False,use_sparse=False,
        use='RDF')
    ACTIONS_CLASSIFIER_SPARSE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=False,#debugging
        test=True, visualize=True, test_against_all=True,
        features=['GHOG','ZHOF'], coders_retrain=False, use_sparse=True,
        kernel='linear')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF_POST_PCA = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','ZHOF'],post_pca=True, post_pca_components=4)
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','ZHOF'],post_pca=True,
        use_sparse=True, coders_retrain=False, post_pca_components=4)
    '''
    ACTIONS_CLASSIFIER_SPARSE_WITH_ZHOF_RBF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','ZHOF'], coders_retrain=False, use_sparse=True,
        kernel='rbf')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF_RBF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG','ZHOF'], coders_retrain=False, use_sparse=False,
        kernel='rbf')
    ACTIONS_CLASSIFIER_SIMPLE_RBF = construct_dynamic_actions_classifier(
        train=False,
        test=False,
        visualize=False,
        use_sparse=False,
        test_against_all=True,
        kernel='rbf')
    '''
    '''
    #construct classifiers comparative table
    '''
    tmp = Classifier(features=[''])
    tmp.construct_classifiers_matrix()
    sys.exit(0)

    #    visualize_feat=True)
    #    visualize_feat=['Fingerwave in'])

LOG = logging.getLogger('__name__')
CH = logging.StreamHandler(sys.stderr)
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)
if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
