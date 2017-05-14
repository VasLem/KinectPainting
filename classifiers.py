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

def timeit(func):
    '''
    Decorator to time extraction
    '''
    def wrapper(self,*arg, **kw):
        t1 = time.time()
        res = func(self,*arg, **kw)
        t2 = time.time()
        self.time.append(t2-t1)
        del self.time[:-5000]
        return res
    return wrapper

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
                 sparse_dim_rat=co.CONST['sparse_dim_rat'],
                 features=None,
                 post_pca=False,
                 post_pca_components=None,
                 action_type='Dynamic',
                 use='SVM', num_of_cores=4, name='',
                 svm_c=None,
                 n_estimators=None,
                 add_info=None,
                 use_sparse=None,
                 kernel=None,
                 save_all_steps=False,
                 post_scores_processing_method=None,
                 hardcore=False):
        '''
        use_sparse = [Buffer, Features, None]
        '''
        self.time = []
        self.classifiers_ids = None
        self.test_ind = None
        # General configuration
        if not isinstance(features, list):
            features = [features]
        features = sorted(features)
        ###
        features_params = {}
        coders_params = {}
        for descriptor in features:
            features_params[descriptor] = {}
            features_params[descriptor]['params'] = {attrib.replace(descriptor, ''):
                                           co.CONST[attrib] for
                                           attrib in co.CONST if
                                           attrib.startswith(descriptor)}
            features_params[descriptor]['sparsecoded'] = use_sparse
            features_params[descriptor]['action_type'] = action_type
            coders_params[descriptor] = {}
            if not use_sparse:
                features_params[descriptor]['sparse_params'] = None
            else:
                features_params[descriptor]['sparse_params'] = {
                    attrib.replace('sparse', ''):
                    co.CONST[attrib] for
                    attrib in co.CONST if
                    attrib.startswith('sparse')}
                coders_params[descriptor] = {
                    attrib.replace('sparse', ''):
                    co.CONST[attrib] for
                    attrib in co.CONST if
                    attrib.startswith('sparse') and
                    'fss' not in attrib}
        self.kernel = kernel
        self.svm_c = svm_c
        self.n_estimators = n_estimators
        self.sparse_dim_rat = sparse_dim_rat
        if 'SVM' in use and kernel is None:
            self.kernel = 'linear'
        if 'SVM' in use:
            if svm_c is None:
                self.svm_c = co.CONST['SVM_C']
            if post_scores_processing_method == 'prob_check':
                LOG.warning('Invalid post_scores_processing_method for SVM')
                if hardcore:
                    raise Exception
                else:
                    LOG.warning('Changing method to std_check')
                    post_scores_processing_method = 'std_check'
        if 'RDF' in use:
            if svm_c is not None:
                LOG.warning('svm_c is not None for RDF experimentation')
                if hardcore:
                    raise Exception
        if post_scores_processing_method is None:
            if 'RDF' in use:
                post_scores_processing_method = 'prob_check'
            else:
                post_scores_processing_method = 'std_check'
        classifier_params = {}
        if 'RDF' in use and n_estimators is None:
            self.n_estimators = co.CONST['RDF_trees']
        if 'SVM' in use:
            classifier_params['SVM_kernel'] = self.kernel
            classifier_params['SVM_C'] = self.svm_c
        if 'RDF' in use:
            classifier_params['RDF_n_estimators'] = self.n_estimators
        if action_type != 'Passive':
            dynamic_params = {'buffer_size': buffer_size,
                              'buffer_confidence_tol': co.CONST['buffer_confidence_tol'],
                              'filter_window_size': co.CONST['STD_big_filt_window'],
                              'filter_window_confidence_tol':
                              co.CONST['filt_window_confidence_tol']}
        else:
            dynamic_params = {'buffer_size':1}
        if post_pca and post_pca_components is None:
            post_pca_components = co.CONST['PTPCA_components']
        post_pca_params = {'PTPCA_components': post_pca_components}
        for descriptor in features:
            features_params[descriptor]['dynamic_params'] = dynamic_params
        if use_sparse:
            if not isinstance(sparse_dim_rat, list):
                sparse_dim_rat = [sparse_dim_rat] * len(features)
            if len(list(sparse_dim_rat)) != len(features):
                raise Exception('<sparse_dim_rat> should be either an integer/None or' +
                                ' a list with same length with <features>')
            sparse_params = dict(zip(features, sparse_dim_rat))
            sparse_params['fss_max_iter'] = co.CONST['sparse_fss_max_iter']
        else:
            sparse_params = None

        testing_params = {'online': None}
        testing_params['post_scores_processing_method'] = \
                post_scores_processing_method
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
        action_params = {'Passive': self.passive_actions,
                         'Dynamic': self.dynamic_actions}
        LOG.debug('Extracting: ' + str(features))
        self.parameters = {'classifier': use,
                           'features': features,
                           'features_params': features_params,
                           'coders_params': coders_params,
                           'dynamic_params': dynamic_params,
                           'classifier_params': classifier_params,
                           'sparse_params': sparse_params,
                           'action_type': action_type,
                           'sparsecoded': use_sparse,
                           'testing': False,
                           'testing_params': testing_params,
                           'actions_params': action_params,
                           'PTPCA': post_pca,
                           'PTPCA_params': post_pca_params}
        self.training_parameters = {k:self.parameters[k] for k in
                                    ('classifier','features',
                                     'features_params',
                                     'dynamic_params',
                                     'classifier_params',
                                     'sparse_params',
                                     'action_type',
                                     'sparsecoded',
                                     'PTPCA',
                                     'PTPCA_params') if k in
                                    self.parameters}
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
        self.action_recog = ara.ActionRecognition(
            self.parameters,
            log_lev=log_lev)
        self.available_tests = sorted(os.listdir(co.CONST['test_save_path']))
        self.update_experiment_info()
        if 'SVM' in self.use:
            from sklearn.svm import LinearSVC
            self.classifier_type = LinearSVC(
                                       class_weight='balanced',C=self.svm_c,
                                       multi_class='ovr',
                                            dual=False)
        else:
            from sklearn.ensemble import RandomForestClassifier
            self.classifier_type =\
                RandomForestClassifier(10)
        self.unified_classifier = None
        if use_sparse:
            if not(use_sparse == 'Features' or use_sparse == 'Buffer'):
                raise Exception('Invalid use_sparse, its value shoud be '
                                + 'None/False/Buffer/Features')
        self.sparsecoded = use_sparse
        self.decide = None
        # Training variables
        self.training_data = None
        self.train_ground_truth = None  # is loaded from memory after training
        self.train_classes = None  # is loaded from memory after training
        # Testing general variables
        self.accuracy = None
        self.f1_scores = None
        self.confusion_matrix = None
        self.scores_savepath = None
        self.scores_std = []
        self.scores_std_mean = []
        self.scores = None
        self.scores_filter_shape = None
        self.std_big_filter_shape = None
        self.std_small_filter_shape = None
        self.recognized_classes = []
        self.crossings = None
        self.testname = ''
        self.save_fold = None
        self.online = False
        # Testing offline variables
        self.testdataname = ''
        self.test_instances = None
        # Testing online variables
        self.frame_prev = None
        self.count_prev = None
        self.buffer_exists = None
        self.scores_exist = None
        self.img_count = -1
        self._buffer = []
        self.scores_running_mean_vec = []
        self.big_std_running_mean_vec = []
        self.small_std_running_mean_vec = []
        self.saved_buffers_scores = []
        self.new_action_starts_count = 0
        self.test_ground_truth = None
        self.mean_from = -1
        self.on_action = False
        self.act_inds = []
        self.max_filtered_score = 0
        self.less_filtered_scores_std = None
        self.high_filtered_scores_std = None
        self.classifier_folder = None
        self.testing_initialized = False
        self.classifiers_list = {}
        self.classifier_savename = 'trained_'
        self.classifier_savename += self.full_info.replace(' ', '_').lower()
        try:
            [self.unified_classifier,
             info] = co.file_oper.load_labeled_data(
                ['Classifier'] + self.classifier_id)
            if isinstance(info,tuple):
                self.training_params = info[0]
                self.additional_params = info[1:]
            else:
                self.training_params = info
            self.loaded_classifier = True
            LOG.info('Loaded Classifier')
        except TypeError:
            self.loaded_classifier = False
            LOG.info('Classifier not Loaded')

        self.load_tests()
        try:
            self.classifier_folder = str(self.classifiers_list[
                self.classifier_savename])
        except KeyError:
            self.classifier_folder = str(len(self.classifiers_list))

        self.coders_to_train = []
        # parameters bound variables
        self.frames_preproc = ara.FramesPreprocessing(self.parameters)
        available_descriptors =\
            ara.Actions(self.parameters).available_descriptors
        try:
            self.features_extractors = [available_descriptors[nam](
                self.parameters, self.frames_preproc)
                for nam in self.parameters['features']]
            self.buffer_operators = [
                ara.BufferOperations(self.parameters)
                for nam in self.parameters['features']]
            if self.sparsecoded:
                [self.action_recog.
                 actions.load_sparse_coder(ind) for ind in range(
                     len(self.parameters['features']))]
        except:
            pass

    def load_tests(self, reset=True):
        if reset:
            self.testdata = [None] * len(self.available_tests)
            self.fscores = [None] * len(self.available_tests)
            self.accuracies = [None] * len(self.available_tests)
            self.results = [None] * len(self.available_tests)
            self.conf_mats = [None] * len(self.available_tests)
            self.test_times = [None] * len(self.available_tests)
        for count, test in enumerate(self.available_tests):
            if (self.testdata[count] is None or
                self.testdata[count]['Accuracy'] is None):
                self.testdata[count] = co.file_oper.load_labeled_data(
                    ['Testing'] + self.tests_ids[count])
            if (self.testdata[count] is not None and
                self.testdata[count]['Accuracy'] is not None):
                self.accuracies[count] = self.testdata[count]['Accuracy']
                self.fscores[count] = self.testdata[count]['FScores']
                self.results[count] = self.testdata[count]['Results']
                self.conf_mats[count] = self.testdata[count]['ConfMat']
                self.test_times[count] = self.testdata[count]['TestTime']
            else:
                self.testdata[count] = {}
                self.testdata[count]['Accuracy'] = None
                self.testdata[count]['FScores'] = None
                self.testdata[count]['Results'] = {}
                self.testdata[count]['ConfMat'] = None
                self.testdata[count]['TestTime'] = None



    def update_experiment_info(self):
        if self.parameters['action_type'] == 'Passive':
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
        if self.parameters['action_type'] == 'Dynamic':
            info += ' with buffer size ' + str(self.buffer_size)
        if self.parameters['sparsecoded']:
            info += ' with sparsecoding by ratio of ' + \
                str(self.sparse_dim_rat)
        if self.post_pca:
            info += (' with ' +
                     str(self.parameters['PTPCA_params']['PTPCA_components']) +
                     ' post-time-pca components')
        self.full_info = info.title()
        if self.add_info:
            info += self.add_info
        self.classifier_savename = 'trained_'
        self.classifier_savename += self.full_info.replace(' ', '_').lower()
        self.update_classifier_id()
        self.update_tests_ids()

    def update_classifier_id(self):
        self.features_file_id = []
        self.features_id = []
        for count in range(len(self.parameters['features'])):
            _id, file_id = self.action_recog.actions.retrieve_descriptor_possible_ids(count,
                                                                           assume_existence=True)
            self.features_id.append(_id)
            self.features_file_id.append(file_id)
            self.classifier_id = [co.dict_oper.create_sorted_dict_view(
                {'Classifier':str(self.use)}),
                                  co.dict_oper.create_sorted_dict_view(
                                      {'ClassifierParams':str(co.dict_oper.create_sorted_dict_view(
                                      self.parameters['classifier_params']))}),
                                  co.dict_oper.create_sorted_dict_view(
                                  {'ActionsType':str(self.action_type)}),
                                  co.dict_oper.create_sorted_dict_view(
                                  {'FeaturesParams':str(self.features_file_id)})]

    def update_tests_ids(self):
        self.tests_ids = []
        for count, test in enumerate(self.available_tests):
            self.tests_ids.append([co.dict_oper.create_sorted_dict_view({'Test':str(test)}),
                                   co.dict_oper.create_sorted_dict_view(
                                   {'TestingParams':str(co.dict_oper.create_sorted_dict_view(
                                       self.parameters['testing_params']))})]
                                  + [self.classifier_id])


    def initialize_classifier(self, classifier):
        '''
        Add type to classifier and set methods
        '''
        self.unified_classifier = classifier
        if 'SVM' in self.use:
            self.unified_classifier.decide = self.unified_classifier.decision_function
            self.unified_classifier.predict = self.unified_classifier.predict
        elif 'RDF' in self.use:
            self.unified_classifier.decide = self.unified_classifier.predict_proba
            self.unified_classifier.predict = self.unified_classifier.predict
        co.file_oper.save_labeled_data(['Classifier'] + self.classifier_id,
                                       [self.unified_classifier,
                                        self.training_parameters])

    def reset_offline_test(self):
        '''
        Reset offline testing variables
        '''
        # Testing general variables
        self.scores_std = []
        self.scores_std_mean = []
        self.scores = None
        self.recognized_classes = []
        self.crossings = None
        self.save_fold = None
        self.testing_initialized = True
        # Testing offline variables

    def reset_online_test(self):
        '''
        Reset online testing variables
        '''
        # Testing general variables
        self.scores_std = []
        self.scores_std_mean = []
        self.scores = []
        self.recognized_classes = []
        self.crossings = []
        self.save_fold = None

        # Testing online variables
        self.frame_prev = None
        self.count_prev = None
        self.buffer_exists = []
        self.scores_exist = []
        self.img_count = -1
        self._buffer = []
        self.scores_running_mean_vec = []
        self.big_std_running_mean_vec = []
        self.small_std_running_mean_vec = []
        self.saved_buffers_scores = []
        self.new_action_starts_count = 0
        self.test_ground_truth = None
        self.mean_from = -1
        self.on_action = False
        self.act_inds = []
        self.max_filtered_score = 0
        self.less_filtered_scores_std = None
        self.high_filtered_scores_std = None
        self.testing_initialized = True

    def add_train_classes(self, training_datapath):
        '''
        Set the training classes of the classifier
        '''
        self.train_classes = [name for name in os.listdir(training_datapath)
                              if os.path.isdir(os.path.join(training_datapath, name))][::-1]
        self.all_actions = ['Undefined'] + self.train_classes
        # Compare actions in memory with actions in file 'gestures_type.csv'
        if self.passive_actions is not None:
            passive_actions = [clas for clas in
                               (self.passive_actions) if clas
                               in self.train_classes]

        if self.dynamic_actions is not None:
            dynamic_actions = [clas for clas in
                               (self.dynamic_actions) if clas
                               in self.train_classes]
        if (self.dynamic_actions is not None and
                self.passive_actions is not None):
            if 'Sync' in self.use:
                self.train_classes = {'Passive': passive_actions,
                                      'Dynamic': dynamic_actions}
            else:
                classes = []
                if self.action_type == 'Dynamic' or self.action_type == 'All':
                    classes += dynamic_actions
                if self.action_type == 'Passive' or self.action_type == 'All':
                    classes += passive_actions
                self.train_classes = classes

    def run_training(self, coders_retrain=False,
                     classifiers_retrain=False,
                     training_datapath=None, classifier_savename=None,
                     num_of_cores=4, classifier_save=True,
                     max_act_samples=None,
                     min_dict_iterations=5,
                     visualize_feat=False, just_sparse=False,
                     init_sc_traindata_num=200):
        '''
        <Arguments>
        For coders training:
            Do not train coders if coder already exists or <coders_retrain>
            is False. <min_dict_iterations> denote the minimum training iterations to
            take place after the whole data has been processed from the trainer
            of the coder.<init_dict_traindata_num> denotes how many samples
            will be used in the first iteration of the sparse coder training
        For svm training:
            Train ClassifierS with <num_of_cores>.
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

        if self.unified_classifier is None:
            LOG.info('Missing trained classifier:' +
                     self.full_info)
            LOG.info('Classifier will be retrained')
            classifiers_retrain = True
        self.prepare_training_data(training_datapath, max_act_samples,
                                   visualize_feat=visualize_feat)
        if just_sparse:
            return

        if self.sparsecoded and self.coders_to_train and classifiers_retrain:
            # Enters only if coders were not initially trained or had to be
            # retrained. Otherwise, sparse features are computed when
            #<Action.add_features> is called
            LOG.info('Trained' + str([self.parameters['features'][coder] for coder in
                                      self.coders_to_train]))
            LOG.info('Making Sparse Features..')
            self.action_recog = ara.ActionRecognition(
                self.parameters,
                log_lev=self.log_lev,
                feat_filename=os.path.join(co.CONST['feat_save_path'],
                                           'saved'))
            self.prepare_training_data(training_datapath, max_act_samples,
                                       visualize_feat=visualize_feat)
        self.process_training(num_of_cores, classifiers_retrain,
                              self.classifier_savename, classifier_save)

    def prepare_training_data(self, path=None, max_act_samples=None,
                              visualize_feat=False):
        '''
        Read actions from the <path> and name them according to their parent
        folder name
        '''
        LOG.info('Adding actions..')
        while True:
            for act_count, action in enumerate(self.train_classes):
                if not isinstance(visualize_feat, bool):
                    try:
                        visualize = action.startswith(visualize_feat)
                    except TypeError:
                        visualize = action in visualize_feat
                else:
                    visualize = visualize_feat
                LOG.info('Action:' + action)
                features, _, trained_coders, _= self.action_recog.add_action(name=action,
                                                                                     data=os.path.join(
                                                                                         path, action),
                                                                                     use_dexter=False,
                                                                                     action_type=self.action_type,
                                                                                     max_act_samples=max_act_samples,
                                                                                     visualize_=visualize)
                if not(self.sparsecoded and None in trained_coders):
                    features = np.hstack(tuple(features))
                    features = features[np.prod(np.isfinite(
                        features),axis=1).astype(bool)]
                    LOG.info('Action \'' + action + '\' has ' +
                             'features of shape ' + str(features.shape))
                    if self.training_data is None:
                        self.training_data = features
                    else:
                        self.training_data = np.vstack((self.training_data,
                                                        features))
                    if self.train_ground_truth is None:
                        self.train_ground_truth = []
                    self.train_ground_truth += features.shape[0] * [act_count]
                else:
                    self.training_data = None
                    self.train_ground_truth = None
            if None in trained_coders and self.sparsecoded:
                self.action_recog.actions.train_sparse_dictionary()
            else:
                break
        finite_samples = np.prod(
            np.isfinite(
                self.training_data),
            axis=1).astype(bool)
        self.train_ground_truth = np.array(
            self.train_ground_truth)[finite_samples]
        self.training_data = self.training_data[finite_samples, :]
        LOG.info('Total Training Data has shape:'
                 + str(self.training_data.shape))

    def process_training(self, num_of_cores=4, retrain=False,
                         savepath=None, save=True):
        '''
        Train (or load trained) Classifiers with number of cores num_of_cores, with buffer size (stride
            is 1) <self.buffer_size>. If <retrain> is True, Classifiers are retrained, even if
            <save_path> exists.
        '''
        loaded = 0
        if save and savepath is None:
            raise Exception('savepath needed')
        if retrain or self.unified_classifier is None:
            if retrain and self.unified_classifier is not None:
                LOG.info('retrain switch is True, so the Classifier ' +
                         'is retrained')
            classifier_params = {elem.replace(self.use + '_', ''):
                                 self.parameters['classifier_params'][elem]
                                 for elem in
                                 self.parameters['classifier_params']
                                 if elem.startswith(self.use)}
            if any([isinstance(classifier_params[elem], list)
                    for elem in classifier_params]):
                grid_search_params = classifier_params.copy()
                from sklearn.multiclass import OneVsRestClassifier
                if isinstance(self.classifier_type, OneVsRestClassifier):
                    grid_search_params = {('estimator__' + key): classifier_params[key]
                                          for key in classifier_params}
                grid_search_params = {key:(grid_search_params[key] if
                                           isinstance(
                                               grid_search_params[key],list)
                                           else [
                                               grid_search_params[key]]) for key in
                                      classifier_params}
                best_params, best_scores, best_estimators = optGridSearchCV(
                    self.classifier_type, self.training_data,
                    self.train_ground_truth, grid_search_params, n_jobs=4,
                    fold_num=3)
                best_params = best_params[-1]
                best_scores = best_scores[-1]
                best_estimator = best_estimators[-1]
                if isinstance(self.classifier_type, OneVsRestClassifier):
                    best_params = {key.replace('estimator__', ''):
                                   classifier_params[
                        key.replace('estimator__', '')]
                        for key in best_params}
                classifier_params = {self.use + '_' + key: best_params[key] for key
                                     in best_params}
                self.parameters['classifier_params'].update(classifier_params)
                self.training_parameters['classifier_params'].update(classifier_params)
                self.classifier_type = best_estimator
                self.update_experiment_info()
                savepath = self.classifier_savename

            self.initialize_classifier(self.classifier_type.fit(self.training_data,
                                                                self.train_ground_truth))

    def compute_testing_time(self, testname):
        testing_time = {}
        features_extraction_time = 0
        if not self.online:
            for count in range(len(self.parameters['features'])):
                try:
                    loaded = co.file_oper.load_labeled_data(
                        [str(self.features_id[count][-1])]+
                         self.features_file_id[count] +
                         [str(testname)])
                    (_, _, feat_times) = loaded
                except:
                    return None
                for key in feat_times:
                    LOG.info('Time:' + str(key) +':'+
                             str(np.mean(feat_times[key])))
                    features_extraction_time += np.mean(feat_times[key])
            try:
                testing_time['Classification'] = self.time[
                -1] / float(self.scores.shape[0])
            except IndexError:
                testing_time['Classification'] = (
                    co.file_oper.load_labeled_data(
                        ['Testing']+self.tests_ids[
                            self.available_tests.index(
                                testname)])['TestTime'][
                                    'Classification'])
        else:
            testing_time['Classification'] = np.mean(self.time)
        testing_time['Features Extraction'] = features_extraction_time
        return testing_time

    def offline_testdata_processing(self, datapath):
        '''
        Offline testing data processing, using data in <datapath>.
        '''
        LOG.info('Processing test data..')
        LOG.info('Extracting features..')
        (features, test_name, _, _) = self.action_recog.add_action(
            name=None,
            data=datapath,
            for_testing=True,
            action_type=self.action_type)
        testdata = np.hstack(tuple(features))
        self.parameters['testing_params'][test_name] = test_name
        self.parameters['testing_params']['current'] = test_name
        return testdata

    def construct_ground_truth(self, data=None, classes_namespace=None,
                               length=None, ground_truth_type=None,
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
                                truth label for all .png files inside data
                                if * is a valid action and exists
                                inside <classes_namespace>
        <data> should be either a string refering to the datapath
        of the numbered frames or a boolean list/array.
        if <testing> is True, compare ground truth classes with
            <classes_namespace> and remove classes that do not exist
            inside <classes_namespace>
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
            class_match = {}
            for key in keys:
                try:
                    class_match[key] = classes_namespace.index(key)
                except ValueError:
                    ground_truth_init.pop(key, None)
            if not ground_truth_init:
                raise Exception(
                    'No classes found matching with training data ones')
            if not isinstance(data, basestring):
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
        return ground_truth

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
                    if not self.online:
                        fold_name = co.file_oper.load_labeled_data(['Testing'],
                                                       just_catalog=True,
                                                        include_all_catalog=True)[
                                                           str(self.tests_ids[
                                                               self.available_tests.
                                                               index(self.test_name)])]
                    else:
                        fold_name = 'Online'
                    self.save_fold = os.path.join(
                        co.CONST['results_fold'], 'Classification', fold_name,
                        self.test_name)
                    if self.add_info is not None:
                        self.save_fold = os.path.join(
                            self.save_fold, self.add_info.replace(' ', '_').lower())
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
        plt.close('all')

    def init_testing(self, data=None, online=True, save=True, load=True,
                     testname=None, scores_savepath=None,
                     scores_filter_shape=5,
                     std_small_filter_shape=co.CONST['STD_small_filt_window'],
                     std_big_filter_shape=co.CONST['STD_big_filt_window'],
                     testdatapath=None, save_results=True):
        '''
        Initializes paths and names used in testing to save, load and visualize
        data.
        Built as a convenience method, in case <self.run_testing> gets overriden.
        '''
        self.parameters['testing'] = True
        self.parameters['testing_params']['online'] = online
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
        self.parameters['testing_params']['current'] = self.testname
        if online:
            if testdatapath is not None:
                self.testdataname = ('online (using '
                                     + os.path.basename(testdatapath) + ')')
            else:
                self.testdataname = 'online'
        else:
            self.testdataname = os.path.basename(data)
        if not self.online:
            if self.test_ind is not None:
                available_tests_ids =co.file_oper.load_labeled_data(['Testing'],
                                               just_catalog=True,
                                                include_all_catalog=True)
                if available_tests_ids is None:
                    fold_name = '0'
                else:
                    curr_test_id = self.tests_ids[self.available_tests.
                                                  index(self.test_name)]
                    if str(curr_test_id) in available_tests_ids:
                        fold_name = str(available_tests_ids[str(curr_test_id)])
                    else:
                        fold_name = str(len(available_tests_ids))
        else:
            self.test_name = 'Online'

            try:
                fold_name = os.path.join(*[co.CONST['results_fold'],
                                        'Classification','Online'])
            except OSError:
                fold_name = '0'
        if self.test_ind is not None:
            self.save_fold = os.path.join(
                co.CONST['results_fold'], 'Classification', self.test_name,
                fold_name)
            co.makedir(self.save_fold)
            if save or load:
                fold_name = self.classifier_folder

                if scores_savepath is None:
                    self.scores_savepath = self.testdataname + '_scores_for_'
                    self.scores_savepath += self.full_info.replace(' ',
                                                                   '_').lower()
                    self.scores_savepath += '.pkl'
                else:
                    self.scores_savepath = scores_savepath
        return True

    def run_testing(self, data=None, derot_angle=None, derot_center=None,
                    online=True,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=None,
                    img_count=None, save=True, scores_savepath=None,
                    load=False, testname=None, display_scores=True,
                    construct_gt=True, just_scores=False, testdatapath=None,
                    compute_perform=True,
                    save_results=True):
        '''
        Test Classifiers using data (.png files) located in <data>. If <online>, the
        testing is online, with <data> being a numpy array, which has been
        firstly processed by <hand_segmentation_alg>. The scores retrieved from
        testing are filtered using a box filter of shape <box_filter_shape>.
        The running mean along a buffer
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
                                     testdatapath=testdatapath,
                                     save_results=save_results):
                return False
        if not online:
            if self.test_ind is not None and (
                load and self.accuracies[self.available_tests.index(test_name)]
                is not None):
                LOG.info('Tests already performed, loaded data')
                self.scores = self.results['scores']
                loaded = True
            if not loaded:
                if self.test_ind is not None:
                    testdata = self.offline_testdata_processing(
                        os.path.join(co.CONST['test_save_path'],
                                     test_name))
                else:
                    testdata = self.offline_testdata_processing(
                        data)

                try:
                    self.test_ind = self.available_tests.index(data)
                except:
                    self.test_ind = None
                LOG.info(self.full_info + ':')
                LOG.info('Testing Classifiers using testdata with size: '
                         + str(testdata.shape))
                fmask = np.prod(np.isfinite(testdata), axis=1).astype(bool)
                fin_scores = self.unified_classifier.decide(
                    testdata[fmask, :])
                self.scores = np.zeros(
                    (testdata.shape[0], fin_scores.shape[1]))
                self.scores[:] = None
                self.scores[fmask] = fin_scores
                if self.test_ind is not None:
                    self.testdata[self.test_ind]['Results']['Scores'] = self.scores
            if construct_gt:
                LOG.info('Constructing ground truth vector..')
                self.test_ground_truth = self.construct_ground_truth(
                    os.path.join(co.CONST['test_save_path'], test_name),
                    classes_namespace=self.train_classes,
                    length=self.scores.shape[0],
                    ground_truth_type=ground_truth_type)
            if not just_scores:
                self.classify_offline(save=save, display=display_scores,
                                      compute_perform=compute_perform,
                                      extraction_method=
                                      self.parameters['testing_params']['post_scores_processing_method'])

                self.correlate_with_ground_truth(save=save,
                                                 display=display_scores,
                                                 compute_perform=compute_perform)
            self.display_scores_and_time(save=save)
            if self.test_ind is not None:
                co.file_oper.save_labeled_data(['Testing']+self.tests_ids[
                    self.test_ind],self.testdata[self.test_ind])
            if not just_scores:
                if display_scores:
                    if self.parameters['testing_params'][
                        'post_scores_processing_method']=='std_check':
                        self.plot_result(np.concatenate((
                            self.less_filtered_scores_std[:, None],
                            self.high_filtered_scores_std[:, None]), axis=1),
                            info='Scores Statistics',
                            xlabel='Frames',
                            labels=['STD', 'STD Mean'],
                            colors=['r', 'g'],
                            save=save)
                        mean_diff = (np.array(self.high_filtered_scores_std) -
                                     np.array(self.less_filtered_scores_std))
                        mean_diff = (mean_diff) / float(np.max(np.abs(mean_diff[
                            np.isfinite(mean_diff)])))
                        plots = [mean_diff]
                        labels = ['ScoresSTD - ScoresSTDMean']

                        if self.test_ground_truth is not None:
                            plots += [((self.test_ground_truth - np.mean(self.test_ground_truth[
                                np.isfinite(self.test_ground_truth)])) / float(
                                    np.max(self.test_ground_truth[
                                        np.isfinite(self.test_ground_truth)])))[:, None]]
                            labels += ['Ground Truth']
                        self.plot_result(np.concatenate(plots, axis=1), labels=labels,
                                         info='Metric of actions starting and ending ' +
                                         'points', xlabel='Frames', save=save)
            self.plot_result(self.scores,
                             labels=self.train_classes,
                             info='Scores',
                             xlabel='Frames',
                             save=save)
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
                                          action_name=action,
                                          *args, **kwargs))
                        prev_action = action
        try:
            return map(list, zip(*res))
        except TypeError:
            return res
    def display_scores_and_time(self, save=False):
        '''
        Displays scores and elapsed time
        '''
        if self.online:
            self.plot_result(np.array(self.scores),
                             labels=self.train_classes,
                             xlabel='Frames',
                             save=save)
            LOG.info(self.name.title())
        if self.test_ind is not None:
            test_times = self.compute_testing_time(self.available_tests[
                self.test_ind])
            LOG.info('Mean Testing Times:\n\t'+str(test_times))
        if not self.online and self.test_ind is not None:
            self.testdata[self.test_ind]['TestTime'] = test_times


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
        if not self.action_type == 'Passive':
            if not self.img_count or (img_count == 0):
                self._buffer = []
                self.mean_from = 0
                self.buffer_exists = []
                self.scores = []
                self.scores_std_mean = []
                self.scores_std = []
                self.small_std_running_mean_vec = []
                self.big_std_running_mean_vec = []
                self.scores_running_mean_vec = []
                self.act_inds = []
                self.crossings = []
                self.count_prev = self.img_count - 1
            if img_count is not None:
                self.scores += [None] * (img_count - self.img_count - 1)
                self.mean_from = img_count - self.img_count + self.mean_from
                self.img_count = img_count
        elif not self.img_count:
            self.scores = []
        if data is None:
            self.scores.append(None)
            return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                                        np.float64)
        if self.frame_prev is None:
            self.frame_prev = data.copy()
        if not self.frames_preproc.update(data,
                                          angle=derot_angle,
                                          center=derot_center,
                                          masks_needed=True,
                                          img_count=self.img_count,
                                          isderotated=False):
            return False, np.array([None] * len(self.train_classes))
        features = [descriptor.extract() for
                    descriptor in self.features_extractors]
        if None not in features:
            for count in range(len(features)):
                if self.sparsecoded == 'Features':
                    features[count] = (self.action_recog.actions.
                                       coders[count].code(features[count]))
                self.buffer_operators[count].update_buffer_info(
                    self.img_count, samples=features[count])
                self.buffer_operators[count].add_buffer()
                features[count] = self.buffer_operators[count].buffer
                if features[count] is None:
                    return False, np.array([[None] *
                                            len(self.train_classes)]).astype(
                        np.float64)
                if self.sparsecoded == 'Buffer':
                    features[count] = (self.action_recog.actions.
                                       coders[count].code(features[count]))
                if self.post_pca:
                    features[count] = self.buffer_operators[
                        count].perform_post_time_pca(
                            features[count])

        else:
            return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                np.float64)

        inp = np.hstack(tuple(features))
        try:
            score = (self.unified_classifier.decide(inp))
        except Exception as e:
            raise
        self.scores.append(score)
        if not just_scores:
            self.classify_online(score, self.img_count,
                                 self.mean_from)
        return True, np.array(score).reshape(1, -1)

    @timeit
    def classify_online(self, score, img_count, mean_from):
        '''
        To be used after scores from <online_processing_data> have been
        computed. It is a convenience function to allow the modification of
        the scores, if this is wanted, before performing classification
        '''
        if self.action_type == 'Passive':
            self.scores_filter_shape = 3
        if len(self.scores_running_mean_vec) < self.scores_filter_shape:
            self.scores_running_mean_vec.append(score.ravel())
        else:
            self.scores_running_mean_vec = (self.scores_running_mean_vec[1:]
                                            + [score.ravel()])
        # filter scores using a mean window
        self.scores[-1] = np.mean(np.array(self.scores_running_mean_vec), axis=0)
        if not self.action_type == 'Passive':
            start_from = min(self.scores_filter_shape, mean_from)
            score_std = np.std(self.scores[-1])
            if len(self.small_std_running_mean_vec) < self.std_small_filter_shape:
                self.small_std_running_mean_vec.append(score_std)
            else:
                self.small_std_running_mean_vec = (
                    self.small_std_running_mean_vec[1:] +
                    [score_std])
            filtered_score_std = np.mean(self.small_std_running_mean_vec)
            self.scores_std.append(filtered_score_std)
            if len(self.big_std_running_mean_vec) < self.std_big_filter_shape:
                self.big_std_running_mean_vec.append(filtered_score_std)
            else:
                self.big_std_running_mean_vec = (self.big_std_running_mean_vec[1:]
                                                 + [filtered_score_std])
            if mean_from >= self.std_big_filter_shape:
                start_from = 0
            else:
                start_from = - mean_from
            self.scores_std_mean.append(
                np.mean(self.big_std_running_mean_vec[-start_from:]))
            std_mean_diff = self.scores_std_mean[
                -1] - self.scores_std[-1]
            if (np.min(std_mean_diff) > co.CONST['action_separation_thres'] and not
                    self.on_action) or not self.recognized_classes:
                self.crossings.append(img_count)
                self.on_action = True
                if self.recognized_classes is not None:
                    self.recognized_classes.add(length=img_count -
                                                self.new_action_starts_count +
                                                1)
                    LOG.info('Frame ' + str(img_count) + ': ' +
                             self.recognized_classes.name +
                             ', starting from frame ' +
                             str(self.recognized_classes.start) +
                             ' with length ' +
                             str(self.recognized_classes.length))
                else:
                    self.recognized_classes = RecognitionVectorizer(
                        self.train_classes)
                index = np.argmax(self.scores[-1])
                self.max_filtered_score = self.scores[-1][index]
                self.act_inds = [index]
                self.new_action_starts_count = img_count
                LOG.info('Frame ' + str(img_count) + ': ' +
                         self.recognized_classes.name)
                self.saved_buffers_scores = []
                return None
            else:
                if len(self.recognized_classes) > 0:
                    _arg = np.argmax(self.scores[-1])
                    if self.max_filtered_score < self.scores[
                            -1][_arg]:
                        self.max_filtered_score = self.scores[
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
            if np.max(self.scores[-1]) >= 0.6 or len(
                    self.recognized_classes) == 0:
                self.recognized_classes.append(
                    self.scores[-1].argmax())
            else:
                self.recognized_classes.append(
                    self.recognized_classes[-1])
            LOG.info('Pose detected:'
                     + self.train_classes[self.recognized_classes[-1]])
            return self.train_classes[self.recognized_classes[-1]]

    def extract_actions(self, scores, method='prob_check', tol=0.7,
                        filterr=True):

        if filterr:
            scores = co.noise_proc.masked_filter(scores,
                                                 self.scores_filter_shape)
        extracted_actions = []
        if method == 'prob_check':
            for count, score in enumerate(scores):
                if not np.prod(np.isfinite(score)).astype(bool):
                    extracted_actions.append(np.nan)
                    continue
                if (np.max(score) >= tol or len(
                        extracted_actions) == 0 or
                        np.isnan(extracted_actions[-1])):
                    extracted_actions.append(score.argmax())
                else:
                    extracted_actions.append(
                        extracted_actions[-1])
            return extracted_actions, None
        elif method == 'std_check':
            fmask = np.prod(np.isfinite(scores), axis=1).astype(bool)
            scores_std = np.zeros(scores.shape[0])
            scores_std[:] = None
            scores_std[fmask] = np.std(scores[fmask, :],
                                       axis=1)
            less_filtered_scores_std = co.noise_proc.masked_filter(
                scores_std, self.std_small_filter_shape)

            high_filtered_scores_std = co.noise_proc.masked_filter(
                scores_std,
                self.std_big_filter_shape)
            positive = np.zeros(scores.shape[0])
            positive[:] = None
            positive[fmask] = ((high_filtered_scores_std -
                                less_filtered_scores_std)[fmask] > 0).astype(int)
            # We are interested only in finding negative to positive zero crossings,
            # because this is where std falls below its mean
            neg_to_pos_zero_crossings = np.where(positive[1:] -
                                                 positive[:-1] ==
                                                 -1)[0]
            crossings = neg_to_pos_zero_crossings
            interesting_crossings = np.concatenate((np.array([0]),
                                                    neg_to_pos_zero_crossings,
                                                    np.array([scores.shape[0]])),
                                                   axis=0)
            for cross1, cross2 in zip(interesting_crossings[
                    :-1], interesting_crossings[1:]):
                act_scores = scores[cross1:cross2, :]
                mask = fmask[cross1:cross2]
                if not np.any(mask):
                    act = np.zeros(cross2 - cross1)
                    act[:] = None
                    extracted_actions.append(act)
                    continue
                index = np.mean(
                    act_scores[mask, :], axis=0).argmax()
                '''
                index = np.median(
                    act_scores[mask, :], axis=0).argmax()
                '''
                act = index * np.ones(cross2 - cross1)
                act[np.logical_not(mask)] = None
                extracted_actions.append(act)
            extracted_actions = np.concatenate(
                tuple(extracted_actions), axis=0)

            return extracted_actions, (less_filtered_scores_std,
                                       high_filtered_scores_std,
                                       crossings)

    @timeit
    def classify_offline(self, display=True,
                         save=True, compute_perform=True,
                         extraction_method=None, tol=0.7):
        '''
        To be used after offline have been computed. It is a convenience
        function to allow the modification of the scores, if this is wanted,
        before performing classification.

        Process scores using stds as proposed by the paper
        '''
        if 'Sync' in self.use:
            if extraction_method is None:
                extraction_method = 'prob_check'
            if not isinstance(extraction_method, list):
                extraction_method = [extraction_method] * len(self.scores)
            if not isinstance(tol, list):
                tol = [tol] * len(self.scores)
            self.recognized_classes = {}
            for count, key in enumerate(self.scores):
                (extracted_actions,more) = self.extract_actions(
                    self.scores[key], method=extraction_method[count],
                    tol=tol[count])
                self.recognized_classes[key] = extracted_actions
        else:
            (self.recognized_classes,
             more) = self.extract_actions(
                  self.scores, method=extraction_method)
            if extraction_method == 'std_check':
                self.testdata[self.test_ind][
                    'Results'][
                        'LessFilteredScoresSTD'] = more[0]
                self.less_filtered_scores_std = more[0]
                self.testdata[self.test_ind][
                    'Results'][
                        'HighFilteredScoresSTD'] = more[1]
                self.high_filtered_scores_std = more[1]
                self.testdata[self.test_ind][
                    'Results'][
                        'Crossings'] = more[2]
                self.crossings = more[2]
        if self.test_ind is not None:
            self.testdata[self.test_ind]['Results'][
                'Actions'] = self.recognized_classes
        return self.recognized_classes

    def compute_performance_measures(
            self, recognized_classes, ground_truths, act_namespaces, save=True):
        '''
        Extract confusion matrix, accuracy and f scores from the test
        '''
        from sklearn import metrics
        from scipy.linalg import block_diag
        LOG.info('Computing performance measures for ' +
                 self.classifier_savename + ' with dataset:' +
                 self.testdataname)
        y_trues = []
        y_preds = []
        weights = []
        confusion_matrices = []
        f1_scores = []
        existing_classes = []
        accuracies = []
        if not 'Sync' in self.use:
            recognized_classes = {self.action_type: recognized_classes}
            ground_truths = {self.action_type: ground_truths}
            act_namespaces = {self.action_type: act_namespaces}
        undef_exists = False
        for act_type in recognized_classes:
            ground_truth = ground_truths[act_type]
            recognized_actions = np.array(recognized_classes[act_type])
            act_names = act_namespaces[act_type]
            fmask = np.isfinite(ground_truth)
            y_trues.append(np.array(ground_truth[fmask]).astype(int))
            y_preds.append(recognized_actions[fmask])
            y_preds[-1][np.isnan(y_preds[-1])] = -1
            if -1 in y_preds[-1]:
                undef_exists = True
            weights.append(len(np.unique(ground_truth[fmask])))
            f1_scores.append(np.atleast_2d(metrics.f1_score(y_trues[-1],
                                                            y_preds[-1],
                                                            average=None)))
            if undef_exists:
                f1_scores[-1] = np.atleast_2d(f1_scores[-1][0, 1:])
            confusion_matrices.append(metrics.confusion_matrix(
                y_trues[-1], y_preds[-1]))
            accuracies.append(metrics.accuracy_score(y_trues[-1],
                                                     y_preds[-1]))
            classes = set(y_trues[-1].tolist() +
                          y_preds[-1].tolist())
            classes.discard(-1)
            classes = np.array(list(classes)).astype(int)
            existing_classes += (np.array(
                act_names)[classes]).tolist()
        labels = existing_classes
        labels_w_undef = (['Undefined'] + existing_classes if undef_exists
                          else existing_classes)
        self.f1_scores = np.concatenate(f1_scores, axis=1)
        self.actions_id = []
        for clas in labels_w_undef:
            self.actions_id.append(self.all_actions.index(clas)-1)

        self.confusion_matrix = block_diag(*tuple(confusion_matrices))
        self.accuracy = sum([accuracy * weight for accuracy, weight in
                             zip(accuracies, weights)]) / float(sum(weights))
        if not self.online:
            self.testdata[self.test_ind]['Accuracy'] = self.accuracy
            self.testdata[self.test_ind]['FScores'] = [self.f1_scores,
                                                       self.actions_id]
            self.testdata[self.test_ind]['ConfMat'] = [self.confusion_matrix,
                                                       self.actions_id]
        LOG.info('F1 Scores: \n' + np.array2string(self.f1_scores))
        LOG.info(
            'Confusion Matrix: \n' +
            np.array2string(
                self.confusion_matrix))
        if 'Sync' in self.use:
            LOG.info('Partial Accuracies:' + str(accuracies))
        LOG.info('Accuracy: ' + str(self.accuracy))
        LOG.info('Labels of actions:' + str(labels_w_undef))

        if save and self.save_fold is not None:
            if 'Sync' in self.use:
                acc_labels = (['Class.' + str(clas) for clas in
                               self.parameters['sub_classifiers']] +
                              ['Total Mean'])
                with open(os.path.join(self.save_fold, 'accuracy.tex'), 'w') as out:
                    out.write(co.latex.array_transcribe([accuracies,
                                                         np.atleast_2d(self.accuracy)],
                                                        xlabels=acc_labels,
                                                        sup_x_label='Accuracy',
                                                        extra_locs=['right']))
            with open(os.path.join(self.save_fold, 'f1_scores.tex'), 'w') as out:
                out.write(co.latex.array_transcribe([self.f1_scores,
                                                     np.atleast_2d(self.accuracy)],
                                                    xlabels=np.concatenate((labels_w_undef,
                                                                            ['Accuracy']), axis=0),
                                                    sup_x_label='F-Scores',
                                                    extra_locs=['right']))
            with open(os.path.join(self.save_fold,
                                   'Confusion_Matrix.tex'), 'w') as out:
                out.write(co.latex.array_transcribe(self.confusion_matrix,
                                                    ylabels=labels_w_undef,
                                                    xlabels=labels_w_undef,
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
                    'Sync' in parameters[0]['classifier']):
                '''
                if parameters[0]['classifier']=='Double':
                    row.append('%d'%parameters[0]['classifier_params'][
                    'RDF_n_estimators'])
                else:
                    row.append('')
                '''
                row.append('')
                row.append(
                    str(parameters[0]['classifier_params']['SVM_kernel']))
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
                    row.append(str(parameters[0]['PTPCA']))
                except:
                    row.append('')
                if row[-1] != '' and row[-1] == 'True':
                    try:
                        row.append('%d' % parameters[0]['PTPCA_params'][
                            'PTPCA_components'])
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


    def load_tests_mapping(self):
        tests_catalog = co.file_oper.load_labeled_data(['Testing'],
                        fold_lev=1, just_catalog=True, include_all_catalog=
                                                         True)
        return tests_catalog

    def return_description(self, catalog, value):
        try:
            return catalog.keys()[catalog.values().index(value)]
        except:
            LOG.warning('No such value inside catalog')
    def load_all_test_instances(self,test_ind):
        available_test_instances= {}
        test_name = self.available_tests[test_ind]
        import ast
        loaded_instances,keys_list = co.file_oper.load_labeled_data(['Testing'],
                fold_lev=1, all_inside=True)
        for _id in loaded_instances:
            loaded_instance = loaded_instances[_id]
            if dict(keys_list[_id][0])['Test'] == test_name:
                available_test_instances[_id] = loaded_instance

        return available_test_instances

    def extract_test_results_instances(self, test_ind,key,*keys):
        if self.test_instances is None:
            self.test_instances = self.load_all_test_instances(test_ind)
        res = []
        for entry in co.dict_oper.create_sorted_dict_view(self.test_instances):
            if entry[1] is None:
                res.append(None)
            else:
                res.append(co.dict_oper.lookup(entry[0],key,*keys))
        return res


    def correlate_with_ground_truth(self, save=True, display=False,
                                    display_all=False, compute_perform=True):
        '''
        Plot results with title <title>
        <display_all> if a plot of all the classifiers results is wanted
        Do not use this function if more than one classifiers are to be run in
        parallel
        '''
        if self.parameters['testing_params']['online']:
            recognized_classes = self.recognized_classes.recognition_vector
            if isinstance(self.crossings, list):
                self.crossings = np.array(self.crossings)
        if self.test_ground_truth is not None and compute_perform:
            self.compute_performance_measures(
                self.recognized_classes,
                ground_truths=self.test_ground_truth,
                act_namespaces=self.train_classes,
                save=save)

        if save:
            display = True

        if display:
            if display_all:
                self.construct_classifiers_matrix()
                iterat = []
                iterat_name = []
                for name in self.classifiers_list:
                    parameters = self.classified_dict[name][
                        self.testdataname][2]
                    recognized_actions = self.classified_dict[name][
                        self.testdataname][0]
                    if (parameters['action_type'] == self.action_type):
                        if 'Sync' in parameters['classifier']:
                            iterat.append(recognized_actions[self.action_type])
                        else:
                            iterat.append(recognized_actions)
                        iterat_name.append(int(self.classifiers_list[name]))
                # sort iterat based on the index of classifier inside
                #   classifiers_list
                iterat = [x for (_, x) in sorted(zip(iterat_name, iterat),
                                                 key=lambda pair: pair[0])]
                # provide argsort using accuracy measures, to alter line width
                higher_acc = sorted(range(len(iterat)), key=lambda
                                    l: l[1][0], reverse=True)
            else:
                try:
                    iterat = [self.testdata[self.test_ind][
                        'Results']['Actions']]
                    available_ids = co.file_oper.load_labeled_data(
                        ['Classifier'],just_catalog=True
                        )
                    if available_ids is not None:
                        try:
                            iterat_name = available_ids[str(self.classifier_id)]
                        except KeyError:
                            iterat_name = str(len(available_ids))
                    else:
                        iterat_name = str(0)
                    higher_acc = [0]
                    if 'Sync' in self.use:
                        new_iter = []
                        ref = 0
                        for key in iterat[0]:
                            new_iter.append(iterat[0][key])
                            new_iter[-1] = [item + ref for
                                                    item in new_iter[-1]]
                            ref += len(self.parameters['actions_params'][key])
                        iterat = new_iter
                        iterat_name = self.parameters['sub_classifiers']
                        higher_acc.append(1)
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
            width = 1
            dec_q = 0.3
            min_q = 0.2
            if self.test_ground_truth is not None:
                if 'Sync' in self.use:
                    tg_ref = 0
                    for key in (self.test_ground_truth):
                        tg = self.test_ground_truth[key]
                        tg += tg_ref
                        plots.append(tg)
                        tg_ref += len(self.parameters['actions_params'][
                            key])
                        labels.append('Ground Truth')
                        markers.append(',')
                        linewidths.append(1.6)
                        alphas.append(1)
                else:
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
            for count in range(len(higher_acc)):
                syncplots = []
                plots.append(iterat[count])
                labels.append('Class. ' +
                              str(iterat_name[count]) +
                              ' Results')
                markers.append(',')
                linewidths.append(width)
                alphas.append(0.8)
                width -= dec_q
                width = max(min_q, width)
            if 'Sync' in self.use:
                yticks = []
                for key in self.train_classes:
                    yticks += list(self.train_classes[key])
            else:
                yticks = self.train_classes
            ylim = (-1, len(yticks) + 1)
            self.plot_result(np.vstack(plots).T, labels=labels,
                             xticks_locs=xticks, ylim=ylim,
                             yticks_names=yticks,
                             info='Classification Results',
                             markers=markers,
                             linewidths=linewidths,
                             alphas=alphas,
                             xlabel='Frames', save=save)
            return True


class RecognitionVectorizer(object):
    '''
    Class to hold classification classes
    '''

    def __init__(self, class_names):
        self.name = ''
        self.index = 0
        self.start = 0
        self.length = 0
        self.names = class_names
        self.recognition_vector = []

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
            self.recognition_vector.append(
                [None] * (start - len(self.recognition_vector)))
        if length is not None:
            self.length = length
            self.recognition_vector.append(
                [self.index] * self.length)


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
                                                                +
                                                                '_ground_truth'],
            classes_namespace=classifier.train_classes)
        classifier.scores = np.array(
            classifier.scores).squeeze()
        classifier.scores_exist = np.array(classifier.scores_exist)
        expanded_scores = np.zeros(
            (len(classifier.scores_exist), classifier.scores.shape[1]))
        expanded_scores[:] = np.NaN
        expanded_scores[
            classifier.scores_exist.astype(bool),
            :] = classifier.scores
    else:
        expanded_scores = np.array(classifier.scores)
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
                                         sparse_dim_rat=None, test_against_all=False,
                                         visualize_feat=False, kernel=None,
                                         features='GHOG', post_pca=False,
                                         post_pca_components=1,
                                         just_sparse=False,
                                         debug=False,
                                         use='SVM',
                                         action_type='Dynamic',
                                         post_scores_processing_method='std_check'):
    '''
    Constructs an SVM classifier with input 3DHOF and GHOG features
    '''
    if use_sparse:
        if sparse_dim_rat is None:
            sparse_dim_rat = co.CONST['sparse_dim_rat']
    classifier = Classifier('INFO', action_type=action_type,
                            name=name, sparse_dim_rat=sparse_dim_rat,
                            use_sparse=use_sparse,
                            features=features,
                            kernel=kernel, post_pca=post_pca,
                            post_pca_components=post_pca_components,
                            use=use, post_scores_processing_method=
                            post_scores_processing_method)
    if debug:
        classifier.debug = True
    classifier.run_training(classifiers_retrain=train,
                            coders_retrain=coders_retrain,
                            visualize_feat=visualize_feat,
                            just_sparse=just_sparse,
                            # init_sc_traindata_num=5000,
                            init_sc_traindata_num=15000,
                            min_dict_iterations=20)
    if test or visualize:
        if test_against_all:
            iterat = classifier.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                classifier.run_testing(name,
                    ground_truth_type=os.path.join(
                    co.CONST['ground_truth_fold'],
                    name + '.csv'),
                    online=False, load=False)
            else:
                classifier.run_testing(name,
                    ground_truth_type=os.path.join(
                    co.CONST['ground_truth_fold'],
                    name + '.csv'),
                    online=False, load=False)
    return classifier


def construct_passive_actions_classifier(testname='test2',
                                         train=True, test=True, visualize=True,
                                         test_against_all=False,
                                         features='3DXYPCA',
                                         post_scores_processing_method='prob_check'):
    '''
    Constructs a random forests passive_actions classifier with input 3DXYPCA features
    '''
    classifier = Classifier('INFO', action_type='Passive',
                            name='actions', use='RDF',
                            use_sparse=False,
                            features=features,
                            post_scores_processing_method=
                           post_scores_processing_method)
    classifier.run_training(classifiers_retrain=train,
                            max_act_samples=2000)
    if test or visualize:
        if test_against_all:
            iterat = classifier.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                classifier.run_testing(name,
                    ground_truth_type=os.path.join(
                    co.CONST['ground_truth_fold'],
                    name + '.csv'),
                    online=False, load=False)
            else:
                classifier.run_testing(name,
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
            ground_truth_type=os.path.join(co.CONST['test_save_path'], test),
            classes_namespace=classifier.dynamic_actions)
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
        use_sparse=True,just_sparse=True, debug=False)
    '''
    '''
    POSES_CLASSIFIER = construct_passive_actions_classifier(train=True,
                                                            test=True,
                                                            visualize=True,
                                                            test_against_all=True)
    '''
    ACTIONS_CLASSIFIER_SIMPLE = construct_dynamic_actions_classifier(
        train=True,
        test=True,
        visualize=True,
        test_against_all=True)
    '''
    ACTIONS_CLASSIFIER_SPARSE = construct_dynamic_actions_classifier(train=True,
                                                                     coders_retrain=False,
                                                                     test=True,
                                                                     visualize=True,
                                                                     test_against_all=True,
                                                                     use_sparse='Features',
                                                                     use='RDF')
    ACTIONS_CLASSIFIER_SIMPLE_POST_PCA = construct_dynamic_actions_classifier(
        train=True,
        test=True,
        visualize=True,
        test_against_all=True,
        post_pca=True,
        post_pca_components=2)
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['3DHOF'],post_pca=False,use_sparse=False)
    '''
    '''
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['ZHOF'], post_pca=False, use_sparse=False)

    '''
    '''
    ACTIONS_CLASSIFIER_SPARSE_WITH_3DHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', '3DHOF'], coders_retrain=False, use_sparse=True,
        kernel='linear')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF_POST_PCA = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', '3DHOF'], post_pca=True, post_pca_components=4)
    construct_dynamic_actions_classifier(
        #debugging, train=True, test=True,
        train=True, test=True,
        visualize=True, test_against_all=True,
        features=['GHOG', '3DHOF'], post_pca=True, use_sparse=True,
        post_pca_components=2)
    '''
    '''
    ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', '3DHOF'], kernel='linear')
    '''
    '''
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', 'ZHOF'])
    '''
    '''
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['3DXYPCA','GHOG','3DHOF','ZHOF'], use='SVM')
    '''
    # Let's try RDF for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', 'ZHOF'], post_pca=False, use_sparse=False,
        use='RDF')
    exit()
    # Let's try RDF with all features for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', '3DHOF', '3DXYPCA'], post_pca=False, use_sparse=False,
        use='RDF')

    # Let's try RDF for all features for all actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', '3DHOF', '3DXYPCA'], action_type='All', post_pca=False, use_sparse=False,
        use='RDF')
    # Let's try RDF with all features for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', 'ZHOF', '3DXYPCA'], action_type='Dynamic', post_pca=False, use_sparse=False,
        use='RDF')

    # Let's try RDF for all features for all actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', 'ZHOF', '3DXYPCA'], action_type='All', post_pca=False, use_sparse=False,
        use='RDF')
    ACTIONS_CLASSIFIER_SPARSE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=True,
        test=True, visualize=True, test_against_all=True,
        features=['GHOG', 'ZHOF'], coders_retrain=False, use_sparse=True,
        kernel='linear')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF_POST_PCA = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', 'ZHOF'], post_pca=True, post_pca_components=4)
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        features=['GHOG', 'ZHOF'], post_pca=True,
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
    # construct classifiers comparative table
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
