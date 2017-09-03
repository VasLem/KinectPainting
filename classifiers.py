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

    def wrapper(self, *arg, **kw):
        t1 = time.time()
        res = func(self, *arg, **kw)
        t2 = time.time()
        self.time.append(t2 - t1)
        del self.time[:-5000]
        return res
    return wrapper


class Classifier(object):
    '''
    Class to hold all Classifier specific methods.
    <descriptors>:['pca','ghog','3dhof']
    <action_type>:True if no buffers are used
    <sparsecoding_level> is True if sparse coding is used
    Classifier Parameters, for example <AdaBoost_n_estimators> or
    <RDF_n_estimators> or  <kernel> can be
    a list, which will be reduced using optimized grid search with cross
    validation.
    '''

    def __init__(self, log_lev='INFO',
                 visualize=False, masks_needed=True,
                 buffer_size=co.CONST['buffer_size'],
                 sparse_dim_rat=co.CONST['sparse_dim_rat'],
                 descriptors='',
                 ptpca=False,
                 ptpca_components=None,
                 action_type='Dynamic',
                 classifiers_used='SVM', num_of_cores=4, name='',
                 svm_c=None,
                 AdaBoost_n_estimators=None,
                 RDF_n_estimators=None,
                 add_info=None,
                 sparsecoding_level=None,
                 kernel=None,
                 save_all_steps=False,
                 post_scores_processing_method=None,
                 hardcore=False,
                 for_app=False):
        '''
        sparsecoding_level = [Buffer, Features, None]
        '''
        if not os.path.isdir(co.CONST['AppData']):
            os.makedirs(co.CONST['AppData'])
        self.app_dir = co.CONST['AppData']
        self.for_app = for_app
        self.time = []
        self.classifiers_ids = None
        self.test_ind = None
        # General configuration
        if not isinstance(descriptors, list):
            descriptors = [descriptors]
        descriptors = sorted(descriptors)
        ###
        features_params = {}
        coders_params = {}
        for descriptor in descriptors:
            features_params[descriptor] = {}
            features_params[descriptor]['params'] = {attrib.replace(descriptor, ''):
                                                     co.CONST[attrib] for
                attrib in co.CONST if
                attrib.startswith(descriptor)}
            features_params[descriptor]['sparsecoded'] = sparsecoding_level
            features_params[descriptor]['action_type'] = action_type
            coders_params[descriptor] = {}
            if not sparsecoding_level:
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
        self.test_name = None
        self.kernel = kernel
        self.svm_c = svm_c
        self.RDF_n_estimators = RDF_n_estimators
        self.AdaBoost_n_estimators = AdaBoost_n_estimators
        self.sparse_dim_rat = sparse_dim_rat
        if 'SVM' in classifiers_used and kernel is None:
            self.kernel = 'linear'
        if 'SVM' in classifiers_used:
            if svm_c is None:
                self.svm_c = co.CONST['SVM_C']
            if post_scores_processing_method == 'CProb':
                LOG.warning('Invalid post_scores_processing_method for SVM')
                if hardcore:
                    raise Exception
                else:
                    LOG.warning('Changing method to CSTD')
                    post_scores_processing_method = 'CSTD'
        if 'RDF' in classifiers_used or 'AdaBoost' in classifiers_used:
            if svm_c is not None:
                LOG.warning(
                    'svm_c is not None for RDF or AdaBoost experimentation')
                if hardcore:
                    raise Exception
        if post_scores_processing_method is None:
            if 'RDF' in classifiers_used or 'AdaBoost' in classifiers_used:
                post_scores_processing_method = 'CProb'
            else:
                post_scores_processing_method = 'CSTD'
        classifier_params = {}
        if 'RDF' in classifiers_used and RDF_n_estimators is None:
            self.RDF_n_estimators = co.CONST['RDF_trees']
        if 'AdaBoost' in classifiers_used and AdaBoost_n_estimators is None:
            self.AdaBoost_n_estimators = co.CONST['AdaBoost_Estimators']
        if 'SVM' in classifiers_used:
            classifier_params['SVM_kernel'] = self.kernel
            classifier_params['SVM_C'] = self.svm_c
        if 'RDF' in classifiers_used:
            classifier_params['RDF_n_estimators'] = self.RDF_n_estimators
        if 'AdaBoost' in classifiers_used:
            classifier_params['AdaBoost_n_estimators'] = self.AdaBoost_n_estimators
        if action_type != 'Passive':
            dynamic_params = {'buffer_size': buffer_size,
                              'buffer_confidence_tol': co.CONST['buffer_confidence_tol'],
                              'filter_window_size':
                              co.CONST['STD_big_filt_window']}
        else:
            dynamic_params = {'buffer_size': 1}
        if ptpca and ptpca_components is None:
            ptpca_components = co.CONST['PTPCA_components']
        ptpca_params = {'PTPCA_components': ptpca_components}
        for descriptor in descriptors:
            features_params[descriptor]['dynamic_params'] = dynamic_params
        if sparsecoding_level:
            if not isinstance(sparse_dim_rat, list):
                sparse_dim_rat = [sparse_dim_rat] * len(descriptors)
            if len(list(sparse_dim_rat)) != len(descriptors):
                raise Exception('<sparse_dim_rat> should be either an integer/None or' +
                                ' a list with same length with <descriptors>')
            sparse_params = dict(zip(descriptors, sparse_dim_rat))
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
        LOG.debug('Extracting: ' + str(descriptors))
        self.parameters = {'classifier': classifiers_used,
                           'descriptors': descriptors,
                           'features_params': features_params,
                           'coders_params': coders_params,
                           'dynamic_params': dynamic_params,
                           'classifier_params': classifier_params,
                           'sparse_params': sparse_params,
                           'action_type': action_type,
                           'sparsecoded': sparsecoding_level,
                           'testing': False,
                           'testing_params': testing_params,
                           'actions_params': action_params,
                           'PTPCA': ptpca,
                           'PTPCA_params': ptpca_params}
        self.training_parameters = {k: self.parameters[k] for k in
                                    ('classifier', 'descriptors',
                                     'features_params',
                                     'dynamic_params',
                                     'classifier_params',
                                     'sparse_params',
                                     'action_type',
                                     'sparsecoded',
                                     'PTPCA',
                                     'PTPCA_params') if k in
                                    self.parameters}
        self.descriptors = descriptors
        self.add_info = add_info
        self.log_lev = log_lev
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.masks_needed = masks_needed
        self.action_type = action_type
        self.classifiers_used = classifiers_used
        self.num_of_cores = num_of_cores
        self.name = name
        self.ptpca = ptpca
        self.action_recog = ara.ActionRecognition(
            self.parameters,
            log_lev=log_lev)
        if not self.for_app:
            self.available_tests = sorted(os.listdir(co.CONST['test_save_path']))
        else:
            self.available_tests = []
        self.update_experiment_info()
        if 'SVM' in self.classifiers_used:
            from sklearn.svm import LinearSVC
            self.classifier_type = LinearSVC(
                class_weight='balanced', C=self.svm_c,
                multi_class='ovr',
                dual=False)
        elif 'RDF' in self.classifiers_used:
            from sklearn.ensemble import RandomForestClassifier
            self.classifier_type =\
                RandomForestClassifier(self.RDF_n_estimators)
        elif 'AdaBoost' in self.classifiers_used:
            from sklearn.ensemble import AdaBoostClassifier
            self.classifier_type =\
                AdaBoostClassifier(n_estimators=self.AdaBoost_n_estimators)
        self.unified_classifier = None
        if sparsecoding_level:
            if not(sparsecoding_level == 'Features' or sparsecoding_level == 'Buffer'):
                raise Exception('Invalid sparsecoding_level, its value shoud be '
                                + 'None/False/Buffer/Features')
        self.sparsecoded = sparsecoding_level
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
            co.file_oper.save_labeled_data(['Classifier'],
                                           [self.unified_classifier,
                                            self.training_parameters],
                                           name=self.app_dir)
            if isinstance(info, tuple):
                self.training_params = info[0]
                self.additional_params = info[1:]
            else:
                self.training_params = info
            self.loaded_classifier = True
            LOG.info('Loaded Classifier')
        except TypeError:
            if self.for_app:
                [self.unified_classifier,
                 info] = co.file_oper.load_labeled_data(
                     ['Classifier'],
                     name=self.app_dir)
                self.loaded_classifier = True
            else:
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
                for nam in self.parameters['descriptors']]
            self.buffer_operators = [
                ara.BufferOperations(self.parameters)
                for nam in self.parameters['descriptors']]
            if self.sparsecoded:
                [self.action_recog.
                 actions.load_sparse_coder(ind) for ind in range(
                     len(self.parameters['descriptors']))]
        except BaseException: pass

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
                try:
                    self.partial_accuracies[count] = self.testdata[count][
                        'PartialAccuracies']
                except BaseException: pass
            else:
                self.testdata[count] = {}
                self.testdata[count]['Accuracy'] = {}
                self.testdata[count]['FScores'] = {}
                self.testdata[count]['Results'] = {}
                self.testdata[count]['ConfMat'] = {}
                self.testdata[count]['TestTime'] = {}
                self.testdata[count]['Labels'] = {}
                try:
                    self.testdata[count]['PartialAccuracies'] = {}
                except BaseException: pass

    def update_experiment_info(self):
        if self.parameters['action_type'] == 'Passive':
            info = 'passive '
        else:
            info = 'dynamic '
        info = info + self.name + ' ' + self.classifiers_used + ' '
        info += 'using'
        if self.parameters['sparsecoded']:
            info += ' sparsecoded'
        for feature in self.parameters['descriptors']:
            info += ' ' + feature
        info += ' descriptors '
        if 'SVM' in self.parameters['classifier']:
            info += 'with ' + self.parameters[
                'classifier_params']['SVM_kernel'] + ' kernel'
        elif 'RDF' in self.parameters['classifier']:
            info += ('with ' + str(self.parameters['classifier_params'][
                'RDF_n_estimators']) + ' estimators')
        elif 'AdaBoost' in self.parameters['classifier']:
            info += ('with ' + str(self.parameters['classifier_params'][
                'AdaBoost_n_estimators']) + ' estimators')

        if self.parameters['action_type'] == 'Dynamic':
            info += ' with buffer size ' + str(self.buffer_size)
        if self.parameters['sparsecoded']:
            info += ' with sparsecoding by ratio of ' + \
                str(self.sparse_dim_rat)
        if self.ptpca:
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
        for count in range(len(self.parameters['descriptors'])):
            _id, file_id = self.action_recog.actions.retrieve_descriptor_possible_ids(count,
                                                                                      assume_existence=True)
            self.features_id.append(_id)
            self.features_file_id.append(file_id)
            self.classifier_id = [co.dict_oper.create_sorted_dict_view(
                {'Classifier': str(self.classifiers_used)}),
                co.dict_oper.create_sorted_dict_view(
                                      {'ClassifierParams': str(co.dict_oper.create_sorted_dict_view(
                                          self.parameters['classifier_params']))}),
                co.dict_oper.create_sorted_dict_view(
                                  {'ActionsType': str(self.action_type)}),
                co.dict_oper.create_sorted_dict_view(
                                  {'FeaturesParams': str(self.features_file_id)})]

    def update_tests_ids(self):
        self.tests_ids = []
        for count, test in enumerate(self.available_tests):
            self.tests_ids.append([co.dict_oper.create_sorted_dict_view({'Test': str(test)}),
                                   co.dict_oper.create_sorted_dict_view(
                                   {'TestingParams': str(co.dict_oper.create_sorted_dict_view(
                                       self.parameters['testing_params']))})]
                                  + [self.classifier_id])

    def initialize_classifier(self, classifier):
        '''
        Add type to classifier and set methods
        '''
        self.unified_classifier = classifier
        if 'SVM' in self.classifiers_used:
            self.unified_classifier.decide = self.unified_classifier.decision_function
            self.unified_classifier.predict = self.unified_classifier.predict
        elif 'RDF' in self.classifiers_used or 'AdaBoost' in self.classifiers_used:
            self.unified_classifier.decide = self.unified_classifier.predict_proba
            self.unified_classifier.predict = self.unified_classifier.predict
        co.file_oper.save_labeled_data(['Classifier'] + self.classifier_id,
                                       [self.unified_classifier,
                                        self.training_parameters])
        co.file_oper.save_labeled_data(['Classifier'],
                                       [self.unified_classifier,
                                        self.training_parameters],
                                       name=self.app_dir)

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
        try:
            self.train_classes = [name for name in os.listdir(training_datapath)
                              if os.path.isdir(os.path.join(training_datapath, name))][::-1]
            
        except:
            if self.for_app:
                with open(os.path.join(self.app_dir,
                                       'train_classes'),'r') as inp:
                    self.train_classes = pickle.load(inp)
            else:
                raise
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
            if 'Sync' in self.classifiers_used:
                self.train_classes = {'Passive': passive_actions,
                                      'Dynamic': dynamic_actions}
            else:
                classes = []
                if self.action_type == 'Dynamic' or self.action_type == 'All':
                    classes += dynamic_actions
                if self.action_type == 'Passive' or self.action_type == 'All':
                    classes += passive_actions
                self.train_classes = classes
        with open(os.path.join(self.app_dir,
                               'train_classes'),'w') as out:
            pickle.dump(self.train_classes, out)

    def run_training(self, coders_retrain=False,
                     classifiers_retrain=False,
                     training_datapath=None, classifier_savename=None,
                     num_of_cores=4, classifier_save=True,
                     max_act_samples=None,
                     min_dict_iterations=5,
                     visualize_feat=False, just_sparse=False,
                     init_sc_traindata_num=200,
                     train_all=False):
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
        self.train_all = train_all
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
        else:
            if not self.sparsecoded:
                return
        self.prepare_training_data(training_datapath, max_act_samples,
                                   visualize_feat=visualize_feat)
        if just_sparse:
            return

        if self.sparsecoded and self.coders_to_train and classifiers_retrain:
            # Enters only if coders were not initially trained or had to be
            # retrained. Otherwise, sparse descriptors are computed when
            #<Action.add_features> is called
            LOG.info('Trained' + str([self.parameters['descriptors'][coder] for coder in
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
            self.training_data = []
            self.training_samples_inds = []
            for act_count, action in enumerate(self.train_classes):
                LOG.info('Action:' + action)
                descriptors, samples_indices, mean_depths, _, trained_coders, _ = self.add_action(name=action,
                                                                                              data=os.path.join(
                                                                                         path, action),
                    use_dexter=False,
                    action_type=self.action_type,
                    max_act_samples=max_act_samples)

                if not(self.sparsecoded and None in trained_coders):
                    descriptors = np.hstack(tuple(descriptors))
                    fmask = np.prod(np.isfinite(
                        descriptors), axis=1).astype(bool)
                    descriptors = descriptors[fmask]
                    LOG.info('Action \'' + action + '\' has ' +
                             'descriptors of shape ' + str(descriptors.shape))
                    self.training_data.append(descriptors)
                    self.training_samples_inds.append(
                        np.array(samples_indices)[fmask])
                else:
                    self.training_samples_inds = []
                    self.training_data = []
                    self.train_ground_truth = []
            if self.training_data:
                if self.action_type == 'Dynamic':
                    self.training_data = co.preproc_oper.equalize_samples(
                        samples=self.training_data,
                        utterance_indices=self.training_samples_inds,
                        mode='random')
                self.train_ground_truth = []
                for act_count, clas in enumerate(self.training_data):
                    self.train_ground_truth += clas.shape[0] * [act_count]
                self.training_data = np.vstack((self.training_data))

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
            classifier_params = {elem.replace(self.classifiers_used + '_', ''):
                                 self.parameters['classifier_params'][elem]
                                 for elem in
                                 self.parameters['classifier_params']
                                 if elem.startswith(self.classifiers_used)}
            if any([isinstance(classifier_params[elem], list)
                    for elem in classifier_params]):
                grid_search_params = classifier_params.copy()
                from sklearn.multiclass import OneVsRestClassifier
                if isinstance(self.classifier_type, OneVsRestClassifier):
                    grid_search_params = {('estimator__' + key): classifier_params[key]
                                          for key in classifier_params}
                grid_search_params = {key: (grid_search_params[key] if
                                           isinstance(
                                               grid_search_params[key], list)
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
                classifier_params = {self.classifiers_used + '_' + key: best_params[key] for key
                                     in best_params}
                self.parameters['classifier_params'].update(classifier_params)
                self.training_parameters['classifier_params'].update(
                    classifier_params)
                self.classifier_type = best_estimator
                self.update_experiment_info()
                savepath = self.classifier_savename
            self.initialize_classifier(self.classifier_type.fit(self.training_data,
                                                                self.train_ground_truth))

    def compute_testing_time(self, testname):
        testing_time = {}
        features_extraction_time = 0
        if not self.online:
            for count in range(len(self.parameters['descriptors'])):
                try:
                    loaded = co.file_oper.load_labeled_data(
                        [str(self.features_id[count][-1])] +
                        self.features_file_id[count] +
                        [str(testname)])
                    (_, _, _, feat_times) = loaded
                except BaseException:
                    return None
                for key in feat_times:
                    LOG.info('Time:' + str(key) + ':' +
                             str(np.mean(feat_times[key])))
                    features_extraction_time += np.mean(feat_times[key])
            try:
                testing_time['Classification'] = self.time[
                    -1] / float(self.scores.shape[0])
            except IndexError:
                testing_time['Classification'] = (
                    co.file_oper.load_labeled_data(
                        ['Testing'] + self.tests_ids[
                            self.available_tests.index(
                                testname)])['TestTime'][
                                    'Classification'])
        else:
            testing_time['Classification'] = np.mean(self.time)
        testing_time['Features Extraction'] = features_extraction_time
        return testing_time

    def add_action(self, name=None, data=None, visualize=False, offline_vis=False,
                   to_visualize=[], exit_after_visualization=False,
                    use_dexter=False,
                    action_type=None,
                   max_act_samples=None):
        return self.action_recog.add_action(
            name=name,
            use_dexter=use_dexter,
            action_type=self.action_type,
            max_act_samples=max_act_samples,
            data=data,
            offline_vis=offline_vis,
            to_visualize=to_visualize,
            exit_after_visualization=exit_after_visualization)


    def offline_testdata_processing(self, datapath):
        '''
        Offline testing data processing, using data in <datapath>.
        '''
        LOG.info('Processing test data..')
        LOG.info('Extracting descriptors..')
        (descriptors, _, mean_depths, test_name, _, _) = self.add_action(
            name=None, data=datapath)
        testdata = np.hstack(tuple(descriptors))
        self.parameters['testing_params'][test_name] = test_name
        self.parameters['testing_params']['current'] = test_name
        return testdata

    def save_plot(self, fig, lgd=None, display_all=False, info=None):
        '''
        <fig>: figure
        <lgd>: legend of figure
        <display_all>: whether to save as Total plot
        Saves plot if the action resides in self.available_tests
        '''
        filename = None
        if display_all:
            testname = self.action_type.lower()
            filename = os.path.join(*self.save_fold.split(os.sep)[:-1] +
                                    ['Total', testname + '.pdf'])
        else:
            if self.test_name is None:
                self.test_name = (self.name + ' ' + self.classifiers_used).title()
            if self.test_name in self.available_tests:
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
                LOG.info('Saving to ' + self.save_fold)

                if info is not None:
                    filename = os.path.join(
                        self.save_fold, (self.testname + ' ' + info +
                                         '.pdf').replace(' ','_'))
                else:
                    filename = os.path.join(
                        self.save_fold, self.testname.replace(' ','_') + '.pdf')
            else:
                LOG.warning('Requested figure to plot belongs to an' +
                            ' action that does not reside in <self.'+
                            'available_tests> .Skipping..')
                filename = None
        import matplotlib.pyplot as plt
        if filename is not None:
            if lgd is None:
                plt.savefig(filename)
            else:
                plt.savefig(filename,
                            bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

    def plot_result(self, data, info=None, save=True, xlabel='Frames', ylabel='',
                    labels=None, colors=None, linewidths=None, alphas=None,
                    xticks_names=None, yticks_names=None, xticks_locs=None,
                    yticks_locs=None, markers=None, markers_sizes=None, zorders=None, ylim=None, xlim=None,
                    display_all=False, title=False):
        '''
        <data> is a numpy array dims (n_points, n_plots),
        <labels> is a string list of dimension (n_plots)
        <colors> ditto
        '''
        import matplotlib
        from matplotlib import pyplot as plt
        #matplotlib.rcParams['text.usetex'] = True
        #matplotlib.rcParams['text.latex.unicode'] = True
        # plt.style.classifiers_used('seaborn-ticks')
        if len(data.shape) == 1:
            data = np.atleast_2d(data).T
        fig, axes = plt.subplots()
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
        if markers_sizes is None:
            markers_sizes = [10] * data.shape[1]
        if colors is None:
            colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        if alphas is None:
            alphas = data.shape[1] * [1]
        if zorders is None:
            zorders = data.shape[1] * [0]
        while len(colors) < data.shape[1]:
            colors += [tuple(np.random.random(3))]
        if linewidths is None:
            linewidths = [1] * data.shape[1]
        lgd = None
        for count in range(data.shape[1]):
            if labels is not None:
                axes.plot(data[:, count], label='%s' % labels[count],
                          color=colors[count],
                          linewidth=linewidths[count],
                          marker=markers[count], alpha=alphas[count],
                          zorder=zorders[count],
                          markersize=markers_sizes[count])
                lgd = co.plot_oper.put_legend_outside_plot(axes,
                                                           already_reshaped=True)
            else:
                axes.plot(data[:, count],
                          color=colors[count],
                          linewidth=linewidths[count],
                          marker=markers[count], alpha=alphas[count],
                          zorder=zorders[count],
                          markersize=markers_sizes[count])
        if title:
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
            self.save_plot(fig, lgd, display_all=display_all, info=info)
        return fig, lgd, axes

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
            self.testname = (self.name + ' ' + self.classifiers_used).title()
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
                available_tests_ids = co.file_oper.load_labeled_data(['Testing'],
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
                                           'Classification', 'Online'])
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
        <co.gd_oper.construct_ground_truth>). If the training is online, the count of
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
                self.test_name = data
            except BaseException:
                if data.split(os.sep)[-1] in self.available_tests:
                    self.test_ind = (
                        self.available_tests.index(data.split(os.sep)[-1]))
                    self.test_name = data.split(os.sep)[-1]
                elif data in self.dynamic_actions or data in self.passive_actions:
                    self.test_ind = None
                elif data.split(os.sep)[-1] in self.dynamic_actions or \
                        data.split(os.sep)[-1] in self.passive_actions:
                    self.test_ind = None
                else:
                    raise Exception('test data must be inside test_save_path,' +
                                    ' check config.yaml')
            if construct_gt and ground_truth_type is None:
                ground_truth_type =os.path.join(
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
                    load and self.accuracies[self.available_tests.index(self.test_name)]
                    is not None):
                LOG.info('Tests already performed, loaded data')
                try:
                    self.scores = self.results['Scores']
                    loaded = True
                except:
                    pass
            if not loaded:
                if self.test_ind is not None:
                    testdata = self.offline_testdata_processing(
                        os.path.join(co.CONST['test_save_path'],
                                     self.test_name))
                else:
                    testdata = self.offline_testdata_processing(
                        data)

                try:
                    self.test_ind = self.available_tests.index(data)
                except BaseException:                    self.test_ind = None
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
                self.test_ground_truth, self.test_breakpoints = co.gd_oper.construct_ground_truth(
                    os.path.join(co.CONST['test_save_path'], self.test_name),
                    classes_namespace=self.train_classes,
                    length=self.scores.shape[0],
                    ground_truth_type=ground_truth_type,
                    ret_breakpoints=True)
                utterances_inds = co.gd_oper.merge_utterances_vectors(
                    co.gd_oper.create_utterances_vectors(
                        self.test_breakpoints, len(self.test_ground_truth)),
                    self.train_classes)
            if not just_scores:
                self.classify_offline(save=save, display=display_scores,
                                      compute_perform=compute_perform,
                                      extraction_method=
                                      self.parameters[
                                          'testing_params']['post_scores_processing_method'])

                self.correlate_with_ground_truth(save=save,
                                                 display=display_scores,
                                                 compute_perform=compute_perform,
                                                 utterances_inds=utterances_inds)
                self.display_scores_and_time(save=save)
            if self.test_ind is not None:
                co.file_oper.save_labeled_data(['Testing'] +self.tests_ids[
                    self.test_ind], self.testdata[self.test_ind])
            if not just_scores:
                if display_scores:
                    if self.parameters['testing_params'][
                        'post_scores_processing_method'] == 'CSTD':
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
            if display_scores:
                self.plot_result(self.scores,
                                 labels=self.train_classes,
                                 info='Scores',
                                 xlabel='Frames',
                                 save=save,
                                 )
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

    def visualize_action(self, action, save=True,
                         save_name=None, *args, **kwargs):
        '''
        Visualizes action or a testing dataset using predefined locations in
        config.yaml and the method co.draw_oper.plot_utterances
        '''
        dataset_loc = '/media/vassilis/Thesis/Datasets/PersonalFarm/'
        results_loc = '/home/vassilis/Thesis/KinectPainting/Results/DataVisualization'
        ground_truth, breakpoints, labels = co.gd_oper.load_ground_truth(action, ret_labs=True,
                                                                         ret_breakpoints=True)
        testing =True
        images_base_loc = os.path.join(dataset_loc, 'actions',
                                       'sets' if not testing else 'whole_result')
        images_loc = os.path.join(
            images_base_loc, action.replace(
                '_', ' ').title())
        imgs, masks, sync, angles, centers, samples_indices = co.imfold_oper.load_frames_data(
            images_loc, masks_needed=True)
        import cv2

        masks_centers = []
        xdim = 0
        ydim = 0
        conts = []
        tmp = []
        for mask, img in zip(masks, imgs):
            conts = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
            conts_areas = [cv2.contourArea(cont) for cont in conts]
            tmp.append(np.sum(mask*img >0))
            if np.sum(mask*img >0) < 500:
                masks_centers.append(None)
            else:
                cont = conts[np.argmax(conts_areas)]
                x, y, w, h = cv2.boundingRect(cont)
                if w == 0 or h == 0:
                    masks_centers.append(None)
                else:
                    masks_centers.append([y +h/2, x+w/2])
                    xdim = max(w, xdim)
                    ydim = max(h, ydim)

        cropped_imgs = []
        for img, center in zip(imgs, masks_centers):
            if center is not None:
                cropped_img = img[max(0, center[0]-ydim/2)
                                  :min(img.shape[0], center[0]+ydim/2),
                                  max(0, center[1]-xdim/2)
                                  :min(img.shape[0], center[1]+xdim/2)]
                inp_img = np.zeros((ydim, xdim))
                inp_img[:cropped_img.shape[0],
     :cropped_img.shape[1]] = cropped_img
                cropped_imgs.append(inp_img)
            else:
                cropped_imgs.append(None)
        fig = co.draw_oper.plot_utterances(frames=cropped_imgs,
                                           frames_sync=sync,
                                           ground_truth=ground_truth,
                                           breakpoints= breakpoints,
                                           labels=labels,
                                           dataset_name=action,
                                           *args, **kwargs)
        if save:
            if save_name is None:
                save_name = 'Full' + action
            fig.savefig(os.path.join(results_loc,
                                   save_name + '.pdf'))
        return fig
        '''
                categories_to_zoom=None,
                #categories_to_zoom = self.dynamic_actions,
                show_breaks=True, show_occ_tab=False,
                show_zoomed_occ=True, show_im_examples=False,
                show_fig_title=True,
               examples_num=15
        '''

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
            LOG.info('Mean Testing Times:\n\t' +str(test_times))
            if not self.online:
                self.testdata[self.test_ind]['TestTime'] = test_times

    def process_single_sample(self, data, img_count,
                              derot_angle=None, derot_center=None):
        if data is None:
            self.scores.append(None)
            return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                                        np.float64)
        if not self.frames_preproc.update(data,
                                          angle=derot_angle,
                                          center=derot_center,
                                          masks_needed=True,
                                          img_count=self.img_count,
                                          isderotated=False):
            return False, np.array([None] * len(self.train_classes))
        descriptors = [descriptor.extract() for
                    descriptor in self.features_extractors]
        if not any([desc is None for desc in descriptors]):
            for count in range(len(descriptors)):
                if self.sparsecoded == 'Features':
                    descriptors[count] = (self.action_recog.actions.
                                       coders[count].code(descriptors[count]))
                self.buffer_operators[count].update_buffer_info(
                    self.img_count, samples=descriptors[count])
                self.buffer_operators[count].add_buffer()
                descriptors[count] = self.buffer_operators[count].buffer
                if descriptors[count] is None:
                    return False, np.array([[None] *
                                            len(self.train_classes)]).astype(
                        np.float64)
                if self.sparsecoded == 'Buffer':
                    descriptors[count] = (self.action_recog.actions.
                                       coders[count].code(descriptors[count]))
                if self.ptpca:
                    descriptors[count] = self.buffer_operators[
                        count].perform_post_time_pca(
                            descriptors[count])

        else:
            return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                np.float64)

        inp = np.hstack(tuple(descriptors))
        try:
            score = (self.unified_classifier.decide(inp))
        except Exception as e:
            raise
        return True, np.array(score).reshape(1, -1)

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
        valid, score = self.process_single_sample(data, img_count,
                                                  derot_angle,
                                                  derot_center)
        self.scores.append(score)
        if not valid:
            return valid, score
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

    def extract_actions(self, scores, method='CProb', tol=0.7,
                        filterr=True):

        if filterr:
            scores = co.noise_proc.masked_filter(scores,
                                                 self.scores_filter_shape)
        extracted_actions = []
        if method == 'CProb':
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
        elif method == 'CSTD':
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
                                less_filtered_scores_std).ravel()[fmask] > 0).astype(int)
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
        if 'Sync' in self.classifiers_used:
            if extraction_method is None:
                extraction_method = 'CProb'
            if not isinstance(extraction_method, list):
                extraction_method = [extraction_method] * len(self.scores)
            if not isinstance(tol, list):
                tol = [tol] * len(self.scores)
            self.recognized_classes = {}
            for count, key in enumerate(self.scores):
                (extracted_actions, more) = self.extract_actions(
                    self.scores[key], method=extraction_method[count],
                    tol=tol[count])
                self.recognized_classes[key] = extracted_actions
        else:
            (self.recognized_classes,
             more) = self.extract_actions(
                self.scores, method=extraction_method)
            if extraction_method == 'CSTD':
                if self.test_ind is not None:
                    self.testdata[self.test_ind][
                        'Results'][
                            'LessFilteredScoresSTD'] = more[0]
                    self.testdata[self.test_ind][
                        'Results'][
                            'HighFilteredScoresSTD'] = more[1]
                    self.testdata[self.test_ind][
                        'Results'][
                            'Crossings'] = more[2]
                self.less_filtered_scores_std = more[0]
                self.high_filtered_scores_std = more[1]
                self.crossings = more[2]
        if self.test_ind is not None:
            self.testdata[self.test_ind]['Results'][
                'Actions'] = self.recognized_classes
        return self.recognized_classes

    def compute_performance_measures(
            self, recognized_classes, ground_truths, act_namespaces,
        utterances_annotation=None, save=True):
        '''
        Extract confusion matrix, accuracy and f scores from the given
        <recognized_classes> predicted classes and <ground_truths> actual
        classes. If <utterances_annotation> is given,  metrics are
        calculated (utterance level), else micro metrics are computed (frames
        level)
        '''
        from sklearn import metrics
        from scipy.linalg import block_diag
        LOG.info('Computing performance measures for ' +
                 self.classifier_savename + ' with dataset:' +
                 self.testdataname)
        y_trues = []
        y_preds = []
        utters = []
        weights = []
        confusion_matrices = []
        f1_scores = []
        existing_classes = []
        accuracies = []
        if not 'Sync' in self.classifiers_used:
            recognized_classes = {self.action_type: recognized_classes}
            ground_truths = {self.action_type: ground_truths}
            act_namespaces = {self.action_type: act_namespaces}
            if utterances_annotation is not None:
                utterances_annotation = {
                    self.action_type: utterances_annotation}
        if utterances_annotation is not None:
            prefix = 'macro'
        else:
            prefix = 'micro'
        undef_exists = False
        undef_vec = []
        for act_type in recognized_classes:
            ground_truth = np.array(ground_truths[act_type]).ravel()
            recognized_actions = np.array(recognized_classes[act_type])
            act_names = act_namespaces[act_type]
            fmask = np.isnan(ground_truth)
            ground_truth[fmask] = -1
            y_trues.append(ground_truth.astype(int))
            recognized_actions[np.isnan(recognized_actions)] = -1
            y_preds.append(np.array(recognized_actions))
            y_preds[-1][np.isnan(y_preds[-1])] = -1
            if utterances_annotation is not None:
                utters.append(np.array(utterances_annotation[act_type]))
                y_trues[-1], y_preds[-1] = co.macro_metrics.construct_vectors(
                    y_trues[-1], y_preds[-1], utters[-1])
            weights.append(len(np.unique(ground_truth)))
            fmask = y_trues[-1] != -1
            y_trues[-1] = y_trues[-1][fmask]
            y_preds[-1] = y_preds[-1][fmask]
            flag_to_set_undef = np.sum(
                y_preds[-1][:, None]
                == np.unique(y_trues[-1])[None, :], axis=1) == 0
            y_preds[-1][flag_to_set_undef] = -1
            fsc = metrics.f1_score(y_trues[-1],
                                           y_preds[-1],
                                           average=None)
            if -1 in y_preds[-1]:
                fsc = fsc[1:]
            f1_scores.append(np.atleast_2d(fsc))
            accuracies.append(metrics.accuracy_score(y_trues[-1],
                                                     y_preds[-1]))
            # now clean undefined from predicted too
            conf_mat = metrics.confusion_matrix(
                y_trues[-1], y_preds[-1])
            if -1 in y_preds[-1]:
                conf_mat = conf_mat[1:, :]
                undef_vec.append(conf_mat[:,0])
                conf_mat = conf_mat[:, 1:]
            else:
                undef_vec.append(np.zeros(conf_mat.shape[0]))
            confusion_matrices.append(conf_mat)
            classes = set(y_trues[-1].tolist() +
                          y_preds[-1].tolist())
            classes.discard(-1)
            classes = np.array(list(classes)).astype(int)
            existing_classes += (np.array(
                act_names)[classes]).tolist()
        labels = existing_classes
        labels_w_undef = (['Undefined'] + existing_classes if undef_exists
                          else existing_classes)
        f1_scores = np.concatenate(f1_scores, axis=1)
        actions_id = []
        for clas in labels_w_undef:
            actions_id.append(self.all_actions.index(clas) -1)
        undef_vec = [vec for vec_list in undef_vec for vec in vec_list ]
        confusion_matrix = block_diag(*tuple(confusion_matrices))
        confusion_matrix = np.concatenate((confusion_matrix,
                                           np.array(undef_vec).reshape(-1, 1)),
                                          axis=1).astype(int)
        accuracy = sum([accuracy * weight for accuracy, weight in
                        zip(accuracies, weights)]) / float(sum(weights))
        if not self.online:
            self.testdata[self.test_ind]['Accuracy'][
                prefix.title()] = accuracy
            self.testdata[self.test_ind]['PartialAccuracies'][
                prefix.title()] = accuracies
            self.testdata[self.test_ind]['FScores'][
                prefix.title()] = [f1_scores, actions_id]
            self.testdata[self.test_ind][
                'ConfMat'][prefix.title()] = [confusion_matrix,
                                              actions_id]
            self.testdata[self.test_ind][
                'Labels'][prefix.title()] = labels

        LOG.info(prefix.title() + ' F1 Scores: \n' +
                 np.array2string(f1_scores))
        LOG.info(
            prefix.title() + ' Confusion Matrix: \n' +
            np.array2string(
                confusion_matrix))
        if 'Sync' in self.classifiers_used:
            LOG.info(prefix.title() + ' Partial Accuracies:' + str(accuracies))
        LOG.info(prefix.title() + ' Accuracy: ' + str(accuracy))
        LOG.info('Labels of actions:' + str(labels))

    def write_metrics_to_file(self):
        accuracies = self.testdata[self.test_ind]['Accuracy']
        partial_accuracies = self.testdata[
            self.test_ind]['PartialAccuracies']
        if 'Sync' in self.classifiers_used:
            acc_labels = (['Class.' + str(clas) for clas in
                           self.parameters['sub_classifiers']] +
                          ['Total Mean'])
            for cnt,acc_label in enumerate(acc_labels):
                if 'mathregular' in acc_label:
                    import re
                    acc_labels[cnt] = re.sub('\\\\mathregular{(.*)}',
                           lambda x: x.group(1), acc_label)
            ylabels = []
            data = []
            extra_locs = []
            for key in partial_accuracies:
                ylabels.append(key)
                if not data:
                    data += [partial_accuracies[key], accuracies[key]]
                    extra_locs = ['right']
                else:
                    data += [np.hstack((partial_accuracies[key],
                                        accuracies[key]))]
                    extra_locs.append('bot')
            with open(os.path.join(self.save_fold, 'sync_accuracy.tex'), 'w') as out:
                out.write(co.latex.array_transcribe(data,
                                                    xlabels=acc_labels,
                                                    ylabels=ylabels,
                                                    sup_x_label='Accuracy',
                                                    extra_locs=extra_locs))
        f1_scores = self.testdata[self.test_ind]['FScores']
        for typ in f1_scores:
            with open(os.path.join(self.save_fold, typ.title() +'_F1_Scores.tex'), 'w') as out:
                out.write(co.latex.array_transcribe([f1_scores[typ][0],
                                                     np.atleast_2d(accuracies[typ])],
                                                    xlabels=np.concatenate((self.testdata[
                                                        self.test_ind]['Labels'][typ],
                                                        ['Accuracy']), axis=0),
                                                    sup_x_label=typ.title() +
                                                    ' Metrics',
                                                    extra_locs=['right']))
        confmats = self.testdata[self.test_ind]['ConfMat']
        for typ in confmats:
            with open(os.path.join(self.save_fold,
                                   typ.title()+'_Confusion_Matrix.tex'), 'w') as out:
                ylabels = self.testdata[self.test_ind]['Labels'][typ]
                xlabels = ylabels[:]
                if confmats[typ][0].shape[0] != confmats[typ][0].shape[1]:
                    xlabels += ['Undefined']
                out.write(co.latex.array_transcribe(confmats[typ][0],
                                                    ylabels=ylabels,
                                                    xlabels=xlabels,
                                                    sup_x_label='Predicted',
                                                    sup_y_label='Actual',
                                                    title=typ.title() +
                                                    ' Metrics',
                                                    wrap=False))

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
                parameters[0]['descriptors']))
            row.append(parameters[0]['sparsecoded'])
            if parameters[0]['sparsecoded']:
                row.append('\n'.join(
                    ['%d' % parameters[0]['sparse_params'][feature]
                     for feature in parameters[0]['descriptors']]))
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
                except BaseException:                    row.append('')
                try:
                    if parameters['testing_params'][
                            'post_scores_processing_method'] == 'CSTD':
                        row.append('%d' % parameters[0]['dynamic_params'][
                            'filter_window_size'])
                except BaseException:                    row.append('')

                try:
                    row.append(str(parameters[0]['PTPCA']))
                except BaseException:                    row.append('')
                if row[-1] != '' and row[-1] == 'True':
                    try:
                        row.append('%d' % parameters[0]['PTPCA_params'][
                            'PTPCA_components'])
                    except BaseException:                        row.append('')

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
        except BaseException:            LOG.warning('No such value inside catalog')

    def load_all_test_instances(self, test_ind):
        available_test_instances = {}
        test_name = self.available_tests[test_ind]
        import ast
        loaded_instances, keys_list = co.file_oper.load_labeled_data(['Testing'],
                                                                     fold_lev=1, all_inside=True)
        for _id in loaded_instances:
            loaded_instance = loaded_instances[_id]
            if dict(keys_list[_id][0])['Test'] == test_name:
                available_test_instances[_id] = loaded_instance

        return available_test_instances

    def extract_test_results_instances(self, test_ind, key,*keys):
        if self.test_instances is None:
            self.test_instances = self.load_all_test_instances(test_ind)
        res = []
        for entry in co.dict_oper.create_sorted_dict_view(self.test_instances):
            if entry[1] is None:
                res.append(None)
            else:
                res.append(co.dict_oper.lookup(entry[0], key,*keys))
        return res

    def correlate_with_ground_truth(self, save=True, display=False,
                                    display_all=False, compute_perform=True,
                                    utterances_inds=None):
        '''
        Plot results with title <title>
        <display_all> if a plot of all the classifiers results is wanted
        Do not classifiers_used this function if more than one classifiers are to be run in
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
            if utterances_inds is not None:
                self.compute_performance_measures(
                    self.recognized_classes,
                    ground_truths=self.test_ground_truth,
                    utterances_annotation= utterances_inds,
                    act_namespaces=self.train_classes,
                    save=save)
            else:
                LOG.warning('Utterances Indices not passed to' +
                            ' function. MacroMetrics won\'t be' +
                            ' computed.')
            if save and self.save_fold is not None:
                self.write_metrics_to_file()

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
                        ['Classifier'], just_catalog=True
                    )
                    if available_ids is not None:
                        try:
                            iterat_name = available_ids[str(
                                self.classifier_id)]
                        except KeyError:
                            iterat_name = str(len(available_ids))
                    else:
                        iterat_name = str(0)
                    higher_acc = [0]
                    if 'Sync' in self.classifiers_used:
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
            markers_sizes = []
            alphas = []
            zorders = []
            colors = []
            xticks = None
            width = 1
            dec_q = 0.3
            min_q = 0.2
            from matplotlib.cm import get_cmap
            available_colors = get_cmap('tab20b')
            if self.crossings is not None:
                xticks = self.crossings
                expanded_xticks = np.zeros_like(self.test_ground_truth)
                expanded_xticks[:] = None
                expanded_xticks[xticks] = -1
                plots.append(expanded_xticks)
                alphas.append(1)
                markers.append('o')
                markers_sizes.append(5)
                if 'Sync' in self.classifiers_used:
                    labels.append('Utterances\n'+
                                   'Predicted\n'+
                                  'break-\npoints\n'+
                                  'by Cl_${dyn}$')
                else:
                    labels.append('Utterances\n'+
                                   'Predicted\n'+
                                  'break-\npoints')
                colors.append('green')
                linewidths.append(1)
                zorders.append(2)
            for count in range(len(higher_acc)):
                syncplots = []
                plots.append(iterat[count])

                if len(higher_acc) == 1:
                    labels.append('Predictions')
                else:
                    labels.append('Class. ' +
                              str(iterat_name[count]) +
                              ' Predictions')
                markers.append(',')
                markers_sizes.append(10)
                linewidths.append(width)
                colors.append(available_colors(count))
                alphas.append(1)
                zorders.append(3)
                width -= dec_q
                width = max(min_q, width)
            if 'Sync' in self.classifiers_used:
                yticks = []
                for key in self.train_classes:
                    yticks += list(self.train_classes[key])
            else:
                yticks = self.train_classes
            ylim = (-1, len(yticks) + 1)
            fig, lgd, axes = self.plot_result(np.vstack(plots).T, labels=labels,
                                              xticks_locs=xticks, ylim=ylim,
                                              yticks_names=yticks,
                                              markers=markers,
                                              linewidths=linewidths,
                                              alphas=alphas,
                                              xlabel='Frames',
                                              zorders=zorders,
                                              markers_sizes= markers_sizes,
                                              info='Classification Results',
                                              save=False)
            if self.test_breakpoints is not None:
                if 'Sync' in self.classifiers_used:
                    tg_ref = 0
                    for key in (self.test_ground_truth):
                        self.draw_breakpoints(axes,
                                              self.test_breakpoints[key],
                                              yticks)
                else:
                    self.draw_breakpoints(axes,
                                          self.test_breakpoints, yticks)
            if save:
                save_info = 'Classification Results'
                if self.crossings is not None:
                    save_info += ' with Crossings'
                self.save_plot(fig, lgd, info=save_info)
            return True

    def draw_breakpoints(self, axes, breakpoints, yticks, lw=3,
                         zorder=2):
        '''
        Draws colored ground truth
        '''
        from matplotlib.cm import get_cmap
        cmap = get_cmap('Spectral')
        max_plotpoints_num = 0
        for act in breakpoints:
            max_plotpoints_num = max(max_plotpoints_num,
                                     len(breakpoints[act][0]))
        c_num = max_plotpoints_num
        yticks = [ytick.lower() for ytick in yticks]
        for act_cnt, act in enumerate(breakpoints):
            drawn = 0
            for cnt, (start, end) in enumerate(zip(breakpoints[act][0],
                                                   breakpoints[act][1])):
                gest_dur = np.arange(int(start), int(end))
                if act.lower() in yticks:
                    axes.plot(gest_dur, np.ones(gest_dur.size) *(
                        yticks.index(act.lower()) ),
                        color=cmap(cnt/float(c_num))
                        , linewidth=lw,
                        solid_capstyle="butt", zorder=zorder)


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
        classifier.test_ground_truth = co.gd_oper.construct_ground_truth(
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
                                         name='actions', sparsecoding_level=False,
                                         sparse_dim_rat=None, test_against_all=False,
                                         visualize_feat=False, kernel=None,
                                         descriptors='GHOG', ptpca=False,
                                         ptpca_components=1,
                                         just_sparse=False,
                                         debug=False,
                                         classifiers_used='SVM',
                                         action_type='Dynamic',
                                         post_scores_processing_method='CSTD'):
    '''
    Constructs an SVM classifier with input 3DHOF and GHOG descriptors
    '''
    if sparsecoding_level:
        if sparse_dim_rat is None:
            sparse_dim_rat = co.CONST['sparse_dim_rat']
    classifier = Classifier('INFO', action_type=action_type,
                            name=name, sparse_dim_rat=sparse_dim_rat,
                            sparsecoding_level=sparsecoding_level,
                            descriptors=descriptors,
                            kernel=kernel, ptpca=ptpca,
                            ptpca_components=ptpca_components,
                            classifiers_used=classifiers_used, post_scores_processing_method=                            post_scores_processing_method)
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
                                         descriptors='3DXYPCA',
                                         post_scores_processing_method='CProb',
                                         for_app=False):
    '''
    Constructs a random forests passive_actions classifier with input 3DXYPCA descriptors
    '''
    classifier = Classifier('INFO', action_type='Passive',
                            name='actions', classifiers_used='RDF',
                            sparsecoding_level=False,
                            descriptors=descriptors,
                            post_scores_processing_method=
                            post_scores_processing_method, for_app=for_app)
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
        classifier.test_ground_truth = co.gd_oper.construct_ground_truth(
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
        train=[0],descriptors=['GHOG', 'ZHOF', '3DHOF', '3DXYPCA'],
        sparsecoding_level=True,just_sparse=True, debug=False)
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
        test_against_all=True,
        ptpca=False,
        classifiers_used='RDF',
        descriptors=['GHOG','ZHOF'],
        post_scores_processing_method='CSTD')
    exit()
    '''
    ACTIONS_CLASSIFIER_SPARSE = construct_dynamic_actions_classifier(train=True,
                                                                     coders_retrain=False,
                                                                     test=True,
                                                                     visualize=True,
                                                                     test_against_all=True,
                                                                     sparsecoding_level='Features',
                                                                     classifiers_used='RDF')
    ACTIONS_CLASSIFIER_SIMPLE_POST_PCA = construct_dynamic_actions_classifier(
        train=True,
        test=True,
        visualize=True,
        test_against_all=True,
        ptpca=True,
        ptpca_components=2)
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['3DHOF'],ptpca=False,sparsecoding_level=False)
    '''
    '''
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['ZHOF'], ptpca=False, sparsecoding_level=False)

    '''
    '''
    ACTIONS_CLASSIFIER_SPARSE_WITH_3DHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', '3DHOF'], coders_retrain=False, sparsecoding_level=True,
        kernel='linear')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF_POST_PCA = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', '3DHOF'], ptpca=True, ptpca_components=4)
    construct_dynamic_actions_classifier(
        #debugging, train=True, test=True,
        train=True, test=True,
        visualize=True, test_against_all=True,
        descriptors=['GHOG', '3DHOF'], ptpca=True, sparsecoding_level=True,
        ptpca_components=2)
    '''
    '''
    ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', '3DHOF'], kernel='linear')
    '''
    '''
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', 'ZHOF'])
    '''
    '''
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['3DXYPCA','GHOG','3DHOF','ZHOF'], classifiers_used='SVM')
    '''
    # Let's try RDF for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', 'ZHOF'], ptpca=False, sparsecoding_level=False,
        classifiers_used='RDF')
    exit()
    # Let's try RDF with all descriptors for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', '3DHOF', '3DXYPCA'], ptpca=False, sparsecoding_level=False,
        classifiers_used='RDF')

    # Let's try RDF for all descriptors for all actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', '3DHOF', '3DXYPCA'], action_type='All', ptpca=False, sparsecoding_level=False,
        classifiers_used='RDF')
    # Let's try RDF with all descriptors for dynamic actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', 'ZHOF', '3DXYPCA'], action_type='Dynamic', ptpca=False, sparsecoding_level=False,
        classifiers_used='RDF')

    # Let's try RDF for all descriptors for all actions
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', 'ZHOF', '3DXYPCA'], action_type='All', ptpca=False, sparsecoding_level=False,
        classifiers_used='RDF')
    ACTIONS_CLASSIFIER_SPARSE_WITH_ZHOF = construct_dynamic_actions_classifier(
        train=True,
        test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', 'ZHOF'], coders_retrain=False, sparsecoding_level=True,
        kernel='linear')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF_POST_PCA = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', 'ZHOF'], ptpca=True, ptpca_components=4)
    construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG', 'ZHOF'], ptpca=True,
        sparsecoding_level=True, coders_retrain=False, ptpca_components=4)
    '''
    ACTIONS_CLASSIFIER_SPARSE_WITH_ZHOF_RBF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG','ZHOF'], coders_retrain=False, sparsecoding_level=True,
        kernel='rbf')
    ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF_RBF = construct_dynamic_actions_classifier(
        train=True, test=True, visualize=True, test_against_all=True,
        descriptors=['GHOG','ZHOF'], coders_retrain=False, sparsecoding_level=False,
        kernel='rbf')
    ACTIONS_CLASSIFIER_SIMPLE_RBF = construct_dynamic_actions_classifier(
        train=False,
        test=False,
        visualize=False,
        sparsecoding_level=False,
        test_against_all=True,
        kernel='rbf')
    '''
    # construct classifiers comparative table
    tmp = Classifier(descriptors=[''])
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
