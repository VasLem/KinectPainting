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
# pylint: disable=no-member


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class Classifier(object):
    '''
    Class to hold all Classifier specific methods.
    <features>:['pca','ghog','3dhof']
    <ispassive>:True if no buffers are used
    <use_sparse> is True if sparse coding is used
    '''

    def __init__(self, log_lev='INFO',
                 visualize=False, masks_needed=True,
                 buffer_size=co.CONST['buffer_size'],
                 sparse_dim=co.CONST['sparse_dim'],
                 features=['3DXYPCA', 'GHOG', '3DHOF'],
                 post_pca=False,
                 post_pca_components=1,
                 ispassive=False,
                 use='svms', num_of_cores=4, name='',
                 num_of_estimators=None,
                 add_info=None,
                 use_sparse=True,
                 kernel=None, *args, **kwargs):

        # General configuration
        self.kernel = kernel
        self.num_of_estimators = num_of_estimators
        self.sparse_dim = sparse_dim
        if use == 'svms' and kernel is None:
            self.kernel = 'linear'
        elif use == 'forest' and num_of_estimators is None:
            self.num_of_estimators = co.CONST['RF_trees']
        classifier_params = {'kernel': self.kernel,
                             'num_of_estimators': self.num_of_estimators}
        dynamic_params = {'buffer_size': buffer_size,
                          'buffer_confidence_tol':co.CONST['buffer_confidence_tol'],
                          'filt_window':co.CONST['STD_big_filt_window'],
                          'filt_window_confidence_tol':
                          co.CONST['filt_window_confidence_tol'],
                          'post_pca':post_pca,
                          'post_pca_components':post_pca_components}
        if use_sparse:
            if not isinstance(sparse_dim, list) :
                   sparse_dim = [sparse_dim] * len(features)
            if len(list(sparse_dim)) != len(features):
                raise Exception('<sparse_dim> should be either an integer/None or'+
                                ' a list with same length with <features>')
            sparse_params = dict(zip(features,sparse_dim))
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
        LOG.info('Extracting: ' + str(features))
        self.parameters = {'classifier': use,
                           'features': features,
                           'dynamic_params': dynamic_params,
                           'classifier_params': classifier_params,
                           'sparse_params': sparse_params,
                           'passive': ispassive,
                           'sparsecoded': use_sparse,
                           'testing': False,
                           'testing_params':testing_params,
                           'dynamic_actions':self.dynamic_actions,
                           'passive_actions':self.passive_actions}
        self.features = features
        self.add_info = add_info
        self.log_lev = log_lev
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.masks_needed = masks_needed
        self.ispassive = ispassive
        self.use = use
        self.num_of_cores = num_of_cores
        self.name = name
        if ispassive:
            info = 'passive '
        else:
            info = 'dynamic '
        info = info + self.name + ' ' + self.use + ' '
        info += 'using'
        if use_sparse:
            info += ' sparsecoded'
        for feature in features:
            info += ' ' + feature
        info += ' features '
        if self.use == 'svms' or self.use == 'mixed':
            info += 'with ' + self.kernel + ' kernel'
        elif self.use == 'forest':
            info += ('with ' + str(self.num_of_estimators) +
                     ' estimators')
        if not ispassive:
            info += ' with buffer size ' + str(self.buffer_size)
        if use_sparse:
            info += ' sparse dimension(s) ' + str(self.sparse_dim)
        if post_pca:
            info += ' with post time-pca'
        self.full_info = info.title()
        if self.add_info:
            info += self.add_info
        if self.use == 'svms':
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
        elif not self.use == 'mixed':
            from sklearn.ensemble import RandomForestClassifier
            self.classifier_type =\
                RandomForestClassifier(self.num_of_estimators)

        self.unified_classifier = None
        self.sparsecoded = use_sparse
        self.sparse_coders = None  # is loaded from memory

        # Training variables
        self.one_v_all_traindata = None
        self.train_ground_truth = None  # is loaded from memory after training
        self.train_classes = None  # is loaded from memory after training

        # Testing general variables
        self.scores = None
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
        except (IOError,EOFError):
            LOG.info('Non existent trained classifiers file')
            self.trained_classifiers = {}
        try:
            with open('all_test_scores.pkl', 'r') as inp:
                self.all_test_scores = pickle.load(inp)
        except (IOError,EOFError):
            LOG.info('Non existent testing scores file')
            self.all_test_scores = {}
        try:
            with open('trained_classifiers_list.yaml','r') as inp:
                try:
                    self.classifiers_list = yaml.load(inp)
                except yaml.YAMLError as exc:
                    raise exc
        except (IOError, EOFError):
            if len(self.trained_classifiers) > 0:
                LOG.warning('Missing trained classifiers list file, '+
                            'reconstructing..')
                with open('trained_classifiers_list.yaml','w') as out:
                    for count,classifier in enumerate(self.trained_classifiers):
                        out.write(classifier + ':' + str(count)+'\n')
                    self.classifiers_list[classifier] = str(count)
        self.coders_to_train = []
        if self.sparsecoded:
            self.sparse_coders = [None] * len(
                self.parameters['features'])
            try:
                with open('all_sparse_coders.pkl', 'r') as inp:
                    LOG.info('Loading coders from: ' +
                             'sparse_coders.pkl')
                    self.all_sparse_coders = pickle.load(inp)
                    for feat_count,feature in enumerate(
                        self.parameters['features']):
                        try:
                            self.sparse_coders[feat_count] =\
                                    self.all_sparse_coders[
                                        feature + ' ' +
                                        str(self.parameters['sparse_params'][
                                            feature])]
                        except KeyError:
                            self.coders_to_train.append(feat_count)
            except (IOError, EOFError):
                self.all_sparse_coders = {}
                self.coders_to_train = range(len(self.parameters['features']))
            self.parameters['sparse_params']['trained_coders'] = len(
                self.coders_to_train)==0
        else:
            self.sparse_coders = None
        #parameters bound variables
        self.features_extraction = ara.FeatureExtraction(self.parameters)
        self.action_recog = ara.ActionRecognition(
            self.parameters,
            coders=self.sparse_coders,
            log_lev=log_lev)

    def initialize_classifier(self, classifier):
        self.unified_classifier = classifier
        if self.use == 'svms' or self.use == 'mixed':
            self.decide = self.unified_classifier.decision_function
            self.predict = self.unified_classifier.predict
        elif self.use == 'forest':
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

    def run_training(self, dict_train_act_num=None, coders_retrain=False,
                     classifiers_retrain=False, test_against_training=False,
                     training_datapath=None, classifier_savename=None,
                     num_of_cores=4, buffer_size=None, classifier_save=True,
                     max_act_samples=None,
                     min_dict_iterations=3,
                     visualize_feat=False):
        '''
        <Arguments>
        For testing:
            If a testing using training data is to be used, then
            <test_against_training> is to be True
        For coders training:
            Train coders using <dict_train_act_num>-th action. Do not
            train coders if save file already exists or <coders_retrain>
            is False. If <dict_train_act_num> is <None> , then train
            coders using all available actions
        For svm training:
            Train ClassifierS with <num_of_cores> and buffer size <buffer_size>.
            Save them if <classifier_save> is True to <classifiers_savepath>. Do not train
            if <classifiers_savepath> already exists and <classifiers_retrain> is False.
        '''
        self.parameters['testing'] = False
        LOG.info(self.full_info + ':')
        if self.sparsecoded:
            if coders_retrain:
                if isinstance(coders_retrain, list):
                    self.coders_to_train = coders_retrain
                else:
                    self.coders_to_train = range(len(self.parameters['features']))
                self.parameters['sparse_params']['trained_coders'] = False
        if classifier_savename is None:
            classifier_savename = 'trained_'
            classifier_savename += self.full_info.replace(' ', '_').lower()
        self.classifier_savename = classifier_savename
        if not classifier_savename in self.trained_classifiers:
            LOG.info('Missing trained classifier:' +
                     self.full_info)
            LOG.info('Classifier will be retrained')
            classifiers_retrain = True
        if buffer_size is not None:
            self.buffer_size = buffer_size
        if classifiers_retrain or self.coders_to_train or test_against_training:
            if self.coders_to_train:
                max_act_samples = None
            self.prepare_training_data(training_datapath, max_act_samples,
                                       visualize_feat=visualize_feat)
        if self.sparsecoded:
            self.process_coders(train_act_num=dict_train_act_num,
                                      min_iterations=min_dict_iterations,
                                      max_act_samples=max_act_samples)

        if self.sparsecoded and self.coders_to_train and (
            classifiers_retrain or test_against_training):
            #Enters only if coders were not initially trained or had to be
            #retrained. Otherwise, sparse features are computed when
            #<Action.add_features> is called
            LOG.info('Making Sparse Features..')
            self.action_recog.actions.update_sparse_features(
                self.sparse_coders,
                max_act_samples=max_act_samples,
                fss_max_iter=self.fss_max_iter,
                last_only=False)
        self.process_training(num_of_cores, classifiers_retrain,
                              classifier_savename, classifier_save,
                              test_against_training)

    def prepare_training_data(self, path=None, max_act_samples=None,
                              fss_max_iter=100, visualize_feat=False):
        '''
        Read actions from the <path> and name them according to their parent
        folder name
        '''
        self.fss_max_iter = fss_max_iter
        LOG.info('Adding actions..')
        if path is None:
            path = co.CONST['actions_path']
        self.train_classes = [name for name in os.listdir(path)
                              if os.path.isdir(os.path.join(path, name))][::-1]
        if self.ispassive:
            if self.passive_actions is not None:
                self.train_classes = [clas for clas in self.train_classes if clas
                                      in self.passive_actions]
        #DEBUGGING
        if not self.ispassive:
            if self.dynamic_actions is not None:
                self.train_classes = [clas for clas in self.train_classes if clas
                                      in self.dynamic_actions]
        for action in self.train_classes:
            if not isinstance(visualize_feat, bool):
                try:
                    visualize = action.startswith(visualize_feat)
                except TypeError:
                    visualize = action in visualize_feat
            else:
                visualize = visualize_feat
            LOG.info('Action:'+action)
            self.action_recog.add_action(name=action,
                                         data=os.path.join(path, action),
                                         use_dexter=False,
                                         ispassive=self.ispassive,
                                         max_act_samples=max_act_samples,
                                         fss_max_iter=fss_max_iter,
                                         visualize_=visualize)

    def process_coders(self, train_act_num=None,
                             coders_savepath=None, min_iterations=10,
                             max_act_samples=None):
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
                coders_to_train = self.coders_to_train,
                codebooks_dict=self.all_sparse_coders,
                min_iterations=min_iterations)
            self.parameters['sparse_params']['trained_coders'] = True
            with open(coders_savepath, 'w') as out:
                pickle.dump(self.all_sparse_coders, out)
            self.sparse_coders[
                self.coders_to_train] = (
                    self.action_recog.sparse_helper.sparse_coders[
                        self.coders_to_train])
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
            raise('savepath needed')
        if (against_training or retrain or
            not savepath in self.trained_classifiers):
            if against_training and (not retrain or savepath in
                                     self.trained_classifiers):
                LOG.info('Preparing Classifiers Train Data for testing..')
            else:
                LOG.info('Preparing Classifiers Train Data..')
            if not self.ispassive:
                acts_buffers = [action.retrieve_buffers()
                               for action in self.action_recog.actions.actions]

                acts_buffers = [np.swapaxes(buffers,1,2).reshape(
                    buffers.shape[0],-1) for buffers in acts_buffers]
                LOG.info('Train Data has ' + str(len(acts_buffers)) +
                         ' buffer lists. First buffer list has length ' +
                         str(len(acts_buffers[0])) +
                         ' and last buffer has shape ' +
                         str(acts_buffers[0][-1].shape))
                multiclass_traindata = acts_buffers
            else:
                if self.sparsecoded:
                    multiclass_traindata = [action.retrieve_sparse_features() for
                        action in
                        self.action_recog.actions.actions]
                else:
                    multiclass_traindata = [action.retrieve_features() for
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
             self.train_classes,
             _) = self.trained_classifiers[savepath]
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
            self.initialize_classifier(self.classifier_type.fit(self.one_v_all_traindata,
                                                                self.train_ground_truth))
        if save and not loaded:
            LOG.info('Saving trained Classifiers to trained' +
                     ' classifiers dictionary with name: ' + savepath)
            found = False
            leng = 0
            if savepath  not in self.classifiers_list:
                with open('trained_classifiers_list.yaml','a') as out:
                    out.write(savepath+': '+
                              str(len(self.classifiers_list))+'\n')
                self.classifiers_list[savepath] = str(len(self.classifiers_list))

            self.trained_classifiers[savepath] = (self.unified_classifier,
                                                  self.train_classes,
                                                  self.parameters)
            with open('trained_classifiers.pkl','w') as out:
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
            ispassive=self.ispassive,
            fss_max_iter=100)
        if not self.ispassive:
            self.test_sync = frame_inds[0]
            test_buffers_start_inds = frame_inds[1]
            test_buffers_end_inds = frame_inds[2]
            testdata = np.swapaxes(features,1,2).reshape(
                    features.shape[0],-1)
            return testdata, test_buffers_start_inds, test_buffers_end_inds
        else:
            self.test_sync = frame_inds
            testdata = features
            return testdata

    def construct_ground_truth(self, data=None, ground_truth_type=None,
                               testing=True):
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
            for root, dirs, files in os.walk(data):
                root_separ = root.split(os.path.sep)
                if root_separ[-2] != co.CONST['hnd_mk_fold_name']:
                    paths += [os.path.join(root, filename)
                              for filename in sorted(files) if
                              filename.endswith('.png')]
            files = [os.path.basename(path) for path in paths]
            files = sorted(files)
            if not paths:
                raise Exception(data + ' does not include any png file')
        else:
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
            if testing:
                class_match = {}
                for key in keys:
                    try:
                        class_match[key] = self.train_classes.index(key)
                    except ValueError:
                        ground_truth_init.pop(key, None)
                if not ground_truth_init:
                    raise Exception(
                        'No classes found matching with training data ones')
            else:
                class_match = {}
                for count, key in enumerate(keys):
                    class_match[key] = count
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
            else:
                iterat = np.where(data)[0]
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
                    if testing:
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
                                  .split(os.path.sep)[-3]
                                  ] = int(filter(str.isdigit, os.path.basename(
                                      filename)))
            keys = ground_truth_init.keys()
            if testing:
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
                ground_truth[int(filter(str.isdigit, os.path.basename(fil)
                                        ))] = ground_val

        else:
            raise Exception('Invalid ground_truth_type\n' +
                            self.construct_ground_truth.__doc__)
        return ground_truth

    def masked_mean(self, data, win_size):
        mask = np.isnan(data)
        K = np.ones(win_size, dtype=int)
        return np.convolve(np.where(mask, 0, data), K) / np.convolve(~mask, K)

    def upgr_filter(self, data, win_size):
        '''
        Mean filter data with missing values, along axis 1
        '''
        if len(data.shape) == 1:
            inp = np.atleast_2d(data).T
        else:
            inp = data
        return np.apply_along_axis(self.masked_mean,
                                   0, inp, win_size)[:-win_size + 1]

    def plot_result(self, data, info=None, save=True, xlabel='Frames', ylabel='',
                    labels=None, colors=None, linewidths=None,
                    xticks_names=None, yticks_names=None, xticks_locs=None,
                    yticks_locs=None, markers=None, ylim=None, xlim=None):
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
        while len(colors) < data.shape[1]:
            colors += [tuple(np.random.random(3))]
        if linewidths is None:
            linewidths = [1] * data.shape[1]
        if labels is not None:
            for count in range(data.shape[1]):
                axes.plot(data[:, count], label='%s' % labels[count],
                          color=colors[count],
                          linewidth=linewidths[count],
                          marker=markers[count])
            lgd = self.put_legend_outside_plot(axes)
        else:
            for count in range(data.shape[1]):
                axes.plot(data[:, count], colors=colors[count])
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
                     testdatapath=None, *kwargs):
        '''
        Initializes paths and names used in testing to save, load and visualize
        data.
        Built as a convenience method, in case <self.run_testing> gets overriden.
        '''
        self.parameters['testing'] = True
        self.parameters['testing_params']['online'] = online
        self.tester = ara.Action(self.parameters,name='test',coders=self.sparse_coders)
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
        if (save or load):
            self.classifier_folder = str(self.classifiers_list[
                self.classifier_savename])
            fold_name = self.classifier_folder
            self.save_fold = os.path.join(
                co.CONST['results_fold'], 'Classification', fold_name)
            if self.add_info is not None:
                self.save_fold = os.path.join(
                    self.save_fold, self.add_info.replace(' ', '_').lower())
            self.save_fold = os.path.join(self.save_fold, self.testdataname)
            makedir(self.save_fold)

            if scores_savepath is None:
                self.scores_savepath = self.testdataname + '_scores_for_'
                self.scores_savepath += self.full_info.replace(' ',
                                                               '_').lower()
                self.scores_savepath += '.pkl'
            else:
                self.scores_savepath = scores_savepath

    def run_testing(self, data=None, derot_angle=None, derot_center=None,
                    online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=co.CONST['test_actions_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=False, testname=None, display_scores=True,
                    construct_gt=True, just_scores=False, testdatapath=None):
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
        if not online:
            LOG.info('Testing:' + data)
        if isinstance(data, tuple):
            derot_angle = data[1]
            derot_center = data[2]
            data = data[0]
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
                                  'STD_big_filt_window'],
                              testdatapath=testdatapath)
        if not online:
            if (load and (self.classifier_savename in self.all_test_scores)
             and (self.testdataname in
             self.all_test_scores[self.classifier_savename])):
                LOG.info('Loading saved scores, created by'
                         +'testing \'' + self.full_info + '\' with \'' +
                         self.testdataname + '\'')
                (self.scores, self.test_sync) = self.all_test_scores[self.classifier_savename][
                    self.testdataname]
            else:
                if self.use == 'svms':
                    LOG.info('Classifier contains ' +
                             str(len(self.unified_classifier.estimators_)) + ' estimators')
                if against_training:
                    LOG.info('Testing Classifiers against training data..')
                    self.scores = self.decide(
                        self.one_v_all_traindata)
                else:
                    if not self.ispassive:
                        (testdata,
                         test_buffers_start_inds,
                         test_buffers_end_inds) = self.offline_testdata_processing(
                             data)
                    else:
                        testdata = self.offline_testdata_processing(
                            data)
                    self.testdata = testdata
                    LOG.info(self.full_info + ':')
                    LOG.info('Testing Classifiers using testdata with size: '
                             +str(testdata.shape))
                    self.scores = self.decide(
                        testdata)
                    if not self.ispassive:
                        expanded_scores = np.zeros((self.test_sync[-1] + 1,
                                                    self.scores.shape[1]))
                        expanded_scores[:] = np.NaN
                        for score, start, end in zip(self.scores,
                                                     test_buffers_start_inds,
                                                     test_buffers_end_inds):
                            expanded_scores[start:end + 1, :] = score[None, :]
                        self.scores = expanded_scores
                    if save:
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
                self.test_ground_truth = self.construct_ground_truth(
                    data, ground_truth_type)
            if not just_scores:
                self.classify_offline(save=save)
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
        self.plot_result(np.array(self.filtered_scores),
                         labels=self.train_classes,
                         xlabel='Frames',
                         save=save)
        LOG.info(self.name.title() + ':')
        if (self.action_recog.actions.
                features_extract is not None):
            times_mat = []
            hor_labels = [
                'Mean(ms)', 'Max(ms)', 'Min(ms)', 'Median(ms)']
            ver_labels = []
            orient = []
            _t_ = np.array(
                self.action_recog.actions.preproc_time) * 1000
            if len(self.action_recog.actions.preproc_time) > 0:
                preproc_t = np.array([np.mean(_t_), np.max(_t_),
                                      np.min(_t_), np.median(_t_)])
                LOG.info('Mean preprocessing time ' +
                         str(preproc_t[0]) + ' ms')
                LOG.info('Max preprocessing time ' +
                         str(preproc_t[1]) + ' ms')
                LOG.info('Min preprocessing time ' +
                         str(preproc_t[2]) + ' ms')
                LOG.info('Median preprocessing time ' +
                         str(preproc_t[3]) + ' ms')
                times_mat.append(preproc_t)
                orient.append('bot')
                ver_labels.append('Preprocessing')
            try:
                _t_ = np.atleast_2d(np.array(
                    self.action_recog.actions.sparse_time)) * 1000
                _t_ = np.array(_t_).T
                if len(_t_[0]) > 0:
                    sparse_t = np.concatenate([np.mean(_t_, axis=1)[:, None],
                                               np.max(_t_, axis=1)[
                        :, None],
                        np.min(_t_, axis=1)[
                        :, None],
                        np.median(_t_, axis=1)[:,
                                               None]], axis=1)
                    LOG.info('Mean sparse coding time ' +
                             str(sparse_t[:, 0]) + 'ms')
                    LOG.info('Max sparse coding time ' +
                             str(sparse_t[:, 1]) + ' ms')
                    LOG.info('Min sparse coding time ' +
                             str(sparse_t[:, 2]) + ' ms')
                    LOG.info('Median sparse coding time ' +
                             str(sparse_t[:, 3]) + ' ms')
                    times_mat.append(sparse_t)
                    orient.append('bot')
                    ver_labels += ['Sparse Coding '
                                   + feature for feature in
                                   self.parameters['features']]
            except TypeError:
                pass
            _t_ = np.array(self.action_recog.actions.
                           features_extract.extract_time) * 1000
            if len(_t_) > 0:
                feat_t = np.array([np.mean(_t_), np.max(_t_),
                                   np.min(_t_), np.median(_t_)])
                LOG.info('Mean feature extraction time ' +
                         str(feat_t[0]) + ' ms')
                LOG.info('Max feature extraction time ' +
                         str(feat_t[1]) + ' ms')
                LOG.info('Min feature extraction time ' +
                         str(feat_t[2]) + ' ms')
                LOG.info('Median feature extraction time ' +
                         str(feat_t[3]) + ' ms')
                ver_labels.append('Feature Extaction')
                times_mat.append(feat_t)
                orient.append('bot')

            time_array = co.latex.array_transcribe(times_mat, xlabels=hor_labels,
                                                   ylabels=ver_labels,
                                                   extra_locs=orient[:-1])
            if save:
                with open(os.path.join(self.save_fold, 'times.tex'),
                          'w') as out:
                    out.write(time_array)

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
        if not self.ispassive:
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
        self.features_extraction.update(data,
                                        angle=derot_angle,
                                        center=derot_center,
                                        masks_needed=True,
                                        img_count=self.img_count,
                                        isderotated=False)
        features = self.features_extraction.extract_features()
        valid = False
        if features is not None:
            valid, _ = self.tester.add_features_group(self.img_count, features=features)
        if not valid or features is None:
            self.scores_exist.append(False)
            return False, np.array([None]*len(self.train_classes))
        else:
            self.scores_exist.append(True)
        if not self.sparsecoded:
            inp = self.tester.retrieve_features()
        else:
            inp = self.tester.retrieve_sparse_features()
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
        if not self.ispassive:
            if len(self.scores_running_mean_vec) < self.scores_filter_shape:
                self.scores_running_mean_vec.append(score.ravel())
            else:
                self.scores_running_mean_vec = (self.scores_running_mean_vec[1:]
                                                + [score.ravel()])
            start_from = min(self.scores_filter_shape, mean_from)
            self.filtered_scores.append(
                np.mean(np.array(self.scores_running_mean_vec), axis=0))
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
            if np.max(score) >= 0.7 or len(
                    self.recognized_classes) == 0:
                self.recognized_classes.append(score.argmax())
            else:
                self.recognized_classes.append(
                    self.recognized_classes[-1])
            LOG.info('Pose detected:'
                     + self.train_classes[self.recognized_classes[-1]])
            return self.train_classes[self.recognized_classes[-1]]

    def classify_offline(self, display=True,
                         save=True):
        '''
        To be used after offline have been computed. It is a convenience
        function to allow the modification of the scores, if this is wanted,
        before performing classification.

        Process scores using stds as proposed by the paper
        '''
        if not self.ispassive:
            self.filtered_scores = self.upgr_filter(self.scores,
                                                    self.scores_filter_shape)
            #self.filtered_scores = self.scores
            fmask = np.prod(np.isfinite(self.scores), axis=1).astype(bool)
            self.filtered_scores_std = np.zeros(self.scores.shape[0])
            self.filtered_scores_std[:] = None
            self.filtered_scores_std[fmask] = np.std(self.scores[fmask, :],
                                                     axis=1)
            self.less_filtered_scores_std = self.upgr_filter(self.filtered_scores_std,
                                                             self.std_small_filter_shape)

            self.high_filtered_scores_std = self.upgr_filter(self.filtered_scores_std,
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
            count = 0
            finite_scores = self.filtered_scores[
                np.isfinite(self.filtered_scores)].reshape(
                -1, self.scores.shape[1])
            time_norm = np.max(finite_scores, axis=0) - \
                np.min(finite_scores, axis=0)
            self.time_normalized_filtered_scores = ((self.filtered_scores -
                                                     np.min(finite_scores, axis=0)) /
                                                    time_norm)
            self.time_normalized_filtered_scores[
                self.time_normalized_filtered_scores < 0] = 0
            #self.filtered_scores = self.time_normalized_filtered_scores
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
                    linewidths = [1, 1.5]
                self.plot_result(np.concatenate(plots, axis=1), labels=labels,
                                 info='Metric of actions starting and ending ' +
                                 'points', xlabel='Frames', save=save)
        else:
            # self.filtered_scores = self.upgr_filter(self.scores,
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
        return self.recognized_classes

    def put_legend_outside_plot(self, axes):
        '''
        Remove legend from the insides of the plots
        '''
        # Shrink current axis by 20%
        box = axes.get_position()
        axes.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        lgd = axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return lgd

    def compute_performance_measures(self, fmask):
        from sklearn import metrics
        y_true = np.array(self.test_ground_truth[fmask]).astype(int)
        y_pred = np.array(self.recognized_classes[fmask]).astype(int)
        f1_scores = metrics.f1_score(y_true, y_pred, average=None)
        LOG.info(self.train_classes)
        LOG.info('F1 Scores: \n' + np.array2string(f1_scores))
        confusion_mat = metrics.confusion_matrix(y_true, y_pred)
        LOG.info('Confusion Matrix: \n' + np.array2string(confusion_mat))
        accuracy = metrics.accuracy_score(y_true, y_pred)
        LOG.info('Accuracy: ' + str(accuracy))
        #labels = self.train_classes
        labels = np.array(
            self.train_classes)[np.unique(y_true).astype(int)]
        if self.save_fold is not None:
            with open(os.path.join(self.save_fold, 'f1_scores.tex'), 'w') as out:
                out.write(co.latex.array_transcribe([f1_scores, np.atleast_2d(accuracy)],
                                                    xlabels=np.concatenate((labels,
                                                                            ['Accuracy']), axis=0),
                                                    sup_x_label='F-Scores',
                                                    extra_locs=['right']))
            with open(os.path.join(self.save_fold,
                                   'Confusion_Matrix.tex'), 'w') as out:
                out.write(co.latex.array_transcribe(confusion_mat,
                                                    ylabels=labels,
                                                    xlabels=labels,
                                                    sup_x_label='Predicted',
                                                    sup_y_label='Actual'))

    def visualize_scores(self):
        '''
        Plot results with title <title>
        '''
        fmask = None
        if self.scores_exist is not None:
            self.scores_exist = np.array(self.scores_exist)
            expanded_recognized_classes = np.zeros(self.scores_exist.size)
            expanded_recognized_classes[:] = None
            for clas in self.recognized_classes:
                expanded_recognized_classes[clas.start:clas.start + clas.length + 1][
                    self.scores_exist[clas.start:clas.start + clas.length + 1]] = clas.index
            self.recognized_classes = expanded_recognized_classes
            self.crossings = np.array(self.crossings)
            if self.test_ground_truth is not None:
                fmask = self.scores_exist * np.isfinite(self.test_ground_truth)
        elif self.test_ground_truth is not None:
            if self.ispassive:
                recognized_classes_expanded = np.zeros_like(
                    self.test_ground_truth)
                recognized_classes_expanded[:] = np.nan
                recognized_classes_expanded[self.test_sync
                                            ] = self.recognized_classes[np.isfinite(
                                                self.recognized_classes)]
                self.recognized_classes = recognized_classes_expanded
                fmask = \
                    np.isfinite(self.test_ground_truth) * np.isfinite(
                        recognized_classes_expanded)
            else:
                fmask = (np.isfinite(self.recognized_classes) * np.isfinite(
                    self.test_ground_truth)).astype(bool)
        if fmask is not None:
            self.compute_performance_measures(fmask)
        plots = []
        linewidths = []
        labels = []
        markers = []
        xticks = None
        if self.test_ground_truth is not None:
            plots.append(self.test_ground_truth)
            labels.append('Ground Truth')
            linewidths.append(2)
            markers.append(',')
        if self.crossings is not None:
            xticks = self.crossings
            expanded_xticks = np.zeros_like(self.test_ground_truth)
            expanded_xticks[:] = None
            expanded_xticks[xticks] = 0
            plots.append(expanded_xticks)
            markers.append('o')
            labels.append('Actions\nbreak-\npoints')
            linewidths.append(1)
        plots.append(self.recognized_classes)
        labels.append('Identified\nClasses')
        yticks = self.train_classes
        ylim = (-1, len(self.train_classes) + 1)
        markers.append(',')
        linewidths.append(1)
        self.plot_result(np.vstack(plots).T, labels=labels,
                         xticks_locs=xticks, ylim=ylim,
                         yticks_names=yticks,
                         info='Classification Results',
                         markers=markers,
                         linewidths=linewidths,
                         xlabel='Frames', save=self.save_fold is not None)


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
    show = 0
    '''
    if POSES_CLASSIFIER.online:
        from matplotlib import pyplot as plt
        POSES_CLASSIFIER.display_scores_and_time()
        show = 1
    if ACTIONS_CLASSIFIER_SPARSE.online:
        from matplotlib import pyplot as plt
        ACTIONS_CLASSIFIER_SPARSE.display_scores_and_time()
        show = 1
    if ACTIONS_CLASSIFIER_SIMPLE.online:
        from matplotlib import pyplot as plt
        ACTIONS_CLASSIFIER_SIMPLE.display_scores_and_time()
        show = 1
    if show:
        plt.show()
    '''
    sys.exit(0)


def construct_dynamic_actions_classifier(testname='actions', train=False,
                                         test=True, visualize=True,
                                         coders_retrain=False, hog_num=None,
                                         name='actions', use_sparse=False,
                                         sparse_dim=None, test_against_all=False,
                                         visualize_feat=False, kernel=None,
                                         features=['GHOG'], post_pca=False,
                                         post_pca_components=1):
    '''
    Constructs an SVMs classifier with input 3DHOF and GHOG features
    '''
    if use_sparse:
        if sparse_dim is None:
            sparse_dim = 256
    actions_svm = Classifier('INFO', ispassive=False,
                             name=name, sparse_dim=sparse_dim,
                             use_sparse=use_sparse,
                             features=features,
                             kernel=kernel, post_pca=post_pca,
                             post_pca_components=post_pca_components)
    actions_svm.run_training(classifiers_retrain=train,
                             coders_retrain=coders_retrain,
                             visualize_feat=visualize_feat)
    if test or visualize:
        if test_against_all:
            iterat = ['actions', 'poses']
        else:
            iterat = [testname]
        for name in iterat:
            if not test:
                actions_svm.run_testing(co.CONST['test_' + name],
                                        ground_truth_type=co.CONST[
                    'test_' + name + '_ground_truth'],
                    online=False, load=True)
            else:
                actions_svm.run_testing(co.CONST['test_' + name],
                                        ground_truth_type=co.CONST[
                    'test_' + name + '_ground_truth'],
                    online=False, load=False)
            if visualize:
                actions_svm.visualize_scores()
    return actions_svm


def construct_passive_actions_classifier(testname='actions',
                                         train=True, test=True, visualize=True,
                                         pca_num=None,
                                         test_against_all=False,
                                         features=['3DXYPCA']):
    '''
    Constructs a random forests passive_actions classifier with input 3DXYPCA features
    '''
    passive_forest = Classifier('INFO', ispassive=True,
                                name='actions', use='forest',
                                use_sparse=False,
                                features=['3DXYPCA'])
    passive_forest.run_training(classifiers_retrain=train,
                                max_act_samples=2000)
    if test or visualize:
        if test_against_all:
            iterat = ['actions', 'poses']
        else:
            iterat = [testname]
        for name in iterat:
            if test:
                passive_forest.run_testing(co.CONST['test_' + name],
                                           ground_truth_type=co.CONST[
                    'test_' + name + '_ground_truth'],
                    online=False, load=False)
            else:
                passive_forest.run_testing(co.CONST['test_' + name],
                                           ground_truth_type=co.CONST[
                    'test_' + name + '_ground_truth'],
                    online=False, load=True)

            if visualize:
                passive_forest.visualize_scores()

    return passive_forest


def main():
    '''
    Example Usage
    '''
    from matplotlib import pyplot as plt
    testname = 'actions'
    # construct_passive_actions_classifier(test)
    '''
    ACTIONS_CLASSIFIER_SIMPLE.run_testing(co.CONST['test_' + testname],
                            ground_truth_type=co.CONST[
        'test_' + testname + '_ground_truth'],
        online=False, load=True)
    fake_online_testing(ACTIONS_CLASSIFIER_SIMPLE, testname)
    ACTIONS_CLASSIFIER_SIMPLE.visualize_scores()
    '''
    #plt.show()

LOG = logging.getLogger('__name__')
CH = logging.StreamHandler(sys.stderr)
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)
TRAIN_ALL_SPARSE = construct_dynamic_actions_classifier(
    train=True,features=['GHOG', 'ZHOF', '3DHOF', '3DXYPCA'],
    use_sparse=True)
POSES_CLASSIFIER = construct_passive_actions_classifier(train=False, test=False,
                                                        visualize=False,
                                                        test_against_all=False)
ACTIONS_CLASSIFIER_SIMPLE = construct_dynamic_actions_classifier(
    train=False,
    test=False,
    visualize=False,
    use_sparse=False,
    test_against_all=True)
ACTIONS_CLASSIFIER_SPARSE = construct_dynamic_actions_classifier(train=False,
                                                                 coders_retrain=False,
                                                                 test=False,
                                                                 visualize=False,
                                                                 test_against_all=True,
                                                                 use_sparse=True)
ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF = construct_dynamic_actions_classifier(
    train=False, test=False, visualize=False, test_against_all=False,
    features=['GHOG','ZHOF'])
ACTIONS_CLASSIFIER_SPARSE_WITH_ZHOF = construct_dynamic_actions_classifier(
    train=False, test=False, visualize=False, test_against_all=True,
    features=['GHOG','ZHOF'], coders_retrain=False, use_sparse=True,
    kernel='linear')
ACTIONS_CLASSIFIER_SIMPLE_WITH_ZHOF_POST_PCA = construct_dynamic_actions_classifier(
    train=True, test=True, visualize=True, test_against_all=True,
    features=['GHOG','ZHOF'],post_pca=True, post_pca_components=4)


ACTIONS_CLASSIFIER_SIMPLE_POST_PCA = construct_dynamic_actions_classifier(
    train=False,
    test=False,
    visualize=False,
    use_sparse=False,
    test_against_all=False,
    post_pca=True,
    post_pca_components=2)

ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF = construct_dynamic_actions_classifier(
    train=False, test=False, visualize=False, test_against_all=False,
    features=['GHOG','3DHOF'], kernel='linear')
ACTIONS_CLASSIFIER_SPARSE_WITH_3DHOF = construct_dynamic_actions_classifier(
    train=False, test=False, visualize=False, test_against_all=True,
    features=['GHOG','3DHOF'], coders_retrain=False, use_sparse=True,
    kernel='linear')
ACTIONS_CLASSIFIER_SIMPLE_WITH_3DHOF_POST_PCA = construct_dynamic_actions_classifier(
    train=True, test=True, visualize=True, test_against_all=True,
    features=['GHOG','3DHOF'],post_pca=True, post_pca_components=2)
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

#    visualize_feat=True)
#    visualize_feat=['Fingerwave in'])

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
