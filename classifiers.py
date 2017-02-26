'''
Implementation of Classifier Training, partly described inside Fanello et al.
'''
import sys,signal
import errno
import glob
import numpy as np
import class_objects as co
import action_recognition_alg as ara
import cv2
import os.path
import cPickle as pickle
import logging
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

    '''

    def __init__(self, log_lev='INFO',
                 visualize=False, masks_needed=False,
                 buffer_size=co.CONST['buffer_size'], des_dim=None,
                 ispassive=False,
                 use='svms', num_of_cores=4, name='',
                 add_info=None,
                 use_dicts=True,
                 feature_params=None):

        # General configuration
        self.add_info=add_info
        self.log_lev = log_lev
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.masks_needed = masks_needed
        self.ispassive = ispassive
        self.use = use
        self.num_of_cores = num_of_cores
        self.name = name
        self.feature_params = feature_params
        if ispassive:
            info='passive '
        else:
            info='dynamic '
        info = info + self.name +' '+self.use
        if self.add_info:
            info += ' ' + self.add_info
        self.full_name = info.title()

        if self.use == 'svms':
            from sklearn.svm import LinearSVC
            from sklearn.multiclass import OneVsRestClassifier
            self.classifier_type = OneVsRestClassifier(LinearSVC(),
                                                       self.num_of_cores)
        elif not self.use == 'mixed':
            from sklearn.ensemble import RandomForestClassifier
            self.classifier_type = RandomForestClassifier(co.CONST['RF_trees'])

        # Core variables
        self.features_extraction = ara.FeatureExtraction()
        self.action_recog = ara.ActionRecognition(log_lev, des_dim=des_dim)
        self.unified_classifier = None
        self.use_dicts = use_dicts
        self.sparse_coders = None  # is loaded from memory


        # Sparse coding variables
        self.dicts = None

        # Training variables
        self.one_v_all_traindata = None
        self.train_ground_truth = None  # is loaded from memory after training
        self.train_classes = None  # is loaded from memory after training
        self.allowed_train_actions = None
        if ispassive:
            fil = os.path.join(co.CONST['rosbag_location'],
                               'gestures_type.csv')
            self.allowed_train_actions = []
            if os.path.exists(fil):
                with open(fil, 'r') as inp:
                    for line in inp:
                        if line.split(':')[0] == 'Static':
                            self.allowed_train_actions = line.split(
                                ':')[1].rstrip('\n').split(',')
            else:
                self.allowed_train_actions = None

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
        self.testing_initialized = False

    def initialize_classifier(self, classifier=None):
        if classifier is not None:
            self.unified_classifier = classifier
        else:
            if self.use == 'svms':
                from sklearn.svm import LinearSVC
                from sklearn.multiclass import OneVsRestClassifier
                self.unified_classifier = OneVsRestClassifier(
                    LinearSVC(),
                    self.num_of_cores)
            else:
                from sklearn.ensemble import RandomForestClassifier
                self.unified_classifier = RandomForestClassifier(10)

        if self.use == 'svms':
            self.decide = self.unified_classifier.decision_function
            self.predict = self.unified_classifier.predict
        else:
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

    def run_training(self, dict_train_act_num=None, dicts_retrain=False,
                     classifiers_retrain=False, test_against_training=False,
                     training_datapath=None, classifiers_savepath=None,
                     num_of_cores=4, buffer_size=None, classifiers_save=True,
                     max_act_samples=None,
                     min_dict_iterations=3, feature_params=None):
        '''
        <Arguments>
        For testing:
            If a testing using training data is to be used, then
            <test_against_training> is to be True
        For dictionaries training:
            Train dictionaries using <dict_train_act_num>-th action. Do not
            train dictionaries if save file already exists or <dicts_retrain>
            is False. If <dict_train_act_num> is <None> , then train
            dictionaries using all available actions
        For svm training:
            Train ClassifierS with <num_of_cores> and buffer size <buffer_size>.
            Save them if <classifiers_save> is True to <classifiers_savepath>. Do not train
            if <classifiers_savepath> already exists and <classifiers_retrain> is False.
        '''
        LOG.info(self.full_name+':')
        if feature_params is not None:
            self.feature_params = feature_params
        if classifiers_savepath is None:
            classifiers_savepath = 'trained_'
            classifiers_savepath += self.full_name.replace(' ','_').lower()
            classifiers_savepath += '.pkl'

        if not os.path.isfile(classifiers_savepath):
            LOG.info('Missing trained classifier save file:'+classifiers_savepath)
            LOG.info(self.full_name + ' will be retrained')
            classifiers_retrain = True
        if buffer_size is not None:
            self.buffer_size = buffer_size
        if classifiers_retrain or dicts_retrain or test_against_training:
            if dicts_retrain:
                max_act_samples = None
            self.prepare_training_data(training_datapath, max_act_samples)
        if self.use_dicts:
            self.process_dictionaries(dict_train_act_num, dicts_retrain,
                                      min_iterations=min_dict_iterations,
                                      max_act_samples=max_act_samples)

        if self.use_dicts and (classifiers_retrain or test_against_training):
            LOG.info('Making Sparse Features..')
            self.action_recog.actions.update_sparse_features(
                                              self.sparse_coders,
                                              max_act_samples=max_act_samples,
                                              fss_max_iter=self.fss_max_iter)
        self.process_training(num_of_cores, classifiers_retrain,
                              classifiers_savepath, classifiers_save,
                              test_against_training)

    def prepare_training_data(self, path=None, max_act_samples=None,
                              fss_max_iter=100):
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
        if self.allowed_train_actions is not None:
            self.train_classes = [clas for clas in self.train_classes if clas
                                  in self.allowed_train_actions]
        for action in self.train_classes:
            self.action_recog.add_action(os.path.join(path, action),
                                         self.masks_needed,
                                         use_dexter=False,
                                         ispassive=self.ispassive,
                                         max_act_samples=max_act_samples,
                                         feature_params=self.feature_params,
                                         fss_max_iter=fss_max_iter)

    def process_dictionaries(self, train_act_num=None, retrain=False,
                             dictionaries_savepath=None, min_iterations=3,
                             max_act_samples=None):
        '''
        Train dictionaries using <train_act_num>-th action or load them if
            <retrain> is False and save file exists. <max_act_samples> is the
            number of samples to be sparse coded after the completion of the
            training/loading phase and defines the training data size
            of each action.
        '''
        if dictionaries_savepath is None:
            dictionaries_savepath = 'trained_coders_for'
            if self.ispassive:
                dictionaries_savepath += '_passive_'
            else:
                dictionaries_savepath += '_dynamic_'
            dictionaries_savepath += self.name+'_' + self.use
            dictionaries_savepath += '.pkl'
        read_from_mem = 0
        self.action_recog.dictionaries.save_path = dictionaries_savepath
        if not retrain:
            try:
                with open(self.action_recog.dictionaries.save_path, 'r') as inp:
                    LOG.info('Loading dictionaries from: '+
                             self.action_recog.dictionaries.save_path)
                    self.sparse_coders, self.dicts = pickle.load(inp)
                    self.action_recog.dictionaries.dicts = self.dicts
                    self.action_recog.dictionaries.sparse_dicts = self.sparse_coders
                    read_from_mem = 1
            except (IOError, EOFError):
                read_from_mem = 0
        else:
            LOG.info('retrain switch is True, so the dictionaries ' +
                     'are retrained')
        if not read_from_mem:
            LOG.info('Training dictionaries..')
            self.dicts = self.action_recog.train_sparse_dictionaries(
                act_num=train_act_num, min_iterations=min_iterations)
            self.sparse_coders = self.action_recog.dictionaries.sparse_dicts

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
        if against_training or retrain or not os.path.exists(savepath):
            if against_training and (not retrain or os.path.exists(savepath)):
                LOG.info('Preparing Classifiers Train Data for testing..')
            else:
                LOG.info('Preparing Classifiers Train Data..')
            if not self.ispassive:
                initial_traindata = []
                if self.use_dicts:
                    for action in self.action_recog.actions.actions:
                        initial_traindata.append(np.concatenate(tuple(action.
                                                                      sparse_features),
                                                                axis=0))
                else:
                    for action in self.action_recog.actions.actions:
                        initial_traindata.append(np.concatenate(tuple(action.features),
                                                                axis=0))

                traindata_samples_inds = [
                    action.samples_indices
                    for action in self.action_recog.actions.actions]
                traindata_frames_inds = [
                    action.sync for action in self.action_recog.actions.actions]
                acts_buffers = []
                for data, frames_samples_inds, frames_inds in zip(
                        initial_traindata, traindata_samples_inds,
                        traindata_frames_inds):
                    act_buffers = []
                    for count in range(data.shape[1] - self.buffer_size):
                        # Checking if corresponding frames belong to same sample
                        # and are not too timespace distant from each other
                        # If every condition holds then the buffer is generated
                        if (len(np.unique(
                            frames_samples_inds[count:count +
                                                self.buffer_size])) == 1
                            and
                            np.all(np.abs(np.diff(frames_inds[count:count +
                                                              self.buffer_size]))
                                   <=
                                   self.buffer_size / 4)):
                            act_buffers.append(np.atleast_2d(
                                data[:, count:count + self.buffer_size].ravel()))
                    acts_buffers.append(act_buffers)
                LOG.info('Train Data has ' + str(len(acts_buffers)) +
                         ' buffer lists. First buffer list has length ' +
                         str(len(acts_buffers[0])) +
                         ' and last buffer has shape ' +
                         str(acts_buffers[0][-1].shape))
                LOG.info('Joining buffers..')
                multiclass_traindata = []
                for act_buffers in acts_buffers:
                    multiclass_traindata.append(
                        np.concatenate(tuple(act_buffers), axis=0))
            else:
                if self.use_dicts:
                    multiclass_traindata = [np.concatenate(
                        tuple(action.sparse_features), axis=0).T for
                                            action in
                                            self.action_recog.actions.actions]
                else:
                    multiclass_traindata = [np.concatenate(
                        tuple(action.features), axis=0).T for
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
        if not retrain and os.path.exists(savepath):
            LOG.info('Loading trained Classifiers from: ' +
                     savepath)
            with open(savepath, 'r') as inp:
                (self.unified_classifier, self.train_ground_truth,
                 self.train_classes) = pickle.load(inp)
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
            LOG.info('Saving trained Classifiers to: ' + savepath)
            with open(savepath, 'w') as out:
                pickle.dump((self.unified_classifier, self.train_ground_truth,
                             self.train_classes), out)

    def offline_testdata_processing(self, datapath):
        '''
        Offline testing data processing, using data in <datapath>.
        '''
        LOG.info('Processing test data..')
        LOG.info('Extracting features..')
        features, self.test_sync = self.action_recog.add_action(
            datapath, masks_needed=False,
            for_testing=True,
            ispassive=self.ispassive,
            feature_params=self.feature_params,
            fss_max_iter=100)
        features = np.concatenate(tuple(features), axis=0)
        if not self.ispassive:
            act_buffers = []
            frames_inds = self.action_recog.actions.testing.sync
            test_buffers_start_inds = []
            test_buffers_end_inds = []
            for count in range(features.shape[1] - self.buffer_size):
                if np.all(np.abs(np.diff(frames_inds[count:count +
                                                     self.buffer_size])) <=
                          self.buffer_size / 4):

                    act_buffers.append(np.atleast_2d(features[:, count:count +
                                                              self.buffer_size].ravel()))
                    test_buffers_start_inds.append(frames_inds[count])
                    test_buffers_end_inds.append(frames_inds[count +
                                                             self.buffer_size])
            testdata = np.concatenate(tuple(act_buffers), axis=0)
            return testdata, test_buffers_start_inds, test_buffers_end_inds
        else:
            testdata = features.T
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
            if os.path.isdir(os.path.join(data, '0')):
                paths = [filename for filename in
                         glob.glob(os.path.join(data, '0', '*.png'))]
            else:
                paths = [filename for filename in
                         glob.glob(os.path.join(data, '*.png'))]
            files = [os.path.basename(path) for path in paths]
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
                ground_truth = np.zeros(max([int(filter(str.isdigit, filename)) for
                                             filename in files]) + 1)
            ground_truth[:] = np.NaN
            all_bounds = [map(list, zip(*ground_truth_init[key])) for key in
                          ground_truth_init.keys()]
            if isinstance(data, basestring):
                iterat = [int(filter(str.isdigit, filename))
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
                                  ] = int(filter(str.isdigit, filename))
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
                ground_truth = np.zeros(max([int(filter(str.isdigit, filename)) for
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
            ground_truth = np.zeros(max([int(filter(str.isdigit, filename)) for
                                         filename in files]) + 1)
            ground_truth[:] = np.NaN
            for fil in files:
                ground_truth[int(filter(str.isdigit, fil))] = ground_val

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


    def init_testing(self,data=None, online=True, save=True,load=True,
                     testname=None, scores_savepath=None,
                     scores_filter_shape = 5,
                     std_small_filter_shape=co.CONST['STD_small_filt_window'],
                     std_big_filter_shape=co.CONST['STD_big_filt_window'],
                     testdatapath=None, *kwargs):
        '''
        Initializes paths and names used in testing to save, load and visualize
        data.
        Built as a convenience method, in case <self.run_testing> gets overriden.
        '''
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
            fold_name = (self.name + ' ' + self.use).title()
            self.save_fold = os.path.join(
                co.CONST['results_fold'], 'Classification', fold_name)
            if self.add_info is not None:
                self.save_fold = os.path.join(
                 self.save_fold, self.add_info.replace(' ','_').lower())
            self.save_fold = os.path.join(self.save_fold,self.testdataname)
            makedir(self.save_fold)

            if scores_savepath is None:
                self.scores_savepath = self.testdataname + '_scores_for_'
                self.scores_savepath += self.full_name.replace(' ',
                                                               '_').lower()
                self.scores_savepath += '.pkl'

    def run_testing(self, data=None, derot_angle=None, derot_center=None,
                    online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=co.CONST['test_actions_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=False, testname=None, display_scores=True,
                    construct_gt=True, just_scores=False,testdatapath=None):
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
        if isinstance(data, tuple):
            derot_angle = data[1]
            derot_center = data[2]
            data = data[0]
        if not self.testing_initialized:
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
            if load and os.path.exists(self.scores_savepath):
                LOG.info('Loading saved scores from: ' + self.scores_savepath)
                with open(self.scores_savepath, 'r') as inp:
                    self.scores, self.test_sync, self.testname = pickle.load(
                        inp)
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
                    LOG.info(self.full_name + ':')
                    LOG.info('Testing Classifiers..')
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
                        with open(self.scores_savepath, 'w') as out:
                            LOG.info('Saving scores to: '+self.scores_savepath)
                            pickle.dump((self.scores, self.test_sync,
                                         self.testname), out)
            if construct_gt:
                LOG.info('Constructing ground truth vector..')
                self.test_ground_truth = self.construct_ground_truth(
                    data, ground_truth_type)
            if not just_scores:
                self.classify_offline(save=save)
                if display_scores:
                    self.display_scores_and_time(save=save)
            return True , self.scores
        else:
            '''
            input is processed from hand_segmentation_alg (any data
            processed in such way, that the result is the same with my processing,
            is acceptable, eg. Dexter)
            There must be a continuous data streaming (method called in every
            loop), even if the result of the previous algorithm is None
            '''
            scores_exist,score = self.process_online_data(data, img_count,
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
        LOG.info(self.name.title()+':')
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
            _t_ = np.array(
                self.action_recog.actions.sparse_time) * 1000
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
                               +
                               self.action_recog.
                               actions.
                               features_extract.feat_names[t]
                               for t in range(sparse_t.shape[0])]
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
            self.reset_online_test()
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
        if not self.ispassive:
            self.features_extraction.update(
                co.pol_oper.derotate(self.frame_prev,
                                     derot_angle,
                                     derot_center),
                self.count_prev,
                use_dexter=False,
                masks_needed=False)
            self.features_extraction.update(
                co.pol_oper.derotate(data,
                                     derot_angle,
                                     derot_center),
                self.img_count,
                use_dexter=False,
                masks_needed=False)
        else:
            self.features_extraction.update(
                co.pol_oper.derotate(data,
                                     derot_angle,
                                     derot_center),
                self.img_count,
                use_dexter=False,
                masks_needed=False)
        self.frame_prev = data.copy()
        self.count_prev = self.img_count
        features = self.features_extraction.extract_features(
            ispassive=self.ispassive)
        if not self.ispassive:
            if features is not None:
                self.frame_exists.append(True)
                if self.use_dicts:
                    features = np.atleast_2d(
                        np.concatenate(
                        tuple([coder.code(feature) for (coder, feature) in
                               zip(self.sparse_coders, features)]),
                            axis=0).ravel())
                else:
                    features = np.atleast_2d(np.concatenate(
                        tuple(features), axis=0).ravel())
                if len(self._buffer) < self.buffer_size:
                    self._buffer = self._buffer + [features]
                    self.buffer_exists.append(False)
                    self.scores_exist.append(False)
                    return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                                        np.float64)
                else:
                    self._buffer = self._buffer[1:] + [features]
            else:
                self.frame_exists.append(False)
            #require that buffer is compiled by frames that are approximately
            #   continuous
            if (sum(self.frame_exists[-(self.buffer_size
                                        + co.CONST['buffer_max_misses']):])
                >= self.buffer_size):
                self.buffer_exists.append(True)
            else:
                self.buffer_exists.append(False)
                self.scores_exist.append(False)
                return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                                        np.float64)
            #require that buffers' window is contiguous in order to compute mean STD
            existence = self.buffer_exists[
                - min(self.mean_from,
                      co.CONST['STD_big_filt_window']
                      + self.buffer_size):]
            if sum(existence) < 3 * len(existence) / 4:
                self.scores_exist.append(False)
                return False, np.array([[None] *
                                    len(self.train_classes)]).astype(
                                        np.float64)
            elif sum(existence) == 3 * len(existence) / 4:
                self.mean_from = 0
            inp = np.concatenate(tuple(self._buffer),axis=0).T.reshape(1,-1)
            # scores can be computed if and only if current buffer exists and
            #   buffers' window for mean STD exists and is approximately
            #   continuous
            self.scores_exist.append(True)
        else:
            inp = features[0].reshape(1, -1)
        score = (self.decide(inp))
        self.scores.append(score)
        if not just_scores:
            self.classify_online(score, self.img_count,
                                        self.mean_from)
        else:
            self.filtered_scores.append(score)
        return True, np.array(score).reshape(1,-1)

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
                if len(self.recognized_classes)>=2:
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
                                np.isfinite(self.test_ground_truth)])))[:,None]]
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
        dif = set(np.unique(self.test_ground_truth)).symmetric_difference(
            set(np.unique(self.recognized_classes)))
        for cnt in range(len(fmask)):
            if self.recognized_classes[cnt] in dif:
                fmask[cnt] = False
        y_true = np.array(self.test_ground_truth[fmask]).astype(int)
        y_pred = np.array(self.recognized_classes[fmask]).astype(int)
        f1_scores = metrics.f1_score(y_true, y_pred, average=None)
        LOG.info('F1 Scores:' + np.array2string(f1_scores))
        confusion_mat = metrics.confusion_matrix(y_true, y_pred)
        LOG.info('Confusion Matrix: ' + np.array2string(confusion_mat))
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
                fmask = \
                    np.isfinite(self.test_ground_truth) * np.isfinite(
                        recognized_classes_expanded)
                self.recognized_classes = recognized_classes_expanded
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
        #DEBUGGING
        #cv2.imshow('test', (img % 255).astype(np.uint8))
        #cv2.waitKey(10)
        classifier.run_testing(
            data=img,
            img_count=sync[count],
            online=True,
            load=False,
            derot_angle=angles[count],
            derot_center=centers[count],
            testdatapath = co.CONST['test_' + testname])
    classifier.recognized_classes[-1].add(length=max(sync))
    if len(classifier.scores_exist)>1:
        classifier.test_ground_truth = classifier.construct_ground_truth(
            classifier.scores_exist, ground_truth_type=co.CONST['test_'
                                                                 + testname
                                                                 + '_ground_truth'])
        classifier.filtered_scores = np.array(classifier.filtered_scores).squeeze()
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
    sys.exit(0)

def construct_dynamic_actions_classifier(testname='train', train=False,
                                 test=True, visualize=True,
                                 dicts_retrain=False, hog_num=None,
                                 name='actions', use_dicts=True,
                                 des_dim=None, test_against_all=False):
    '''
    Constructs an SVMs classifier with input 3DHOF and GHOG features
    '''
    if not use_dicts:
        add_info = 'without sparse coding'
    else:
        if des_dim is None:
            des_dim = [256, 128]
        add_info = 'with sparse coding'
    actions_svm = Classifier('INFO', ispassive=False,
                             name=name, des_dim=des_dim,
                             use_dicts=use_dicts,
                             feature_params=hog_num,
                             add_info=add_info)
    actions_svm.run_training(classifiers_retrain=train,
                             dicts_retrain=dicts_retrain,
                             max_act_samples=1000)
    if test or visualize:
        if not test:
            actions_svm.run_testing(co.CONST['test_' + testname],
                                    ground_truth_type=co.CONST[
                'test_' + testname + '_ground_truth'],
                online=False, load=True)
        else:
            if test_against_all:
                iterat = ['actions','poses']
            else:
                iterat = [testname]
            for name in iterat:
                actions_svm.reset_offline_test()
                actions_svm.run_testing(co.CONST['test_' + name],
                                        ground_truth_type=co.CONST[
                    'test_' + name + '_ground_truth'],
                                        online=False, load=False)
                if visualize:
                    actions_svm.visualize_scores()
    return actions_svm


def construct_passive_actions_classifier(testname='train',
                               train=True, test=True, visualize=True,
                               pca_num=None,
                               test_against_all=False):
    '''
    Constructs a random forests passive_actions classifier with input 3DXYPCA features
    '''
    passive_forest = Classifier('INFO', ispassive=True,
                               name='actions', use='forest',
                               feature_params=pca_num, use_dicts=False)
    passive_forest.run_training(classifiers_retrain=train,
                               max_act_samples=2000)
    if test or visualize:
        if not test:
            passive_forest.run_testing(co.CONST['test_' + testname],
                                      ground_truth_type=co.CONST[
                                          'test_' + testname + '_ground_truth'],
                                      online=False, load=True)
        else:
            if test_against_all:
                iterat = ['actions','poses']
            else:
                iterat = [testname]
            for name in iterat:
                passive_forest.reset_offline_test()
                passive_forest.run_testing(co.CONST['test_' + name],
                                      ground_truth_type=co.CONST[
                                          'test_' + name + '_ground_truth'],
                                      online=False, load=False)
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

    POSES_CLASSIFIER.run_testing(co.CONST['test_' + testname],
                            ground_truth_type=co.CONST[
        'test_' + testname + '_ground_truth'],
        online=False, load=False)
    '''
    fake_online_testing(ACTIONS_CLASSIFIER_SIMPLE, testname)
    '''
    ACTIONS_CLASSIFIER_SPARSE.visualize_scores()
    plt.show()

LOG = logging.getLogger('__name__')
CH = logging.StreamHandler(sys.stderr)
CH.setFormatter(logging.Formatter(
    '%(funcName)20s()(%(lineno)s)-%(levelname)s:%(message)s'))
LOG.handlers = []
LOG.addHandler(CH)
LOG.setLevel(logging.INFO)
POSES_CLASSIFIER = construct_passive_actions_classifier(train=False, test=False,
                                              visualize=False,
                                             test_against_all=False)
ACTIONS_CLASSIFIER_SPARSE = construct_dynamic_actions_classifier(train=False,
                                                         dicts_retrain=False,
                                                         test=False,
                                                         visualize=False,
                                                         test_against_all=False)
ACTIONS_CLASSIFIER_SIMPLE = construct_dynamic_actions_classifier(train=False,
                                                         test=False,
                                                         visualize=False,
                                                         use_dicts=False,
                                                         test_against_all=False)

if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    main()
