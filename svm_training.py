import sys
import numpy as np
import class_objects as co
from action_recognition_alg import *
import os.path
import cPickle as pickle
import logging
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot as plt


class SVMTraining(object):

    def __init__(self, log_lev='INFO',
                 visualize=False, masks_needed=False,
                 buffer_size=20):
        logging.basicConfig(format='%(levelname)s:%(message)s')
        logging.getLogger(__name__).setLevel(log_lev)
        self.log_lev = log_lev
        self.visualize = visualize
        self.buffer_size = buffer_size
        self.masks_needed = masks_needed
        self.action_recog = ActionRecognition(log_lev)
        self.sparse_features_lists = None
        self.unified_classifier = None
        self.dicts = None
        self.svms_1_v_all_traindata = None
        self.features_extraction = FeatureExtraction()
        self.scores = []
        self.recognized_classes = []
        self.img_count = 0
        self.actions_list = None
        self._buffer = []
        self.running_small_mean_vec = []
        self.running_big_mean_vec = []
        self.filtered_scores_std = []
        self.filtered_scores_std_mean = []
        self.filtered_scores = []
        self.saved_frames_scores = []
        self.new_action_starts_count = 0
        self.ground_truth = None

    def process_actions(self, train_act_num=0):
        logging.info('Adding actions..')
        path = co.CONST['actions_path']
        self.actions_list = [name for name in os.listdir(path)
                             if os.path.isdir(os.path.join(path, name))]
        action_recog = ActionRecognition(self.log_lev)
        for action in self.actions_list:
            action_recog.add_action(os.path.join(path, action),
                                    self.masks_needed,
                                    use_dexter=False)

        logging.info('Training dictionaries..')
        self.dicts = self.action_recog.train_sparse_dictionaries(
            act_num=train_act_num)
        logging.info('Making Sparse Features..')
        self.sparse_features_lists = (self.action_recog.
                                      actions.update_sparse_features(self.dicts,
                                                                     ret_sparse=True))

    def train_svms(self, num_of_cores=4, buffer_size = 20):
        logging.info('Preparing svm traindata')
        self.buffer_size = buffer_size
        svm_initial_training_data = []
        for sparse_features in self.sparse_features_lists:
            svm_initial_training_data.append(np.concatenate(tuple(sparse_features),
                                                            axis=0))
        svms_buffers = []
        for data in svm_initial_training_data:
            svm_buffers = []
            for count in range(data.shape[1] - self.buffer_size):
                svm_buffers.append(np.atleast_2d(
                    data[:, count:count + self.buffer_size].ravel()))
            svms_buffers.append(svm_buffers)
        logging.info('Train Data has ' + str(len(svms_buffers)) +
                     ' buffer lists. First buffer list has length ' +
                     str(len(svms_buffers[0])) +
                     ' and last buffer has shape ' +
                     str(svms_buffers[0][-1].shape))
        logging.info('Joining buffers..')
        svms_training_data = []
        for svm_buffers in svms_buffers:
            svms_training_data.append(
                np.concatenate(tuple(svm_buffers), axis=0))

        logging.info('Train Data has ' + str(len(svms_training_data)) +
                     ' training datasets for each action. Shape of first dataset is ' +
                     str(svms_training_data[0].shape))
        logging.info(
            'Creating class holding vector and concatenating remaining data..')
        self.ground_truth = []
        self.svms_1_v_all_traindata = np.concatenate(
            tuple(svms_training_data), axis=0)
        for count, data in enumerate(svms_training_data):
            self.ground_truth.append(count * np.ones((data.shape[0])))
        self.ground_truth = np.concatenate(tuple(self.ground_truth), axis=0)
        logging.info('Final training data to be used as input to OneVsRestClassifier' +
                     ' has shape ' + str(self.svms_1_v_all_traindata.shape))
        logging.info('Training SVMs using ' + str(num_of_cores) + ' cores..')
        self.unified_classifier = OneVsRestClassifier(LinearSVC(),
                                                      num_of_cores).fit(self.svms_1_v_all_traindata,
                                                                        self.ground_truth)

    def offline_testdata_processing(self, data):
        logging.info('Processing test data')
        sparse_features = self.action_recog.add_action(data, masks_needed=False,
                                                       for_testing=True)
        sparse_features = np.concatenate(tuple(sparse_features), axis=0)
        svm_buffers = []
        for count in range(sparse_features.shape[1] - self.buffer_size):
            svm_buffers.append(np.atleast_2d(sparse_features[:, count:count +
                                                             self.buffer_size].ravel()))
        svm_testing_data = np.concatenate(tuple(svm_buffers), axis=0)
        return svm_testing_data

    def test_svms(self, data, online=True, against_training=False,
                  box_filter_shape=co.CONST['STD_small_filt_window'],
                  mean_filter_shape=co.CONST['STD_big_filt_window'],
                  img_count=None):
        if not online:
            logging.info('Classifier contains ' +
                         str(len(self.unified_classifier.estimators_)) + ' estimators')
            if against_training:
                logging.info('Testing SVMS using training data..')
                scores = self.unified_classifier.decision_function(
                    self.svms_1_v_all_traindata)
            else:
                svm_testing_data = self.offline_testdata_processing(data)
                scores = self.unified_classifier.decision_function(
                    svm_testing_data)
            box_filter = np.ones(box_filter_shape) / float(box_filter_shape)
            mean_filter = np.ones(mean_filter_shape) / float(mean_filter_shape)
            filtered_scores = np.pad(np.apply_along_axis(np.convolve, 0,
                                                         scores, box_filter, mode='valid'),
                                     ((box_filter_shape / 2, box_filter_shape / 2),
                                      (0, 0)), 'edge')
            filtered_scores_std = np.std(filtered_scores, axis=1)
            filtered_scores_std_mean = np.pad(np.convolve(filtered_scores_std,
                                                          mean_filter,
                                                          mode='valid'),
                                              (mean_filter_shape / 2 - 1,
                                               mean_filter_shape / 2), 'edge')
            mean_diff = filtered_scores_std_mean - filtered_scores_std
            positive = mean_diff > 0
            zero_crossings = np.where(
                np.bitwise_xor(positive[1:], positive[:-1]))[0]
            interesting_crossings = np.concatenate((np.array([0]),
                                                    zero_crossings[::2],
                                                    np.array([mean_diff.size])), axis=0)
            identified_classes = []
            for cross1, cross2 in zip(interesting_crossings[:-1], interesting_crossings[1:]):
                self.recognized_classes.append(ClassObject())
                index = np.median(scores[cross1:cross2, :], axis=0).argmax()
                identified_classes.append(index * np.ones(cross2 - cross1))
                self.recognized_classes[-1].add(name=self.actions_list[index],
                                                index=index,
                                                start=cross1,
                                                length=cross2 - cross1)

            identified_classes = np.concatenate(
                tuple(identified_classes), axis=0)

        else:
            if img_count is not None:
                self.img_count = img_count
            self.features_extraction.update(
                data, self.img_count, use_dexter=False, masks_needed=False)

            features = self.features_extraction.extract_features()

            self.img_count += 1
            if features is not None:
                sparse_features = np.concatenate(
                    tuple([np.dot(dic, feature) for (dic, feature) in
                           zip(self.dicts, features)]), axis=0)
                if len(self._buffer) < self.buffer_size:
                    self._buffer = self._buffer + [sparse_features]
                else:
                    self._buffer = self._buffer[1:] + [sparse_features]
                if len(self._buffer) == self.buffer_size:
                    inp = np.atleast_2d(np.concatenate(tuple(self._buffer),
                                                       axis=0))
                    score = (self.unified_classifier.
                             decision_function(inp))
                    if len(self.running_small_mean_vec) < box_filter_shape:
                        self.running_small_mean_vec.append(score)
                    else:
                        self.running_small_mean_vec = (self.running_small_mean_vec[1:]
                                                       + [score])
                    self.filtered_scores.append(
                        np.mean(np.array(self.running_small_mean_vec), axis=0))
                    filtered_score_std = np.std(self.filtered_scores[-1])
                    self.filtered_scores_std.append(filtered_score_std)
                    if len(self.running_big_mean_vec) < mean_filter_shape:
                        self.running_big_mean_vec.append(
                            self.filtered_scores_std[-1])
                    else:
                        self.running_big_mean_vec = (self.running_big_mean_vec[1:]
                                                     + [self.filtered_scores_std[-1]])
                    self.filtered_scores_std_mean.append(
                        np.mean(self.running_big_mean_vec))
                    mean_diff = self.filtered_scores_std_mean[
                        -1] - filtered_scores_std
                    if mean_diff > co.CONST['action_separation_thres']:
                        self.recognized_classes.append(ClassObject())
                        index = np.median(
                            np.array(self.saved_frames_scores), axis=0).argmax()
                        self.recognized_classes[-1].add(name=self.actions_list[index],
                                                        index=index,
                                                        start=self.new_action_starts_count,
                                                        length=self.img_count
                                                        - self.new_action_starts_count)
                        self.new_action_starts_count = self.img_count
                        self.saved_frames_scores = []
                    else:
                        self.saved_frames_scores.append(score)

    def visualize_scores(self):
        plt.figure(1)
        plt.plot(np.array(self.filtered_scores_std), color='r', label='STD')
        plt.plot(np.array(self.filtered_scores_std_mean),
                 color='g', label='STD Mean')
        if self.ground_truth is not None:
            plt.plot((self.ground_truth) *
                     np.max(self.filtered_scores_std) /
                     float(np.max(self.ground_truth)),
                     label='Ground Truth')
        plt.legend()
        plt.title('Filtered Scores Statistics')
        plt.xlabel('Frames')

        plt.figure(2)
        for count, score in enumerate(np.transpose(np.array(
                self.filtered_scores))):
            plt.plot(score, label=self.actions_list[count])
        plt.legend()
        plt.title('Filtered Scores')
        plt.xlabel('Frames')
        plt.figure(3)
        mean_diff = (np.array(self.filtered_scores_std_mean) -
                     np.array(self.filtered_scores_std))
        plt.plot((mean_diff - np.min(mean_diff)) / float(np.max(mean_diff) - np.min(mean_diff)),
                 label='Filtered scores normalized mean difference')
        plt.plot(self.ground_truth / float(np.max(self.ground_truth)),
                 label='Ground Truth')
        plt.legend()
        plt.figure(4)
        plt.plot(self.ground_truth, label='Ground Truth')
        plt.plot(self.recognized_classes, label='Identified Classes')
        plt.xlabel('Frames')
        plt.ylim((np.min(self.ground_truth) - 1, np.max(self.ground_truth) + 1))
        plt.title('Result')
        plt.legend()
        plt.show()


class ClassObject(object):

    def __init__(self):
        self.name = ''
        self.index = 0
        self.start = 0
        self.length = 0

    def add(self, name='', index=0, start=0, length=0):
        self.name = name
        self.index = index
        self.start = start
        self.length = length


def main():
    '''
    Checking action learning stage and offline testing
    '''
    svm=SVMTraining()
    svm.process_actions()
    svm.train_svms()
    svm.test_svms(against_training = True)
    '''
    count = 0
    visualize = False
    log_lev = 'INFO'
    reset = False
    for _, arg in enumerate(sys.argv[1:]):
        try:
            [arg_name, arg_val] = arg.split('=')
            if arg_name == 'log_lev':
                log_lev = arg_val
            elif arg_name == 'visualize':
                visualize = True if arg_val == ('True' or 1) else False
            elif arg_name == 'reset':
                reset = arg_val
        except:
            if count == 0:
                log_lev = True if arg == 'True' else False
            elif count == 1:
                visualize = True if arg == 'True' else False
            elif count == 2:
                reset = True if arg == 'True' else False
        count = count + 1
    logging.basicConfig(format='%(levelname)s:%(message)s')
    logging.getLogger().setLevel(log_lev)

    dexter1_path = '/media/vassilis/Thesis/Datasets/dexter1/data/'
    action_names = ['adbadd', 'fingercount', 'fingerwave',
                    'flexex1', 'pinch', 'random', 'tigergrasp']
    suffix = '/tof/depth/'
    logging.info('Adding actions...')
    actions = [dexter1_path + action + suffix for action in action_names]

    action_recog = ActionRecognition(log_lev)
    path = 'scores.pkl'
    try:
        if reset:
            raise IOError
        with open(path, 'r') as inp:
            logging.info('Loading test results..')
            scores, svm_classes = pickle.load(inp)
    except (IOError, EOFError) as e:
        path = action_recog.actions.save_path
        try:
            if reset:
                raise IOError
            with open(path, 'r') as inp:
                actions = pickle.load(inp)
                sparse_features_lists = []
                for action in actions:
                    sparse_features_lists.append(action.sparse_features)
        except (EOFError, IOError) as e:
            # FARMING FEATURES STAGE
            for action in actions:
                action_recog.add_action(action, visualize=visualize)
            logging.info('Training dictionaries..')
            path = action_recog.dictionaries.save_path
            try:
                if reset:
                    raise IOError
                with open(path, 'r') as inp:
                    dicts = pickle.load(inp)
            except (EOFError, IOError) as e:
                # DICTIONARIES TRAINING STAGE
                dicts = action_recog.train_sparse_dictionaries(act_num=0)
            logging.info('Making sparse features')
            sparse_features_lists = (action_recog.actions.
                                     update_sparse_features(dicts,
                                                            ret_sparse=True))
            action_recog.actions.save()
        # sparse_features_lists is a list of lists of sparse features per action
        # To train i-th svm we get sparse_features_lists[i]
        path = 'unified_classifier.pkl'
        logging.info('Checking if trained SVMs exist..')
        try:
            if reset:
                raise IOError
            with open(path, 'r') as inp:
                logging.info('Loading existent trained SVM classifier')
                unified_classifier, svms_1_v_all_traindata, svm_classes = pickle.load(
                    inp)
        except (EOFError, IOError) as e:
            # TRAINING SVM CLASSIFIER STAGE
            logging.info('Preparing svm traindata')
            svm_initial_training_data = []
            for sparse_features in sparse_features_lists:
                svm_initial_training_data.append(np.concatenate(tuple(sparse_features),
                                                                axis=0))
            buffer_size = 20
            svms_buffers = []
            for data in svm_initial_training_data:
                svm_buffers = []
                for count in range(data.shape[1] - buffer_size):
                    svm_buffers.append(np.atleast_2d(
                        data[:, count:count + buffer_size].ravel()))
                svms_buffers.append(svm_buffers)
            logging.info('Train Data has ' + str(len(svms_buffers)) +
                         ' buffer lists. First buffer list has length ' +
                         str(len(svms_buffers[0])) +
                         ' and last buffer has shape ' +
                         str(svms_buffers[0][-1].shape))
            logging.info('Joining buffers..')
            svms_training_data = []
            for svm_buffers in svms_buffers:
                svms_training_data.append(
                    np.concatenate(tuple(svm_buffers), axis=0))

            logging.info('Train Data has ' + str(len(svms_training_data)) +
                         ' training datasets for each action. Shape of first dataset is ' +
                         str(svms_training_data[0].shape))
            training_data_shapes = [data.shape[0]
                                    for data in svms_training_data]
            logging.info(
                'Creating class holding vector and concatenating remaining data..')
            svm_classes = []
            svms_1_v_all_traindata = np.concatenate(
                tuple(svms_training_data), axis=0)
            for count, data in enumerate(svms_training_data):
                svm_classes.append(count * np.ones((data.shape[0])))
            svm_classes = np.concatenate(tuple(svm_classes), axis=0)
            logging.info('Final training data to be used as input to OneVsRestClassifier' +
                         ' has shape ' + str(svms_1_v_all_traindata.shape))
            num_of_cores = 4
            logging.info('Training SVMs using ' +
                         str(num_of_cores) + ' cores..')
            unified_classifier = OneVsRestClassifier(LinearSVC(), num_of_cores).fit(svms_1_v_all_traindata,
                                                                                    svm_classes)
            with open(path, 'wb') as outp:
                logging.info('Saving trained classifier and training data')
                pickle.dump(
                    (unified_classifier, svms_1_v_all_traindata, svm_classes), outp)

        logging.info('Classifier contains ' +
                     str(len(unified_classifier.estimators_)) + ' estimators')
        logging.info('Testing SVMS using training data..')
        # scores=unified_classifier.predict_proba(svms_1_v_all_traindata)
        scores = unified_classifier.decision_function(svms_1_v_all_traindata)
        path = 'scores.pkl'
        with open(path, 'wb') as outp:
            logging.info('Saving test results..')
            pickle.dump((scores, svm_classes), outp)

    box_filter_shape = 30
    box_filter = np.ones(box_filter_shape) / float(box_filter_shape)
    mean_filter_shape = 300
    mean_filter = np.ones(mean_filter_shape) / float(mean_filter_shape)
    filtered_scores = np.pad(np.apply_along_axis(np.convolve, 0, scores, box_filter, mode='valid'),
                             ((box_filter_shape / 2, box_filter_shape / 2), (0, 0)), 'edge')
    filtered_scores_std = np.std(filtered_scores, axis=1)
    filtered_scores_std_mean = np.pad(np.convolve(filtered_scores_std,
                                                  mean_filter, mode='valid'),
                                      (mean_filter_shape / 2 - 1, mean_filter_shape / 2), 'edge')
    mean_diff = filtered_scores_std_mean - filtered_scores_std
    positive = mean_diff > 0
    zero_crossings = np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
    interesting_crossings = np.concatenate((np.array([0]),
                                            zero_crossings[::2],
                                            np.array([mean_diff.size])), axis=0)
    identified_classes = []
    for cross1, cross2 in zip(interesting_crossings[:-1], interesting_crossings[1:]):
        clas = np.median(scores[cross1:cross2, :], axis=0).argmax()
        identified_classes.append(clas * np.ones(cross2 - cross1))
    identified_classes = np.concatenate(tuple(identified_classes), axis=0)

    plt.figure(2)
    plt.plot(filtered_scores_std, color='r', label='STD')
    plt.plot(filtered_scores_std_mean, color='g', label='STD Mean')
    plt.plot((svm_classes) * np.max(filtered_scores_std) / float(np.max(svm_classes)),
             label='Ground Truth')
    plt.legend()
    plt.title('Filtered Scores Statistics')
    plt.xlabel('Frames')

    plt.figure(3)
    for count, score in enumerate(filtered_scores.T):
        plt.plot(score,
                 label=action_names[count])
    plt.legend()
    plt.title('Filtered Scores')
    plt.xlabel('Frames')
    plt.figure(4)
    plt.plot((mean_diff - np.min(mean_diff)) / float(np.max(mean_diff) - np.min(mean_diff)),
             label='Filtered scores normalized mean difference')
    plt.plot(svm_classes / float(np.max(svm_classes)), label='Ground Truth')
    plt.legend()
    plt.figure(5)
    plt.plot(svm_classes, label='Ground Truth')
    plt.plot(identified_classes, label='Identified Classes')
    plt.xlabel('Frames')
    plt.ylim((svm_classes.min() - 1, svm_classes.max() + 1))
    plt.title('Result')
    plt.legend()
    plt.show()
    '''
