import os
import sys
import errno
import numpy as np
import classifiers as clfs
import class_objects as co
import cPickle as pickle
import logging
from numpy.linalg import inv
from CDBIMM import EnhancedDynamicClassifier

def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


class MixedClassifier(clfs.Classifier):

    def __init__(self, dynamic_classifier=None,
                 passive_classifier=None, log_lev='INFO',
                 visualize=False, add_info=None,
                 *args, **kwargs):
        self.enhanced_dyn = EnhancedDynamicClassifier(dynamic_classifier,
                                                      passive_classifier)
        clfs.Classifier.__init__(self, log_lev=log_lev, visualize=visualize,
                                 masks_needed=False,
                                 action_type='Dynamic', use='RDF', name='',
                                 features=['Passive Cl:'
                                           + str(passive_classifier.classifier_folder),
                                           'Dynamic Cl:' +
                                           str(self.enhanced_dyn.classifier_folder)],
                                 add_info=add_info, *args, **kwargs)
        self.parameters['sub_classifiers'] = [
            passive_classifier.classifier_folder,
            dynamic_classifier.clsasifier_folder]
        from sklearn.ensemble import RandomForestClassifier
        self.unified_classifier = RandomForestClassifier(10)
        self.train_classes = np.hstack((self.passive_actions,
                                        self.dynamic_actions)).tolist()

    def single_run_training(self, action_path, action):
        self.enhanced_dyn.testing_initialized = False
        passive_scores, en_scores = self.enhanced_dyn.run_testing(action_path,
                                                                  online=False,
                                                                  just_scores=True,
                                                                  display_scores=False,
                                                                  compute_perform=False,
                                                                  construct_gt=False,
                                                                  load=not self.train_all)
        traindata = np.concatenate((passive_scores, en_scores), axis=1)
        ground_truth, _ = (self.construct_ground_truth(
            data=action_path,
            ground_truth_type='constant-' + action))
        return traindata, ground_truth

    def run_training(self, save=True, load=True,
                     classifier_savename=None, train_all=False):
        self.enhanced_dyn.run_training(load=not train_all)
        self.train_all = train_all
        if classifier_savename is not None:
            self.classifier_savename = classifier_savename
        if (self.unified_classifier is not None or not load):
            traindata, ground_truth = self.apply_to_training(
                self.single_run_training)
            traindata = np.concatenate(traindata, axis=0)
            ground_truth = np.concatenate(ground_truth, axis=0)
            fmask = (np.isfinite(np.sum(traindata, axis=1)).astype(int) *
                     np.isfinite(ground_truth).astype(int)).astype(bool)
            self.unified_classifier = self.unified_classifier.fit(traindata[fmask, :],
                                                  ground_truth[fmask])
            LOG.info('Saving classifier: ' + self.classifier_savename)
            co.file_oper.save_labeled_data(['Classifier'] + self.classifier_id,
                                           [self.unified_classifier,
                                            self.training_parameters])
        else:
            (self.unified_classifier,
             self.training_parameters) = co.file_oper.load_labeled_data(
                 ['Classifier'] + self.classifier_id)
        if self.classifier_savename not in self.classifiers_list:
            with open('trained_classifiers_list.yaml', 'a') as out:
                out.write(self.classifier_savename + ': ' +
                          str(len(self.classifiers_list)) + '\n')
            self.classifiers_list[self.classifier_savename] = str(
                len(self.classifiers_list))

    def run_testing(self, data=None, online=True, against_training=False,
                    scores_filter_shape=5,
                    std_small_filter_shape=co.CONST['STD_small_filt_window'],
                    std_big_filter_shape=co.CONST['STD_big_filt_window'],
                    ground_truth_type=co.CONST['test_actions_ground_truth'],
                    img_count=None, save=True, scores_savepath=None,
                    load=False, testname=None, display_scores=True,
                    derot_angle=None, derot_center=None,
                    construct_gt=True, just_scores=False):
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

        testdata = np.hstack(
            self.enhanced_dyn.run_testing(data, online=online, just_scores=True,
                                          compute_perform=False))
        self.scores = np.zeros_like(testdata)
        self.scores[:] = None
        fmask = np.isfinite(np.sum(testdata, axis=1))
        self.scores[fmask] = self.unified_classifier.predict_proba(testdata[
                                                           fmask, :])
        #self.filtered_scores = self.scores
        self.filtered_scores = co.noise_proc.masked_filter(self.scores,
                                                           3)
        recognized_classes = []
        for score in self.filtered_scores:
            if np.sum(score) is None:
                recognized_classes.append(None)
                continue
            if (np.max(score) >= 0.6 or len(
                recognized_classes) == 0 or
                    recognized_classes[-1] is None):
                recognized_classes.append(score.argmax())
            else:
                recognized_classes.append(
                    recognized_classes[-1])
        self.recognized_classes = np.array(recognized_classes)
        if not online:
            self.test_ground_truth, _ = self.construct_ground_truth(
                data=data,
                ground_truth_type=ground_truth_type)
        self.correlate_with_ground_truth(save=save,
                                         display=display_scores)
        self.plot_result(self.filtered_scores,
                         labels=self.train_classes,
                         info='Filtered Scores',
                         xlabel='Frames',
                         save=save)






def construct_mixed_classifier(testname='actions', train=False,
                               test=True, visualize=True,
                               dicts_retrain=False, hog_num=None,
                               name='actions', use_dicts=False,
                               des_dim=None, test_against_all=False,
                               train_all=False):
    '''
    Constructs a enhanced classifier
    '''
    dynamic_classifier = clfs.construct_dynamic_actions_classifier(
        test=False, visualize=False, test_against_all=False,
        features=['GHOG', 'ZHOF'])
    passive_classifier = clfs.construct_passive_actions_classifier(test=False,
                                                                   visualize=False,
                                                                   test_against_all=False)

    mixed = MixedClassifier(
        dynamic_classifier=dynamic_classifier,
        passive_classifier=passive_classifier)
    mixed.run_training(load=not train, train_all=train_all)
    if test or visualize:
        if test_against_all:
            iterat = mixed.available_tests
        else:
            iterat = [testname]
        for name in iterat:
            mixed.testing_initialized = False
            if test:
                mixed.run_testing(os.path.join(
                    co.CONST['test_save_path'], name),
                    ground_truth_type=os.path.join(
                        co.CONST['ground_truth_fold'],
                        name + '.csv'),
                    online=False, load=False)
            else:
                mixed.run_testing(os.path.join(
                    co.CONST['test_save_path'], name),
                    ground_truth_type=os.path.join(
                        co.CONST['ground_truth_fold'],
                        name + '.csv'),
                    online=False, load=False)
    return mixed




def main():
    from matplotlib import pyplot as plt
    construct_mixed_classifier(
        train=True,
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
