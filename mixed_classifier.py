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


class CombinedGesturesClassifier(clfs.Classifier):

    def __init__(self, dynamic_classifier=None,
                 passive_classifier=None, log_lev='INFO',
                 visualize=False, add_info=None,
                 *args, **kwargs):
        self.enhanced_dyn = EnhancedDynamicClassifier(dynamic_classifier,
                                                      passive_classifier,
                                                      in_sync=True)
        classifiers_used = 'Combined RDF'
        clfs.Classifier.__init__(self, log_lev=log_lev, visualize=visualize,
                                 masks_needed=False,
                                 action_type='All',
                                 classifiers_used=classifiers_used, name='',
                                 descriptors=['Passive Cl:'
                                           + str(passive_classifier.classifier_folder),
                                           'Dynamic Cl:' +
                                           str(self.enhanced_dyn.classifier_folder)],
                                 add_info=add_info, *args, **kwargs)
        self.parameters['sub_classifiers'] = [
            passive_classifier.classifier_folder,
            dynamic_classifier.classifier_folder]
        self.train_classes = np.hstack((self.passive_actions,
                                        self.dynamic_actions)).tolist()

    def single_run_training(self, action_path, action_name):
        self.enhanced_dyn.testing_initialized = False
        passive_scores, en_scores = self.enhanced_dyn.run_testing(
            os.path.join(co.CONST['actions_path'],action_name),
                                                                  online=False,
                                                                  just_scores=True,
                                                                  display_scores=False,
                                                                  compute_perform=False,
                                                                  construct_gt=False,
                                                                  load=not
                                                                  self.train_all)
        traindata = np.concatenate((passive_scores, en_scores), axis=1)
        ground_truth = co.gd_oper.construct_ground_truth(
            data=action_path,
            ground_truth_type='constant-' + action_name,
            classes_namespace=self.train_classes)[0]
        return traindata, ground_truth


    def prepare_training_data(self, *args, **kwargs):
        self.enhanced_dyn.run_training(load=not self.train_all)
        traindata, ground_truth = self.apply_to_training(
            self.single_run_training)
        traindata = np.concatenate(traindata, axis=0)
        ground_truth = np.concatenate(ground_truth, axis=0)
        fmask = (np.isfinite(np.sum(traindata, axis=1)).astype(int) *
                 np.isfinite(ground_truth).astype(int)).astype(bool)
        self.training_data = traindata[fmask, :]
        self.train_ground_truth = ground_truth[fmask]



    def offline_testdata_processing(self, data):
        LOG.info('Processing test data..')
        testdata = np.hstack(
            self.enhanced_dyn.run_testing(data, online=False, just_scores=True,
                                          compute_perform=False))
        return testdata

    def process_single_sample(self, data, img_count,
                              derot_angle=None, derot_center=None):
        return self.enhanced_dyn.run_testing(data, img_count=img_count,
                                             online=True,
                                             derot_angle=derot_angle,
                                             derot_center=derot_center)




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
        descriptors=['GHOG', 'ZHOF'])
    passive_classifier = clfs.construct_passive_actions_classifier(test=False,
                                                                   visualize=False,
                                                                   test_against_all=False)

    mixed = CombinedGesturesClassifier(
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
