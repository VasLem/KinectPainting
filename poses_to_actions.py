import os, errno
import numpy as np
import classifiers as clfs
import class_objects as co


def makedir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
class PosesToActions(object):

    def __init__(self):
        # matrix rows are poses labels
        # matrix columns are actions labels
        # matrix entries are coherence probabilities
        self.coherence_matrix = None
        self.poses_classifier = clfs.POSES_CLASSIFIER  # poses classifier
        self.actions_classifier = clfs.ACTIONS_CLASSIFIER  # actions classifier
        # actions ground truth
        self.poses_predictions = None
        self.poses = self.poses_classifier.train_classes
        self.actions_ground_truth = None
        self.actions = None

    def extract_variables(self, set_pathname, csv_pathname,
                          keep_only_valid_actions=True):
        '''
        This function loads from a <csv_pathname> the actions ground truth and
        tests <self.poses_classifier> using images inside <set_pathname>,
        extracting preficted poses.
        <set_pathname> is the pathname of the folder of png images refering
        to the <csv_pathname>, which holds the corresponding actions
        ground truth. So both <set_pathname> and <csv_pathname> should be
        correlated.
        The actions ground truth contains only actions that exist already
        inside <self.actions_classifier>. If this is not desired, set
        <keep_only_valid_actions> to False.
        '''
        self.actions_ground_truth = self.actions_classifier.construct_ground_truth(
            set_pathname, csv_pathname, keep_only_valid_actions)
        self.actions = (
            self.actions_classifier.train_classes)
        self.poses_predictions = self.poses_classifier.run_testing(
            set_pathname, online=False, load=False, save=False,
            display_scores=False)
        poses_predictions_expanded = np.zeros(
            np.max(self.poses_classifier.test_sync) + 1)
        poses_predictions_expanded[:] = None
        poses_predictions_expanded[self.poses_classifier.test_sync
                           ] = self.poses_predictions
        self.poses_predictions = poses_predictions_expanded
        fmask = np.isfinite(self.actions_ground_truth
                           )*np.isfinite(
                               self.poses_predictions)
        self.poses_predictions = self.poses_predictions[
            fmask].astype(int)
        self.actions_ground_truth = self.actions_ground_truth[
            fmask].astype(int)

    def construct_coherence(self):
        '''
        Coherence matrix is a matrix which has as many rows as the poses
        labels and as many columns as the actions labels. Each row sums to 1
        and describes the coherence between the corresponding pose and the
        action labels.
        This function constructs actions to pose <coherence_matrix>.
        <extract_variables> should be run first
        '''
        if self.actions is None:
            raise Exception(self.construct_coherence.__doc__)
        self.coherence_matrix = np.zeros((len(self.poses),
                                          len(self.actions)))
        for action_truth, pose_pred in zip(self.actions_ground_truth,
                                           self.poses_predictions):
            self.coherence_matrix[pose_pred,
                                  action_truth] += 1
        self.coherence_matrix = self.coherence_matrix / np.sum(
            self.coherence_matrix, axis=0)
        return self.coherence_matrix

    def run(self, set_pathname, csv_pathname, keep_only_valid_actions=True):
        '''
        Convenience function
        '''
        self.extract_variables(set_pathname, csv_pathname,
                               keep_only_valid_actions)
        self.construct_coherence()
        return self.coherence_matrix, self.poses, self.actions

    def plot_coherence(self, save=True):
        '''
        plot nicely the coherence matrix
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
        for c1 in xrange(len(self.poses)):
            for c2 in xrange(len(self.actions)):
                ax.annotate(str('$%.2f$' % self.coherence_matrix[c1,c2]), xy=(c2,
                                                                            c1),
                            horizontalalignment='center',
                            verticalalignment='center',fontsize=16)

        cb = fig.colorbar(res)
        ax.set_xlabel(r'Actions', fontsize=16)
        ax.set_ylabel(r'Poses', fontsize=16)
        ax.set_title(r'Coherence', fontsize=18)
        plt.xticks(range(len(self.actions)), [r'%s' % action for action in
                                              self.actions], rotation = 45)
        plt.yticks(range(len(self.poses)), [r'%s' % pose for pose in
                                            self.poses])
        save_fold = os.path.join(
            co.CONST['results_fold'], 'Classification', 'Poses RF',)
        makedir(save_fold)
        if save:
            plt.savefig(os.path.join(
                save_fold, 'Coherence Matrix.pdf'))
        plt.show()


def main():
    poses_to_actions = PosesToActions()
    testname = 'actions'
    coherence = poses_to_actions.run(co.CONST['test_'+ testname ],
                         co.CONST['test_'+ testname + '_ground_truth'])
    poses_to_actions.plot_coherence()


if __name__ == '__main__':
    main()
