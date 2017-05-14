'''
An optimized method for GridSearchCV, which iteratively performs grid search
and reduces the span of the parameters after each iteration. Made to make the
life of an engineer less boring.
'''
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

def optGridSearchCV(classifier, xtrain, ytrain, parameters, reduction_ratio=2,
                    iter_num=3, scoring='f1_macro', fold_num=5, first_rand=False,
                    n_jobs=1,verbose=1,only_rand=False, only_brute=False):
    '''
    The local optimum resides inside the parameters space, with bounds defined
    by the min and max of each parameter, thus a recommended way to run this
    function, if no prior knowledge exists, is to set the min and max of each
    parameter to the corresponding min and max allowed bounds.
    <classifier>: initialized classifier object
    <xtrain>: features of samples, with shape (n_samples, n_features)
    <ytrain>: labels of samples
    <parameters>: dictionary of parameters, same with GridSearchCV <params>
        type
    <reduction_ratio>: the scale of relative reduction of the span of the
        number parameters
    <iter_num>: number of iterations to take place
    <fold_num>: number of folds for CrossValidation
    <first_rand> : True to perform random parameter picking (normally
        distributed) firstly and then brute parameter picking (using linspace).
        If false, the turn of each method changes
    <only_rand> : True to perform only random picking
    <only_brute> : True to perform only brute picking
    '''
    def print_params(parameters, preset=''):
        '''
        print parameters in pandas form, if allowed
        '''
        try:
            from pandas import DataFrame
            if isinstance(parameters, list):
                params = DataFrame(parameters)
            else:
                try:
                    params = DataFrame.from_dict(parameters)
                except ValueError:
                    params = DataFrame([parameters])
            print(params)
        except ImportError:
            print(preset+str(parameters))
    def reduce_list(params, best_params):
        '''
        Reduce parameters list of dictionaries to a parameters dictionary,
        which correspots to the <best_params> found by <GridSearchCV>
        '''
        best_keys = set(best_params.keys())
        for count, dic in enumerate(params):
            if best_keys == set(dic.keys()):
                return dic, count
        raise Exception

    def update_parameters(prev_parameters, best_parameters, num_of_samples,
                          rate=2, israndom=True):
        '''
        Each new parameter has the same number of values as previous one and
        its values are inside the bounds set by the min and max values of the
        old parameter. Furthermore, best value from the previous paramter
        exists inside the new parameter.
        <num_of_samples>: dictionary with keys from the best_parameters.
        <prev_parameters>: previous parameters, which hold all tested values
        <best_parameters>: parameters found to provide the best score (using
            GridSearchCV)
        <israndom>: whether to perform random or brute method
        <rate>: rate of parameters span relative reduction
        '''
        rate = float(rate)
        new_parameters = {}
        for key in best_parameters:
            if (not isinstance(best_parameters[key], str) and
                    not isinstance(best_parameters[key], bool) and
                    not best_parameters[key] is None):
                if israndom:
                    center = best_parameters[key]
                    std = np.std(prev_parameters[key]) / float(rate)
                    pick = np.random.normal(loc=center, scale=std,
                                            size=100 * num_of_samples[key])
                    pick = pick[(pick >=
                                 np.min(prev_parameters[key]))*
                                (pick <= np.max(prev_parameters[key]))]
                    new_parameters[key] = pick[
                        :(num_of_samples[key]-1)]
                else:
                    center = best_parameters[key]
                    rang = np.max(prev_parameters[
                        key]) - np.min(prev_parameters[key])
                    rang = [max(center - rang /
                                float(rate), min(prev_parameters[key])),
                            min(center + rang /
                                float(rate), max(prev_parameters[key]))]
                    new_parameters[key] = np.linspace(
                        rang[0], rang[1], num_of_samples[key]-1)
                if isinstance(best_parameters[key], int):
                    new_parameters[key] = new_parameters[key].astype(int)
                new_parameters[key] = new_parameters[key].tolist()
                new_parameters[key] += [best_parameters[key]]
            else:
                new_parameters[key] = [best_parameters[key]]
        return new_parameters

    num_of_samples = {}
    if not isinstance(parameters, list):
        num_of_samples = {}
        for key in parameters:
            num_of_samples[key] = len(parameters[key])
    best_scores = []
    best_params = []
    best_estimators = []
    rand_flags = [first_rand, not first_rand]
    if only_brute:
        rand_flags = [False]
    if only_rand:
        rand_flags = [True]
    for it_count in range(iter_num):
        for rand_flag in rand_flags:
            if verbose==2:
                print('Parameters to test on:')
                print_params(parameters,'\t')
            try:
                grids = GridSearchCV(
                    classifier,
                    parameters,
                    scoring=scoring,
                    cv=fold_num,
                    n_jobs=n_jobs, verbose=verbose)
                grids.fit(xtrain, ytrain)
                best_scores.append(grids.best_score_)
                best_params.append(grids.best_params_)
                best_estimators.append(grids.best_estimator_)
                grids_params = grids.best_params_
            except ValueError:
                print('Invalid parameters')
                raise
                best_params = parameters
            if rand_flag == rand_flags[1]:
                print('Iteration Number: ' + str(it_count))
            print('\tBest Classifier Params:')
            print_params(best_params[-1],'\t\t')
            print('\tBest Score:' + str(best_scores[-1]))
            if isinstance(parameters, list):
                parameters, _ = reduce_list(parameters, grids_params)
                for key in parameters:
                    num_of_samples[key] = len(parameters[key])
            if rand_flag == rand_flags[1] and it_count == iter_num - 1:
                break
            print('Reducing Parameters using '+ ['random' if rand_flag else
                                                 'brute'][0] + ' method')
            parameters = update_parameters(parameters, grids_params, num_of_samples,
                                           rate=reduction_ratio,
                                           israndom=rand_flag)
    return best_params, best_scores, best_estimators


def example():
    '''
    An example of usage
    '''
    parameters = [{'C': [1, 10, 100, 1000], 'tol': [0.001, 0.0001],
                   'class_weight': [None, 'balanced']},
                  {'C': [1, 10, 100, 1000], 'multi_class': ['crammer_singer'],
                   'tol': [0.001, 0.0001]}]
    xtrain = np.random.random((100, 20))
    xtrain[xtrain < 0] = 0
    ytrain = (np.random.random(100) > 0.5).astype(int)
    lsvc = LinearSVC()
    optGridSearchCV(lsvc, xtrain, ytrain, parameters, reduction_ratio=2,
                    iter_num=3, scoring='f1_macro', fold_num=5, first_rand=False,
                    n_jobs=4)

if __name__ == '__main__':
    example()
