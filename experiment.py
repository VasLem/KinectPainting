
from classifiers import *
from itertools import product
import logging
LOG = logging.getLogger('__name__')
def perform_experiments_on_dynamic_actions():
    descriptors = [{'features':x} for x in [
        ['GHOG','ZHOF'],
        ['GHOG','3DHOF'],['3DXYPCA','ZHOF'],
                   ['3DXYPCA','3DHOF']
    ]]
    classifiers = [{'use':x} for x in ['RDF','SVM']]
    sparsecoding = [{'use_sparse':x} for x in ['Features','Buffer',None]]
    ptpca = [{'post_pca':x} for x in [True, False]]
    extraction_method = [{'post_scores_processing_method':x} for x in
                         ['prob_check','std_check']]
    action_type = [{'action_type':x} for x in ['Dynamic']]
    params = product(descriptors,classifiers,sparsecoding,ptpca,action_type,extraction_method)
    for param in params:
        dict_param = co.dict_oper.join_list_of_dicts(list(param))
        LOG.info('Experimenting with classifier with the following params:\n'+
                 str(co.dict_oper.join_list_of_dicts(list(param))))
        try:
            clas = Classifier(hardcore=True, **dict_param)
        except:
            continue
        clas.run_training()
        for test in clas.available_tests:
            clas.run_testing(test,online=False)
def perform_experiments_on_passive_actions():
    descriptors = [{'features':x} for x in
        ['GHOG','3DXYPCA',['GHOG','3DXYPCA']]]
    sparsecoding = [{'use_sparse':x} for x in ['Features', None]]
    classifiers = [{'use':x} for x in ['RDF', 'SVM']]
    extraction_method = [{'post_scores_processing_method':x} for
                         x in ['prob_check','std_check']]
    action_type = [{'action_type':x} for x in ['Passive']]
    params = product(descriptors,classifiers, sparsecoding, action_type,extraction_method)
    for param in params:
        dict_param = co.dict_oper.join_list_of_dicts(list(param))
        LOG.info('Experimenting with classifier with the following params:\n'+
                 str(co.dict_oper.join_list_of_dicts(list(param))))
        try:
            clas = Classifier(hardcore=True, **dict_param)
        except:
            continue
        clas.run_training()
        for test in clas.available_tests:
            clas.run_testing(test,online=False)

if __name__=='__main__':
    #perform_experiments_on_dynamic_actions()
    perform_experiments_on_passive_actions()
