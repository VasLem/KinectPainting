#encoding=utf-8
from classifiers import *
from CDBIMM import *
from mixed_classifier import CombinedGesturesClassifier
from itertools import product
import logging
import ast
import pandas as pd
LOG = logging.getLogger('__name__')


def perform_single_experiment(dict_param, filt=None, metric='Accuracy',
                              ret_all=False, variation=None):
    try:
        if variation is None:
            clas = Classifier(hardcore=True, **dict_param)
        elif variation.lower() == 'enhanced':
            clas = EnhancedDynamicClassifier(**dict_param)
        elif variation.lower() == 'combined':
            clas = CombinedGesturesClassifier(**dict_param)
    except Exception as e:
        print e
        return None, None, None, None
    clas.run_training()
    metrics = []
    names = []
    save_folds = []
    for test in clas.available_tests:
        if filt is None or filt.lower() in test.lower():
            names.append(test)
            clas.run_testing(test,online=False)
            save_folds.append(clas.save_fold)
            if ret_all:
                metrics.append(clas.testdata[clas.test_ind])
            else:
                metrics.append(clas.testdata[clas.test_ind][
                    metric])
    if len(save_folds)> 1:
        return metrics, clas, save_folds, names
    else:
        return metrics, clas, None, names


def perform_experiments_generic(params, filt, metric_typ='Macro'):
    '''
    <metric_typ> can be Micro, Macro or Mean
    '''
    total_metrics = []
    found_params = []
    for param in params:
        dict_param = co.dict_oper.join_list_of_dicts(list(param))
        LOG.info('Experimenting with classifier with the following params:\n'+
                 str(co.dict_oper.join_list_of_dicts(list(param))))
        metrics, classifier, _, _ = perform_single_experiment(dict_param, filt)
        if metrics is None:
            continue
        if metric_typ == 'Mean':
            metrics = [np.mean([[metric['Micro'], metric['Macro']] for metric in
                             metrics])]
        elif metric_typ == 'All':
            metrics = [[metric['Micro'],metric['Macro']] for metric in metrics]
        else:
            metrics = [np.mean([metric[metric_typ] for metric in metrics])]
        total_metrics.append(metrics)
        found_params.append(dict_param)

    return found_params, total_metrics
    



def perform_experiments_on_dynamic_actions(metric_typ='Macro'):
    descriptors = [{'descriptors':x} for x in [
        ['GHOG','ZHOF'],
        ['GHOG','3DHOF'],['3DXYPCA','ZHOF'],
                   ['3DXYPCA','3DHOF'],
        ['GHOG', '3DXYPCA', 'ZHOF'],
        ['GHOG', '3DXYPCA', '3DHOF']
    ]]
    classifiers = [{'classifiers_used':x} for x in ['rdf','svm','platt svm']]
    sparsecoding = [{'sparsecoding_level':x} for x in [#'Features',
                                                       None]]
    ptpca = [{'ptpca':x} for x in [#True,
                                   False]]
    extraction_method = [{'post_scores_processing_method':x} for x in
                         ['CProb','CSTD']]
    action_type = [{'action_type':x} for x in ['Dynamic']]
    params = product(descriptors,classifiers,sparsecoding,ptpca,action_type,extraction_method)
    return perform_experiments_generic(params, 'validation',
                                       metric_typ=metric_typ)

def perform_experiments_on_passive_actions(metric_typ='Macro'):
    descriptors = [{'descriptors':x} for x in
        ['GHOG','3DXYPCA',['GHOG','3DXYPCA']]]
    sparsecoding = [{'sparsecoding_level':x} for x in ['Features', None]]
    classifiers = [{'classifiers_used':x} for x in ['rdf', 'svm', 'adaboost']]
    extraction_method = [{'post_scores_processing_method':x} for
                         x in ['CProb','CSTD']]
    action_type = [{'action_type':x} for x in ['Passive']]
    params = product(descriptors,classifiers, sparsecoding, action_type,extraction_method)
    return perform_experiments_generic(params, 'validation',
                                       metric_typ=metric_typ)

def retrieve_top_n_experiments(action_type, filt, n=3, bypass=0, ignore=[]):
    dic = {'classifiers_used': 'rdf', 'descriptors': ['GHOG', 'ZHOF'], 'post_scores_processing_method': 'prob_check', 'ptpca': True, 'action_type': 'Dynamic', 'sparsecoding_level': 'Features'}
    classifier = Classifier(**dic)
    tests_instances = {}
    tests_envs = {}
    all_catalog = classifier.load_tests_mapping()
    import ast
    for count,name in enumerate(classifier.available_tests):
        if filt not in name.lower():
            continue
        tests_instances = classifier.load_all_test_instances(count)
        for key in tests_instances:
            desc = classifier.return_description(all_catalog, key)
            if action_type not in str(desc):
                continue
            for ignored_key in ignore:
                if ignored_key.lower() in str(desc).lower():
                    continue
            test_env = str(ast.literal_eval(str(desc))[1:]) 
            if  test_env not in tests_envs:
                tests_envs[test_env] = {}
            while True:
                try:
                    tests_envs[test_env]['Index'].append(key)
                    tests_envs[test_env][
                        'Instance'].append(tests_instances[key])
                    break
                except:
                    tests_envs[test_env]['Index'] = []
                    tests_envs[test_env]['Instance'] = []
    for test_env in tests_envs:
        tests = tests_envs[test_env]['Instance']
        try:
            tests_envs[test_env]['MeanMicroAccuracy'] = sum(
                [test['Accuracy']['Micro'] for test in
                                             tests])/float(len(tests))
            tests_envs[test_env]['MeanMacroAccuracy'] = sum(
                [test['Accuracy']['Macro'] for test in
                                             tests])/float(len(tests))
        except TypeError:
            tests_envs[test_env]['MeanMicroAccuracy'] = 0
            tests_envs[test_env]['MeanMacroAccuracy'] = 0

    #keep n maximum tests, that exist
    sorted_test_envs = []
    listed_envs = sorted(tests_envs, key=lambda x: tests_envs[x]['MeanMicroAccuracy'],
                       reverse=True)
    cnt = bypass
    for env in listed_envs:
        if cnt == bypass + n:
            break
        valid = False
        if any([os.path.isdir(
            os.path.join(co.CONST['results_fold'],'Classification',
                             dataset,str(tests_envs[env]['Index'][0])))
                for dataset in classifier.available_tests]):
            valid = True
        for dataset in classifier.available_tests:
            if os.path.isdir(os.path.join(co.CONST['results_fold'],'Classification',
                                          dataset,str(tests_envs[env]['Index'][0]))):
                for fil in os.listdir(os.path.join(co.CONST['results_fold'],'Classification',
                             dataset,str(tests_envs[env]['Index'][0]))):
                    for ignored_key in ignore:
                        if ignored_key.lower() in fil.lower():
                            valid=False
            if valid:
                sorted_test_envs.append(tests_envs[env])
                cnt += 1
    return sorted_test_envs

def create_unified_tex(action_type,  filt, n=3, bypass=0, ignore = []):
    import subprocess,ast
    sorted_test_envs = retrieve_top_n_experiments(action_type,filt, n, bypass, ignore=ignore)
    dic = {'classifiers_used': 'rdf', 'descriptors': ['GHOG', 'ZHOF'], 'post_scores_processing_method': 'prob_check', 'ptpca': True, 'action_type': 'Dynamic', 'sparsecoding_level': 'Features'}
    classifier = Classifier(**dic)
    all_catalog = classifier.load_tests_mapping()
    save_res_path = os.path.join(co.CONST['results_fold'],
                                 'Classification',
                                 'BestOf')
    co.makedir(save_res_path)
    for count,testing in enumerate(sorted_test_envs):
        test_env_descr = ast.literal_eval(classifier.return_description(all_catalog, testing['Index'][0]))[1:]
        parameters_fil = 'Parameters-'+str(action_type)+'-' + str(count) + str(bypass) + '.pdf'
        graph = co.draw_oper.draw_nested(test_env_descr,'Parameters')
        graph.write_pdf(os.path.join(save_res_path, parameters_fil))
        pdffiles = []
        captions = []
        pdffiles.append(os.path.join(save_res_path, parameters_fil))
        captions.append('Παράμετροι Πειράματος σε μορφή Δέντρου')
        for test_cnt,ind in enumerate(testing['Index']):
            path = os.path.join(co.CONST['results_fold'],'Classification',
                             classifier.available_tests[test_cnt+1],str(ind))
            test_name = classifier.available_tests[test_cnt]
            preamble = 'Δεδομένα '+test_name+': '
            scores_cnt = 0
            try:
                for fil in os.listdir(path):
                    if 'Scores' in fil and fil.endswith('pdf'):
                        if 'Statistics' in fil:
                            captions.append(preamble + 'Μετρική Τυπικής Απόκλισης Scores για τον καθορισμό της έναρξης και της λήξης των δράσεων')
                            pdffiles.append(os.path.join(path,fil))
                        else:
                            if scores_cnt == 0:
                                captions.append(preamble + 'Scores του Ταξινομητή')
                                scores_cnt += 1
                                pdffiles.append(os.path.join(path,fil))
                for fil in os.listdir(path):
                    if 'Classification Results' in fil and fil.endswith('pdf'):
                        captions.append(preamble + 'Αποτελέσματα Ταξινόμησης')
                        pdffiles.append(os.path.join(path,fil))
                for fil in os.listdir(path):
                    if fil.endswith('tex'):
                        co.latex.compile(path, fil)
                for fil in os.listdir(path):
                    if fil.endswith('.tex'):
                        if 'Confusion' in fil:
                            captions.append(preamble + 'Πίνακας Σύγχυσης~(Confusion Matrix) των Δράσεων από τον Ταξινομητή') 
                        elif 'F1_Scores' in fil:
                            captions.append(preamble + 'F-Scores Μετρική της Ταξινόμησης')
                        elif 'accuracy' in fil:
                            captions.append(preamble + 'Ακρίβειες Ταξινόμησης')
                        pdffiles.append(os.path.join(path,
                                                 fil.replace('.tex', '.pdf')))
            except Exception as e:
                print e
                pass
        if action_type == 'Dynamic':
            data_to_write = co.latex.add_graphics(files=pdffiles, captions=captions,
                                                  nomargins=True,options='width=\maxwidth{1.2\linewidth}')
        else:
            data_to_write = co.latex.add_graphics(files=pdffiles, captions=captions,
                                                  options='width=\maxwidth{1\linewidth},keepaspectratio')
        with open(os.path.join(save_res_path,
                                'Results-'+action_type+'-' + str(count))+str(bypass)+'.tex','w') as out:
            out.write(data_to_write)

def create_single_tex_from_files(load_path, save_path, save_name=None,
                                 preamble='', action_type='dynamic'):
    for fil in os.listdir(load_path):
        if fil.endswith('tex'):
            co.latex.compile(load_path, fil)
    captions = []
    references = []
    pdffiles = []
    if 'dynamic' in action_type.lower()  or action_type.lower()=='cdbimm':
        gest = 'των \\textbf{δυναμικών}'
    elif 'passive' in action_type.lower():
        gest = 'των \\textbf{παθητικών}'
    elif ('sync' in action_type.lower() or
          'combined' in action_type.lower()):
        gest ='\\textbf{όλων} των'

    if action_type.lower() == 'cdbimm':
        clas = 'τον CDBIMM Ταξινομητή ('+gest+' χειρονομιών)'
    elif 'sync' in action_type.lower():
        clas = 'τους συγχρονισμένους Ταξινομητές Cl$_{pas}$ και CDBIMM'
    elif 'combined' in action_type.lower():
        clas = 'τον CRDF Ταξινομητή'
    else:
        clas = 'τον τελικό καλύτερο Ταξινομητή ' + gest + ' χειρονομιών '
    for fil in os.listdir(load_path):
        caption = preamble
        reference = action_type+'_'
        if fil.endswith('.tex'):
            if 'confusion' in fil.lower():
                caption += ('Πίνακας Σύγχυσης~(Confusion Matrix) '+
                            'από την ταξινόμηση ' + gest + ' χειρονομιών από ' + clas)
                reference += 'confusion'
                reference = 'tab:'+reference
            elif 'f1_scores' in fil.lower():
                caption += ('F-Scores και Ακρίβεια της ταξινόμησης για ' +
                            clas)
                reference += 'fscores'
                reference = 'tab:'+reference
            elif 'accurac' in fil.lower():
                caption += ('Ακρίβειες Συγχρονισμένων Ταξινομητών '+
                            gest + ' χειρονομιών')
                reference += 'sync_accuracies'
                reference = 'tab:'+reference
            elif 'times' in fil.lower():
                caption += ('Μέσοι Χρόνοι Επεξεργασίας ενός frame.'+
                            ' Παρατηρείται ότι η επεξεργασία ' +
                            'πραγματοποιείται σε πραγματικό χρόνο,'+
                            ' της τάξης των 30 ms.')
        elif  fil.endswith('pdf'):
            if 'Classification_Results' in fil:
                caption += ('Διάγραμμα Αποτελεσμάτων Ταξινόμησης για ' +
                            clas+', ως προς'+
                       ' τα frames.')
                if 'sync' in action_type.lower():
                    caption += ('Οι δύο συνεχείς γραμμές δείχνουν την ' +
                    'προβλεφθείσα κλάση ανά frame από κάθε ταξινομητή, ')
                else:
                    caption += ('Ως συνεχής με σταθερό χρώμα απεικονίζεται η προβλεφθείσα'
                       ' κλάση ανά frame από τον ταξινομητή, ')
                caption += ('ενώ με μεταβαλλόμενα χρώματα'+
                       ' παρουσιάζεται η επισήμανση των δεδομένων')
                if any(['statis' in f.lower() for f in os.listdir(load_path)]):
                    if 'sync' in action_type.lower():
                        caption += ('. Για την προβολή των ορίων προβλεπόμενης'
                                    + ' αρχής και τέλους κάθε εμφάνισης ' +
                                    ' χρησιμοποιείται το αποτέλεσμα που δίνει'+
                                    ' ο περιεχόμενος στον CDBIMM '+
                                    ' ταξινομητής CL$_{dyn}$, ο '+
                                    'οποίος κάνει χρήση της τεχνικής CSTD.')
                    else:
                        caption += ('. Η τεχνική CSTD επιτρέπει την εύρεση των ' +
                                'ορίων αρχής και τέλους κάθε εμφάνισης χειρον'+
                                'ομίας, γεγονός που υποδεικνύουν οι'+
                                ' κατακόρυφες συνεχείς γραμμές')
                reference += 'results_diag'
                reference = 'fig:'+reference
            elif 'Statistics' in fil:
                caption += ('Μετρική Τυπικής Απόκλισης Scores για τον'+
                            ' καθορισμό της έναρξης και της λήξης των' +
                            ' εμφανίσεων~(utterances)' + gest + ' χειρονομιών'
                           + ' από ' + clas)
                reference += 'std_metric'
                reference = 'fig:'+reference
            else:
                continue
        else:
            continue
        if 'macro' in fil.lower():
                caption += (' ως προς τις εμφανίσεις των χειρονομιών. ' +
                            'Μια χειρονομία θεωρείται πώς έχει ' +
                            'εντοπιστεί, αν το 50\% των frames ' +
                            'στη διάρκεια μιας εμφάνισης που ' +
                            'ορίζεται από το Ground Truth των ' +
                            'δεδομένων έχει χαρακτηριστεί από ' +
                            ('κάποιον από τους ταξινομητές' if
                             'sync' in action_type.lower() else
                            'τον ταξινομητή') +
                            ' ότι ανήκει σε αυτή. ' +
                            'Αν δεν υπάρχει τέτοια χειρονομία ' +
                            'για κάποια εμφάνιση, η εμφάνιση αυτή ' +
                            'θεωρείται αρνητικό δείγμα για όλες τις ' +
                            'χειρονομίες.')
                if 'sync' in action_type.lower():
                    caption += (' Λόγω αυτού του ορισμού, μια εμφάνιση'
                                + ' μπορεί να αποκτήσει δύο προβλεφθείσες'
                                + ' τιμές από τους δύο συγχρονισμένους ταξινομητές. '
                                + ' Τότε, αν κάποια από τις τιμές αυτές '
                                + 'αντιστοιχεί στην πραγματική επισήμανση'
                                + ' της εμφάνισης, η εμφάνιση θεωρείται '
                                + 'πως έχει ανιχνευθεί σωστά.')
                reference += '_macro'

        elif 'micro' in fil.lower():
                caption += (' ως προς τα frames.')
                reference += '_micro'
        else:
            caption += '.'
        captions.append(caption)
        references.append(reference)
        pdffiles.append(os.path.join(load_path,
                    fil.replace('.tex', '.pdf')))
    if action_type.lower() == 'dynamic':
        data_to_write = co.latex.add_graphics(files=pdffiles,
                                              captions=captions,
                                              labels=references,
                                              nomargins=True,options='width=\maxwidth{1.2\linewidth}')
    else:
        data_to_write = co.latex.add_graphics(files=pdffiles, captions=captions,
                                              labels=references,
                                              options='width=\maxwidth{1\linewidth},keepaspectratio')
    if save_name is None:
        save_name ='Results-'+action_type + '.tex'
    if not save_name.endswith('.tex'):
        save_name += '.tex'
    with open(os.path.join(save_path,
                            save_name),'w') as out:
        out.write(data_to_write)


def literal_eval(elems):
    if isinstance(elems,list) or isinstance(elems, tuple):
        res = []
        for elem in elems:
            try:
                elem = ast.literal_eval(elem)
            except:
                pass
            if isinstance(elem, list):
                res.append(literal_eval(elem))
            elif isinstance(elem, tuple):
                try:
                    elem = (elem[0],ast.literal_eval(elem[1]))
                except Exception as e:
                    pass
                if isinstance(elem[1], list) or isinstance(elem[1], tuple):
                    elem = (elem[0], literal_eval(elem[1]))
                res.append(elem)
        return res
    else:
        return elems
def convert_to_dict(elems):
    if isinstance(elems, list):
        res = []
        for elem in elems:
            if isinstance(elem , list):
                res.append(convert_to_dict(elem))
            elif isinstance(elem, tuple):
                if isinstance(elem[1], list) or isinstance(elem[1], tuple):
                    elem = (elem[0], convert_to_dict(elem[1]))
                    
                try:
                    res.append(dict(elem))
                except:
                    res.append(elem)
        try:
            res = dict(res)
        except:
            pass
        return res
    else:
        return elems
def merge_list_of_dicts(dicts):
    return { k: v for d in dicts for k, v in d.items() }
def join_rec_dicts(elems):
    dicts = []
    if isinstance(elems, list):
        for elem in elems:
            if isinstance(elem, dict):
                dicts.append(elem)
            else:
                if isinstance(elem, list):
                    dicts.append(join_rec_dicts(elem))
        return merge_list_of_dicts(dicts)
    else:
        return elems

def create_matrix(action_type, ignore=[]):
    dic = {'classifiers_used': 'rdf', 'descriptors': ['GHOG', 'ZHOF'], 'post_scores_processing_method': 'prob_check', 'ptpca': True, 'action_type': 'Dynamic', 'sparsecoding_level': 'Features'}
    classifier = Classifier(**dic)
    tests_instances = {}
    tests_envs = {}
    all_catalog = classifier.load_tests_mapping()
    import ast
    descs = []
    for count,name in enumerate(classifier.available_tests): 
        tests_instances = classifier.load_all_test_instances(count)
        for key in tests_instances:
            desc = classifier.return_description(all_catalog, key)
            if action_type.lower() not in str(desc).lower():
                continue
            for ignored_key in ignore:
                if ignored_key.lower() in str(desc).lower():
                    continue

            test_env = str(ast.literal_eval(str(desc))[1:]) 
            desc = join_rec_dicts(convert_to_dict(literal_eval(ast.literal_eval(desc))))
            keys= {}
            class_keys = {'Classifier':desc['Classifier']}
            '''class_keys = merge_list_of_dicts([
                {'Classifier':desc['Classifier']},
                    merge_list_of_dicts(
                        [{k:desc['ClassifierParams'][k]} for k in desc['ClassifierParams']])])
            '''
            feat_keys = {}
            if 'ptpca' in str(desc).lower():
                feat_keys['With PTPCA'] = True
            else:
                feat_keys['With PTPCA'] = False
            if 'sparsebuffer' in str(desc).lower():
                feat_keys['Sparse'] = 'Buffers'
            elif 'sparse' in str(desc).lower():
                feat_keys['Sparse'] = 'Features'
            else:
                feat_keys['Sparse'] = None
            descriptors_used = []
            for descriptor in desc['FeaturesParams']:
                descriptors_used.append(descriptor[0]['Descriptor'])
            feat_keys['Descriptors'] = str(sorted(descriptors_used))
            test_keys = {'Scores Proc. Method': desc['TestingParams']['post_scores_processing_method']}
            desc = merge_list_of_dicts([feat_keys, class_keys, test_keys])


            if  test_env not in tests_envs:
                tests_envs[test_env] = {}
            if 'Index' not in tests_envs[test_env]:
                tests_envs[test_env]['Index'] = []
            if 'Instance' not in tests_envs[test_env]:
                tests_envs[test_env]['Instance'] = []

            tests_envs[test_env]['Description'] = desc
            tests_envs[test_env]['Index'].append(key)
            tests_envs[test_env][
                'Instance'].append(tests_instances[key])
    list_of_experiments = []
    for test_env in tests_envs:
        tests = tests_envs[test_env]['Instance']
        try:
            tests_envs[test_env]['MeanMicroAccuracy'] = sum(
                [test['Accuracy']['Micro'] for test in
                                             tests])/float(len(tests))
            tests_envs[test_env]['MeanMacroAccuracy'] = sum(
                [test['Accuracy']['Macro'] for test in
                                             tests])/float(len(tests))
        except TypeError:
            tests_envs[test_env]['MeanMicroAccuracy'] = 0
            tests_envs[test_env]['MeanMacroAccuracy'] = 0
        list_of_experiments.append(merge_list_of_dicts([tests_envs[test_env]['Description'],
                             {'MeanMicroAccuracy':tests_envs[test_env]['MeanMicroAccuracy']},
                             {'MeanMacroAccuracy':tests_envs[test_env]['MeanMacroAccuracy']}]))
    return pd.DataFrame.from_dict(list_of_experiments).sort_values('MeanMacroAccuracy',ascending=False)


def process_generic(params, metrics, action_type, n=3, sort_met='Micro',
                    ret_just_params=False,
                    on_valid=False):
    mean_metrics = np.mean(metrics,axis=1)
    sort_met = 0 if sort_met.lower()=='micro' else 1
    best_n_inds = np.argsort(
        mean_metrics[:,sort_met])[-n:][::-1]
    best_params = params[best_n_inds[0]].copy()
    if ret_just_params:
        return best_params
    best_n_dict = {}
    keys_for_sorting = {}
    for count,ind in enumerate(best_n_inds):
        for param in params[ind]:
            field = param.replace('_',' ').title()
            if field=='Ptpca':
                field = 'PTPCA'
            if param.lower() == 'action_type':
                continue
            keys_for_sorting[field] = param
            if field not in best_n_dict:
                best_n_dict[field] = [None] * n
    param_keys = best_n_dict.keys()
    param_keys_to_sort = [keys_for_sorting[key] for key in param_keys]
    param_keys = [x for (y,x) in sorted(zip(param_keys_to_sort, param_keys))]

    best_n_dict['Mean Macro Accuracy'] = [None] * n
    best_n_dict['Mean Micro Accuracy'] = [None] * n
    save_path = os.path.join(co.CONST['results_fold'],'Classification',
                                   'TestingBest')
    co.makedir(save_path)

    for count,ind in enumerate(best_n_inds):
        for param in params[ind]:
            if param.lower() == 'action_type':
                continue
            field = param.replace('_',' ').title()
            if field=='Ptpca':
                field = 'PTPCA'
                if not params[ind][param]:
                    params[ind][param] = 'Not Used'
                else:
                    params[ind][param] = 'Used'
            best_n_dict[field][count] = str(params[ind][param])
        accuracies = mean_metrics[ind,:]
        best_n_dict['Mean Micro Accuracy'][count] = accuracies[0]
        best_n_dict['Mean Macro Accuracy'][count] = accuracies[1]
    import pandas as pd
    import pdfkit
    best_n_df = pd.DataFrame(best_n_dict)
    best_n_df = best_n_df[param_keys+['Mean Micro Accuracy',
                                      'Mean Macro Accuracy']]
    latex_best = best_n_df.to_latex(column_format='c'*
                                    (1+len(best_n_df.keys())))

    preamble = ('\\documentclass{standalone}\n ' +
                  '\\usepackage{booktabs}\n ' +
                  '\\begin{document}\n ')
    preamble += ('\\newcommand{\\specialcell}[2][c]{ \n'+
                 '\\begin{tabular}[#1]{@{}c@{}}#2\\end{tabular}}\n ')
    latex_best = preamble + latex_best
    latex_best = (latex_best + '\n \\end{document}')
    latex_best = co.latex.wrap_latex_table_entries(latex_best)
    with open(os.path.join(save_path,
                                action_type.title()+
                                'Validation.tex'),
              'w') as out:
        out.write(latex_best)
    if not on_valid:
        metrics, classifier, _, _ = perform_single_experiment(best_params,'test')
        create_single_tex_from_files(classifier.save_fold,
                          save_path, action_type.title(),
                          preamble='Δεδομένα Test: ',
                          action_type=action_type.title())
    else:
        metrics, classifiers, save_folds, names = \
        perform_single_experiment(best_params,'validation')
        for name,metric,save_fold in zip(names, metrics,save_folds):
            create_single_tex_from_files(save_fold,
                              save_path, (action_type+'_'+name),
                              preamble='Δεδομένα '+name+': ',
                              action_type=(action_type+'_'+name))
    return best_n_df.head(1)



def process_dynamic_actions(experiment=True, n=3, ret_just_params=False,
                            on_valid=False):
    import pickle
    if 'params_dynamic.pkl' in os.listdir('.') and not experiment:
        with open('params_dynamic.pkl', 'r') as inp:
            params, metrics = pickle.load(inp)
    else:
        params, metrics = perform_experiments_on_dynamic_actions('All')
        with open('params_dynamic.pkl', 'w') as out:
            pickle.dump((params, metrics), out)
    best_row = process_generic(params, metrics, 'dynamic', n=n,
                               sort_met='micro',
                               ret_just_params=ret_just_params,
                               on_valid=on_valid)
    return best_row



def process_passive_actions(experiment=True, n=3, ret_just_params=False,
                            on_valid=False):
    import pickle
    if 'params_passive.pkl' in os.listdir('.') and not experiment:
        with open('params_passive.pkl', 'r') as inp:
            params, metrics = pickle.load(inp)
    else:
        params, metrics = perform_experiments_on_passive_actions('All')
        with open('params_passive.pkl', 'w') as out:
            pickle.dump((params,metrics), out)
    best_row = process_generic(params, metrics, 'passive', n=n, sort_met='micro',
                               ret_just_params=ret_just_params,
                               on_valid=on_valid)
    return best_row

def process_dynamic_CDBIMM_actions(in_sync=False, load=False):
    import pickle
    if load:
        if 'cdbimm.pkl' in os.listdir('.'):
            with open('cdbimm.pkl','r') as inp:
                metrics, classifier = pickle.load(inp)
    pas_params = process_passive_actions(experiment=False,
                                         ret_just_params=True)
    dyn_params = process_dynamic_actions(experiment=False,
                                         ret_just_params=True)
    dyn_clas = Classifier(**dyn_params)
    dyn_clas.run_training()
    pas_clas = Classifier(**pas_params)
    pas_clas.run_training()

    params_dict = {
    'dynamic_classifier':dyn_clas,
    'passive_classifier':pas_clas,
        'in_sync':in_sync,
    'post_scores_processing_method':'CProb'}
  
    metrics, classifier, _, _ = perform_single_experiment(params_dict,'test',
                                                    variation='enhanced')
    save_path = os.path.join(co.CONST['results_fold'],'Classification',
                                   'TestingBest')
    create_single_tex_from_files(classifier.save_fold,
                      save_path, classifier.classifiers_used.replace(' ',''),
                      preamble='Δεδομένα Test: ',
                      action_type=classifier.classifiers_used)
    with open('cdbimm.pkl', 'w') as out:
        pickle.dump((metrics, classifier), out)
    return metrics, classifier

def process_valid_dynamic_CDBIMM_actions(in_sync=False, load=False):
    import pickle
    if load:
        if 'cdbimm.pkl' in os.listdir('.'):
            with open('cdbimm.pkl','r') as inp:
                metrics, classifier = pickle.load(inp)
    pas_params = process_passive_actions(experiment=False,
                                         ret_just_params=True)
    dyn_params = process_dynamic_actions(experiment=False,
                                         ret_just_params=True)
    dyn_clas = Classifier(**dyn_params)
    dyn_clas.run_training()
    pas_clas = Classifier(**pas_params)
    pas_clas.run_training()

    params_dict = {
    'dynamic_classifier':dyn_clas,
    'passive_classifier':pas_clas,
        'in_sync':in_sync,
    'post_scores_processing_method':'CProb'}
  
    metrics, classifier, _, _ = perform_single_experiment(
        params_dict,'validation',
                                                    variation='enhanced')
    print metrics
    return

def process_combined_actions():
    pas_params = process_passive_actions(experiment=False,
                                         ret_just_params=True)
    dyn_params = process_dynamic_actions(experiment=False,
                                         ret_just_params=True)
    dyn_clas = Classifier(**dyn_params)
    dyn_clas.run_training()
    pas_clas = Classifier(**pas_params)
    pas_clas.run_training()

    params_dict = {
    'dynamic_classifier':dyn_clas,
    'passive_classifier':pas_clas}
  
    metrics, classifier, _, _ = perform_single_experiment(params_dict,'test',
                                                    variation='combined')
    save_path = os.path.join(co.CONST['results_fold'],'Classification',
                                   'TestingBest')
    create_single_tex_from_files(classifier.save_fold,
                      save_path, classifier.classifiers_used.replace(' ',''),
                      preamble='Δεδομένα Test: ',
                      action_type=classifier.classifiers_used)
    

def create_CDBIMM_CLDYN_table():
    dyn_row = process_dynamic_actions(False, ret_just_params=True)
    _, cdbimm = process_dynamic_CDBIMM_actions(in_sync=False, load=True)
    clas_dyn = Classifier(**dyn_row)
    clas_dyn.run_training()
    clas_dyn.run_testing('Test',online=False)
    dyn_fscores = clas_dyn.testdata[clas_dyn.test_ind]['FScores']
    cdbimm_fscores = cdbimm.testdata[cdbimm.test_ind]['FScores']
    dyn_accuracy = clas_dyn.testdata[clas_dyn.test_ind]['Accuracy']
    cdbimm_accuracy = cdbimm.testdata[cdbimm.test_ind]['Accuracy']
    labels = clas_dyn.train_classes + ['Accuracy']
    for typ in ['Macro', 'Micro']:
        dyn_fscores[typ]
        cdbimm_fscores[typ]
        accuracies = np.array([[dyn_accuracy[typ]],
                                       [cdbimm_accuracy[typ]]])
        fscores = np.vstack((
                dyn_fscores[typ][0],
                cdbimm_fscores[typ][0]))
        save_path = os.path.join(co.CONST['results_fold'],'Classification',
                                       'TestingBest')
        with open(os.path.join(save_path,
                               typ+'_CDBIMM_CLDYN_compare.tex'),'w') as out:
            out.write(co.latex.array_transcribe([fscores,accuracies],
                                  xlabels = labels,
                                  ylabels = ['Cl$_{dyn}$',
                                             'CDBIMM'],
                                  extra_locs='right',
                                  sup_x_label=typ
                                  + ' Metrics CDBIMM-CL$_{dyn}$ comparison',
                      wrap=False))

def create_best_classifiers_table(dynamic_row, passive_row):
    import pandas as pd
    df = pd.concat([dynamic_row, passive_row] ,keys=['Clasdyn',
          'Claspas'])
    df = df.fillna('cellcolor')
    params = df.keys().tolist()
    params.remove('Mean Micro Accuracy')
    params.remove('Mean Macro Accuracy')
    df = df[sorted(params) + ['Mean Micro Accuracy','Mean Macro Accuracy']]
    latex_table = co.latex.array_transcribe(df, isdataframe=True,
                                            ylabels=['Cl$_{dyn}$',
                                                     'Cl$_{pas}$'])
    latex_table = co.latex.add_package(latex_table,
                         'xcolor','table')
    latex_table = latex_table.replace('cellcolor','\\cellcolor{black}')
    save_path = os.path.join(co.CONST['results_fold'],'Classification',
                                   'TestingBest','BestClassifiers.tex')
    with open(save_path, 'w') as out:
        out.write(latex_table)

if __name__=='__main__':
    dyn_row = process_dynamic_actions(False)
    process_dynamic_actions(False,on_valid=True)
    pas_row = process_passive_actions(False)
    process_passive_actions(False,on_valid=True)
    create_best_classifiers_table(dyn_row, pas_row)
    _ , cdbimm = process_dynamic_CDBIMM_actions(in_sync=False)
    process_valid_dynamic_CDBIMM_actions(in_sync=False)
    create_CDBIMM_CLDYN_table()
    process_dynamic_CDBIMM_actions(in_sync=True)
    process_combined_actions()
