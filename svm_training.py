import numpy as np
import class_objects as co
from action_recognition_alg import *
import os.path
import cPickle as pickle
import logging
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from matplotlib import pyplot as plt
print_info=True
logging.basicConfig(format='%(levelname)s:%(message)s',
                    level=logging.INFO if print_info else
                    logging.WARNING)

dexter1_path='/media/vassilis/Data/Thesis/Datasets/dexter1/data/'
actions=['adbadd','fingercount','fingerwave','flexex1','pinch','random','tigergrasp']
suffix='/tof/depth/'
logging.info('Adding actions...')
actions=[dexter1_path+action+suffix for action in actions]

action_recog=ActionRecognition()
path='scores.pkl'
try:
    with open(path,'r') as inp:
        logging.info('Loading test results..')
        scores=pickle.load(inp)
except IOError,EOFError:
    path=action_recog.actions.save_path
    try:
        with open(path,'r') as inp:
            actions=pickle.load(inp)
            sparse_features_lists=[]
            for action in actions:
                sparse_features_lists.append(action.sparse_features)
    except (EOFError,IOError) as e:
        for action in actions:
            action_recog.add_action(action)
        logging.info('Training dictionaries..')
        path=action_recog.dictionaries.save_path
        try:
            with open(path,'r') as inp:
                dicts=pickle.load(inp)
        except (EOFError,IOError) as e:
            dicts=action_recog.train_sparse_dictionaries(act_num=0,
                                                         print_info=True)
        logging.info('Making sparse features')
        sparse_features_lists = action_recog.actions.update_sparse_features(dicts)
        action_recog.actions.save()
    #sparse_features_lists is a list of lists of sparse features per action
    #To train i-th svm we get sparse_features_lists[i]
    path='unified_classifier.pkl'
    logging.info('Checking if trained SVMs exist..')
    try:
        with open(path,'r') as inp:
            logging.info('Loading existent trained SVM classifier')
            unified_classifier, svms_1_v_all_traindata=pickle.load(inp)
    except (EOFError,IOError) as e:
        logging.info('Preparing svm traindata')
        svm_initial_training_data=[]
        for sparse_features in sparse_features_lists:
            svm_initial_training_data.append(np.concatenate(tuple(sparse_features),
                                             axis=0))
        buffer_size = 20
        svms_buffers = []
        for data in svm_initial_training_data:
            svm_buffers = []
            for count in range(data.shape[1]-buffer_size):
                svm_buffers.append(np.atleast_2d(data[:, count:count + buffer_size].ravel()))
            svms_buffers.append(svm_buffers)
        logging.info('Train Data has '+str(len(svms_buffers)) +
                     ' buffer lists. First buffer list has length ' +
                     str(len(svms_buffers[0])) +
                     ' and last buffer has shape '+
                     str(svms_buffers[0][-1].shape))
        logging.info('Joining buffers..')
        svms_training_data = []
        for svm_buffers in svms_buffers:
            svms_training_data.append(np.concatenate(tuple(svm_buffers),axis=0))

        logging.info('Train Data has '+str(len(svms_training_data))+
                     ' training datasets for each action. Shape of first dataset is '+
                     str(svms_training_data[0].shape))
        logging.info('Creating class holding vector and concatenating remaining data..')
        svm_classes = []
        svms_1_v_all_traindata=np.concatenate(tuple(svms_training_data),axis=0)
        for count,data in enumerate(svms_training_data):
            svm_classes.append(count*np.ones((data.shape[0])))
        svm_classes = np.concatenate(tuple(svm_classes),axis=0)
        logging.info('Final training data to be used as input to OneVsRestClassifier'+
                     ' has shape '+str(svms_1_v_all_traindata.shape))
        num_of_cores=4
        logging.info('Training SVMs using '+str(num_of_cores)+' cores..')
        unified_classifier=OneVsRestClassifier(SVC(kernel='linear',probability=True),num_of_cores).fit(svms_1_v_all_traindata,
                                                                  svm_classes)
        with open(path,'wb') as outp:
            logging.info('Saving trained classifier and training data')
            pickle.dump((unified_classifier,svms_1_v_all_traindata),outp)


    logging.info('Classifier contains ' +
                 str(len(unified_classifier.estimators_)) + ' estimators')
    logging.info('Testing SVMS using training data..')
    #scores=unified_classifier.predict_proba(svms_1_v_all_traindata)
    scores=unified_classifier.decision_function(svms_1_v_all_traindata)
    path='scores.pkl'
    with open(path,'wb') as outp:
        logging.info('Saving test results..')
        pickle.dump(scores,outp)




box_filter_shape=20
box_filter=np.ones(box_filter_shape)/float(box_filter_shape)
filtered_scores=np.apply_along_axis(np.convolve,0,scores,box_filter,mode='same')
scores_std = np.std(scores,axis=1)
scores_mean = np.mean(scores,axis=1)
filtered_scores_std = np.std(filtered_scores,axis=1)
filtered_scores_mean = np.mean(filtered_scores,axis=1)

plt.figure(1)
plt.plot(scores_std,color='r',label='STD')
plt.plot(scores_mean,color='g',label='Mean')
plt.legend()
plt.title('Scores Statistics')
plt.xlabel('Frames')

plt.figure(2)
plt.plot(filtered_scores_std,color='r',label='STD')
plt.plot(filtered_scores_mean,color='g',label='Mean')
plt.legend()
plt.title('Filtered Scores Statistics')
plt.xlabel('Frames')
plt.show()


