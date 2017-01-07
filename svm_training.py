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

count=0
visualize=False
log_lev='INFO'
reset=False
for _,arg in enumerate(sys.argv[1:]):
    try:
        [arg_name,arg_val]=arg.split('=')
        if arg_name == 'log_lev':
            log_lev=arg_val
        elif arg_name == 'visualize':
            visualize=True if arg_val==('True' or 1) else False
        elif arg_name == 'reset':
            reset=arg_val
    except:
        if count==0:
            log_lev=True if arg=='True' else False
        elif count==1:
            visualize = True if arg=='True' else False
        elif count==2:
            reset=True if arg=='True' else False
    count=count+1
logging.basicConfig(format='%(levelname)s:%(message)s')
logging.getLogger().setLevel(log_lev)

dexter1_path='/media/vassilis/Thesis/Datasets/dexter1/data/'
action_names=['adbadd','fingercount','fingerwave','flexex1','pinch','random','tigergrasp']
suffix='/tof/depth/'
logging.info('Adding actions...')
actions=[dexter1_path+action+suffix for action in action_names]

action_recog=ActionRecognition(log_lev)
path='scores.pkl'
try:
    if reset:
        raise IOError
    with open(path,'r') as inp:
        logging.info('Loading test results..')
        scores,svm_classes=pickle.load(inp)
except (IOError,EOFError) as e:
    path=action_recog.actions.save_path
    try:
        if reset:
            raise IOError
        with open(path,'r') as inp:
            actions=pickle.load(inp)
            sparse_features_lists=[]
            for action in actions:
                sparse_features_lists.append(action.sparse_features)
    except (EOFError,IOError) as e:
#FARMING FEATURES STAGE
        for action in actions:
            action_recog.add_action(action,visualize=visualize)
        logging.info('Training dictionaries..')
        path=action_recog.dictionaries.save_path
        try:
            if reset:
                raise IOError
            with open(path,'r') as inp:
                dicts=pickle.load(inp)
        except (EOFError,IOError) as e:
#DICTIONARIES TRAINING STAGE
            dicts=action_recog.train_sparse_dictionaries(act_num=0,
                                                         print_info=True)
        logging.info('Making sparse features')
        sparse_features_lists = (action_recog.actions.
                                 update_sparse_features(dicts,
                                                       ret_sparse=True))
        action_recog.actions.save()
    #sparse_features_lists is a list of lists of sparse features per action
    #To train i-th svm we get sparse_features_lists[i]
    path='unified_classifier.pkl'
    logging.info('Checking if trained SVMs exist..')
    try:
        if reset:
            raise IOError
        with open(path,'r') as inp:
            logging.info('Loading existent trained SVM classifier')
            unified_classifier, svms_1_v_all_traindata,svm_classes=pickle.load(inp)
    except (EOFError,IOError) as e:
#TRAINING SVM CLASSIFIER STAGE
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
        training_data_shapes=[data.shape[0] for data in svms_training_data]
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
        unified_classifier=OneVsRestClassifier(LinearSVC(),num_of_cores).fit(svms_1_v_all_traindata,
                                                                             svm_classes)
        with open(path,'wb') as outp:
            logging.info('Saving trained classifier and training data')
            pickle.dump((unified_classifier,svms_1_v_all_traindata,svm_classes),outp)


    logging.info('Classifier contains ' +
                 str(len(unified_classifier.estimators_)) + ' estimators')
    logging.info('Testing SVMS using training data..')
    #scores=unified_classifier.predict_proba(svms_1_v_all_traindata)
    scores=unified_classifier.decision_function(svms_1_v_all_traindata)
    path='scores.pkl'
    with open(path,'wb') as outp:
        logging.info('Saving test results..')
        pickle.dump((scores,svm_classes),outp)


box_filter_shape = 30
box_filter = np.ones(box_filter_shape)/float(box_filter_shape)
mean_filter_shape = 300
mean_filter=np.ones(mean_filter_shape)/float(mean_filter_shape)
filtered_scores = np.pad(np.apply_along_axis(np.convolve, 0, scores,box_filter, mode='valid'),
                         ((box_filter_shape/2, box_filter_shape/2),(0, 0)),'edge')
filtered_scores_std = np.std(filtered_scores,axis=1)
filtered_scores_std_mean = np.pad(np.convolve(filtered_scores_std,
                                              mean_filter,mode='valid'),
                                  (mean_filter_shape/2-1,mean_filter_shape/2),'edge')
mean_diff=filtered_scores_std_mean-filtered_scores_std
positive = mean_diff > 0
zero_crossings = np.where(np.bitwise_xor(positive[1:], positive[:-1]))[0]
interesting_crossings = np.concatenate((np.array([0]),
                                        zero_crossings[::2],
                                        np.array([mean_diff.size])),axis=0)
identified_classes=[]
for cross1,cross2 in zip(interesting_crossings[:-1],interesting_crossings[1:]):
    clas=np.median(scores[cross1:cross2,:],axis=0).argmax()
    identified_classes.append(clas*np.ones(cross2-cross1))
identified_classes = np.concatenate(tuple(identified_classes),axis=0)

plt.figure(2)
plt.plot(filtered_scores_std,color='r',label='STD')
plt.plot(filtered_scores_std_mean,color='g',label='STD Mean')
plt.plot((svm_classes)*np.max(filtered_scores_std)/float(np.max(svm_classes)),
         label='Ground Truth')
plt.legend()
plt.title('Filtered Scores Statistics')
plt.xlabel('Frames')

plt.figure(3)
for count,score in enumerate(filtered_scores.T):
    plt.plot(score,
             label=action_names[count])
plt.legend()
plt.title('Filtered Scores')
plt.xlabel('Frames')
plt.figure(4)
plt.plot((mean_diff-np.min(mean_diff))/float(np.max(mean_diff)-np.min(mean_diff)),
         label='Filtered scores normalized mean difference')
plt.plot(svm_classes/float(np.max(svm_classes)),label='Ground Truth')
plt.legend()
plt.figure(5)
plt.plot(svm_classes,label='Ground Truth')
plt.plot(identified_classes,label='Identified Classes')
plt.xlabel('Frames')
plt.ylim((svm_classes.min()-1,svm_classes.max()+1))
plt.title('Result')
plt.legend()
plt.show()






