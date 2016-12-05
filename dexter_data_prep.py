import os
import cv2
import numpy as np
import yaml
import glob
import time
import progressbar
import action_recognition_alg as ara
import cPickle as pickle
import matplotlib.pyplot as plt
import class_objects as co
import sparse_coding as sc
def feature_sign_search_step(hof_features,hog_features,
                             hof_bmat=None,hog_bmat=None):
    hof_coding = sc.FeatureSignSearch()
    hog_coding = sc.FeatureSignSearch()
    hog_coding.max_iter = 1000
    hof_coding.max_iter = 1000
    des_dim = 2 * hof_features.shape[0]
    dist_sigma = 0.1
    dist_beta = 0.1
    expanded_hof = np.zeros((des_dim, hof_features.shape[1]))
    expanded_hog = np.zeros_like(expanded_hof)
    count = 0
    hof_error = []
    hof_prev_error = []
    hog_error = []
    hog_prev_error = []
    bar= progressbar.ProgressBar()
    for count in bar(range(hog_features.shape[1])):
        hof = hof_features[:, count]
        hog = hog_features[:, count]
        (err, sing) = hof_coding.feature_sign_search_algorithm(hof[:, None].
                                                               astype(float),
                                                               des_dim,
                                                               dist_sigma,
                                                               dist_beta,
                                                               hof_bmat)
        hof_error.append(err)
        hof_prev_error.append(hof_coding.prev_err)
        if sing:
            print 'HOF:Wrongly handled singularity met while processing:\n\t', hof
        (err, sing) = hog_coding.feature_sign_search_algorithm(hog[:, None].
                                                               astype(float),
                                                               des_dim,
                                                               dist_sigma,
                                                               dist_beta,
                                                               hog_bmat)
        hog_error.append(err)
        hog_prev_error.append(hog_coding.prev_err)
        if sing:
            print 'HOG:Wrongly handled singularity met while processing:\n\t', hog
        expanded_hof[:, count] = hof_coding.out_features.ravel()
        expanded_hog[:, count] = hog_coding.out_features.ravel()
        hof_coding.flush_variables()
        hog_coding.flush_variables()
        count += 1
    hof_bmat = hof_coding.bmat.copy()
    hog_bmat = hog_coding.bmat.copy()
    return(expanded_hof,expanded_hog,
           hof_bmat,hog_bmat,
           hof_error,hog_error)

with open("config.yaml", 'r') as stream:
    try:
        CONST = yaml.load(stream)
    except yaml.YAMLError as exc:
        print exc

path = '/media/vassilis/Data/Thesis/Datasets/dexter1/data/pinch/tof/depth'
imgs = [cv2.imread(filename, -1) for filename in glob.glob(path + '/*.png')]
img = imgs[0]
fmining = ara.ActionRecognition()
features = []
co.counters.im_number = 0
pkl_file = 'pinch.pkl'
mine_features = False
hof_features=None
hog_features=None
if not os.path.isfile(pkl_file) or mine_features:
    print 'Mining features..'
    bar= progressbar.ProgressBar()
    for count,img in enumerate(bar(imgs)):
        binmask = img < 6000
        mask = np.zeros_like(img)
        mask[binmask] = img[binmask]
        mask = mask / (np.max(mask)).astype(float)
        contours = cv2.findContours(
            (binmask).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]
        contours_area = [cv2.contourArea(contour) for contour in contours]
        hand_contour = contours[np.argmax(contours_area)].squeeze()
        hand_patch = mask[np.min(hand_contour[:, 1]):np.max(hand_contour[:, 1]),
                          np.min(hand_contour[:, 0]):np.max(hand_contour[:, 0])]
        med_filt = np.median(hand_patch[hand_patch != 0])
        binmask[np.abs(mask - med_filt) > 0.2] = False
        hand_patch[np.abs(hand_patch - med_filt) > 0.2] = 0
        hand_patch *= 256
        hand_patch = hand_patch.astype(np.uint8)
        '''
        cv2.imshow('whole',img)
        cv2.imshow('binmask',binmask.astype(float))
        cv2.imshow('test',hand_patch)
        cv2.waitKey(0)
        '''
        hand_patch_pos = np.array(
            [np.min(hand_contour[:, 1]), np.min(hand_contour[:, 0])])
        co.data.uint8_depth_im = (256 * mask).astype(np.uint8)
        co.meas.found_objects_mask = binmask
        co.counters.im_number += 1
        fmining.update(hand_patch, hand_patch_pos)
        if co.counters.im_number >= 2:
            hof, hog = fmining.extract_features(CONST)
            if hof_features is None:
                hof_features=np.zeros((hof.shape[0],len(imgs)))
            hof_features[:,count]=hof
            if hog_features is None:
                hog_features=np.zeros((hog.shape[0],len(imgs)))
            hog_features[:,count]=hog

    with open(pkl_file, 'wb') as output:
        pickle.dump((hof_features, hog_features), output)
else:
    with open(pkl_file, 'r') as inp:
        (hof_features, hog_features) = pickle.load(inp)



pkl_file = 'sparse_coded.pkl'
sparsify_features = True

if not os.path.isfile(pkl_file) or sparsify_features:
    print 'Converting features to sparse..'
    (expanded_hof, expanded_hog,
     hof_bmat, hog_bmat,
     hof_error, hog_error)=feature_sign_search_step(hof_features,hog_features)
    with open(pkl_file, 'wb') as output:
        pickle.dump((expanded_hof, expanded_hog,
                     hof_bmat, hog_bmat,
                     hof_error, hog_error), output)
else:
    with open(pkl_file, 'r') as inp:
        (expanded_hof, expanded_hog,
         hof_bmat, hog_bmat,
         hof_error, hog_error) = pickle.load(inp)
print hof_error
print 'HOF sparse coding:'
print '\t Mean Reconstruction Error:'
try:
    out = hof_prev_error
    print '\t\tPrevious:', np.mean(out)
except NameError:
    pass
print '\t\tCurrent:', np.mean(hof_error)
print '\t Reconstruction Error std:'
try:
    out = hof_prev_error
    print '\t\tPrevious:', np.std(out)
except NameError:
    pass
print '\t\tCurrent:', np.std(hof_error)
print 'HOG sparse coding:'
print '\t Mean Reconstruction Error:'
try:
    out = hog_prev_error
    print '\t\tPrevious:', np.mean(out)
except NameError:
    pass
print '\t\tCurrent:', np.mean(hog_error)
print '\t Reconstruction Error std:'
try:
    out = np.std(hog_prev_error)
    print '\t\tPrevious:', out
except NameError:
    pass
print '\t\tCurrent:', np.std(hog_error)

pkl_file = 'pinch_dict.pkl'
train_dictionaries = True

if not os.path.isfile(pkl_file) or train_dictionaries:
    hof_coding = sc.FeatureSignSearch()
    hog_coding = sc.FeatureSignSearch()
    hof_coding.inp_features = hof_features.copy()
    hog_coding.inp_features = hog_features.copy()
    hof_coding.out_features = expanded_hof.copy()
    hog_coding.out_features = expanded_hog.copy()
    print 'Training dictionaries..'
    for _ in range(3):
        hof_coding.bmat = hof_bmat.copy()
        hog_coding.bmat = hog_bmat.copy()
        hof_coding.display=0
        print 'Training 3DHOF dictionary..'
        hof_bmat = hof_coding.dictionary_training()
        hog_coding.display=0
        print 'Training GHOG dictionary..'
        hog_bmat = hog_coding.dictionary_training()
        print 'Readjusting sparse features..'
        tup=feature_sign_search_step(hof_features,
                                     hog_features,
                                     hof_bmat.copy(),
                                     hog_bmat.copy())
        expanded_hof=tup[0]
        expanded_hog=tup[1]
        hof_coding.out_features = expanded_hof.copy()
        hog_coding.out_features = expanded_hog.copy()
        print 'HOF dictionary training:'
        print '\t Reconstruction Error:'
        print '\t\tPrevious:', hof_coding.prev_err
        hof_curr_error = np.linalg.norm(hof_coding.inp_features -
                                        np.dot(hof_bmat,
                                               hof_coding.out_features))

        print '\t\tCurrent:', hof_curr_error

        print 'HOG dictionary training:'
        print '\t Reconstruction Error:'
        print '\t\tPrevious:', hog_coding.prev_err
        hog_curr_error = np.linalg.norm(hog_coding.inp_features -
                                        np.dot(hog_bmat,
                                               hog_coding.out_features))
        print '\t\tCurrent:', hog_curr_error
    hof_dictionary = np.linalg.pinv(hof_bmat)
    hog_dictionary = np.linalg.pinv(hog_bmat)

    with open(pkl_file, 'wb') as output:
        pickle.dump((hof_dictionary, hog_dictionary,
                     [hof_coding.prev_err,
                      hof_curr_error],
                     [hog_coding.prev_err,
                      hog_curr_error]), output)
else:
    with open(pkl_file, 'r') as inp:
        (hof_dictionary, hog_dictionary, hof_errors, hog_errors) = pickle.load(inp)
