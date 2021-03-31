from functools import partial
import numpy as np
import pandas as pd
import os

from src.integral_image import IntegralImage as integral
from src.haar_features import HaarLikeFeature as haar
from src.haar_features import featureType
from src.utils import *


# for processing time
import progressbar
from multiprocessing import cpu_count, Pool


LOADING_BAR_LENGTH = 25


def _create_features(img_width, img_height, min_feat_width, max_feat_wid, min_feat_height, max_feat_height):
    # private function to create all possible features
    # return haar_feats: a list of haar-like features
    # return type: np.array(haar.HaarLikeFeature)

    haar_feats = list()

    # iterate according to types of rectangle features
    for feat_type in featureType:

        # min of feature width is set in featureType enum
        feat_start_width = max(min_feat_width, feat_type.value[0])

        # iterate with step 
        for feat_width in range(feat_start_width, max_feat_wid, feat_type.value[0]):

            # min of feature height is set in featureType enum
            feat_start_height = max(min_feat_height, feat_type.value[1])
            
            # iterate with setp
            for feat_height in range(feat_start_height, max_feat_height, feat_type.value[1]):

                # scan the whole image with sliding windows (both vertical & horizontal)
                for i in range(img_width - feat_width):
                    for j in range(img_height - feat_height):
                        haar_feats.append(haar(feat_type, (i,j), feat_width, feat_height)) 
                        # haar_feats.append(haar(feat_type, (i,j), feat_width, feat_height, 0, -1)) # threshold = 0 (no misclassified images)

    return haar_feats


def _get_feature_score(feature, image):
    # para feature: HaarLikeFeature object
    # para image: integral image
    return feature.calc_score(image)


def _update_feature(feature, weights, labels):
    # updated: list to store the updated features
    data = np.vstack((feature.scores, weights, labels)) # combine data
    df = pd.DataFrame(data.T, columns=['score','weight','label']) # create data frame
    df.sort_values(by=['score']) # ascending order

    # Total sums
    T_pos = df.loc[df['label'] == 1, 'weight'].sum()
    T_neg = df.loc[df['label'] == 0, 'weight'].sum()

    # find best threshold
    for threshold in df['score'].tolist():
        S_pos = df.loc[ (df['label'] == 1) & (df['score'] < threshold), 'weight'].sum()
        S_neg = df.loc[ (df['label'] == 0) & (df['score'] < threshold), 'weight'].sum()
        # find polarity
        error_1 = S_pos + T_neg - S_neg
        error_2 = S_neg + T_pos - S_pos
        if error_1 < error_2:
            min_error = error_1
            feature.polarity = -1
        else:
            min_error = error_2
            feature.polarity = 1
        # update error and feature threshold
        if min_error < feature.error: # initial feature.error is inf
            feature.error = min_error
            feature.threshold = threshold
    # print progress dot
    # print('.', end='')
    return feature
    

def save_scores(scores):
    np.savetxt("./data/scores.txt", scores, fmt='%f')
    print("...scores saved\n")


def load_scores():
    scores = np.loadtxt("./data/scores.txt", dtype=np.float64)
    return scores


def find_best_features(features, pos_int_imgs, neg_int_imgs, pos_weights=-1, neg_weights=-1, n=1, verbose=1):
    # n: number of features to return
    # return: list of festures and their indexes
    
    num_pos = len(pos_int_imgs)
    num_neg = len(neg_int_imgs)
    num_imgs = num_pos + num_neg
    num_features = len(features)

    # default weights if unspecified
    if pos_weights == -1:
        pos_weights = np.ones(num_pos) * 1. / (2 * num_pos) # w = 1/2m
    if neg_weights == -1:
        neg_weights = np.ones(num_neg) * 1. / (2 * num_neg) # w = 1/2l
    weights = np.hstack((pos_weights, neg_weights)) # concatenated weights of images
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg))) # concatenated labels (pos/neg)
    images = pos_int_imgs + neg_int_imgs # concatenated image samples

    if verbose:
        print("\ncalculating integral images ...")

    if os.path.exists("./data/scores.txt"):
        scores = load_scores()
    else:
        # 2D numpy.array, each row is a feature with all image scores
        scores = np.zeros((num_features, num_imgs))

        # visualise learning progress with text signals
        bar = progressbar.ProgressBar()

        # pool object to parallelize the execution of a function across multiple input values
        NUM_PROCESS = cpu_count() * 3 # 8 on T580
        pool = Pool(processes=NUM_PROCESS)

        # get all scores for each image and each feature (quite time-consuming)
        for i in bar(range(num_imgs)):
            scores[:, i] = np.array(list(pool.map(partial(_get_feature_score, image=images[i]), features)))

        save_scores(scores)

    '''
    The partial() is used for partial function application which 'freezes' some portion of a function's arguments and/or keywords
    resulting in a new object with a simplified signature
    In the project, (partial(_get_feature_vote, image=images[i]), features), fixed the argument features
    for i in range(len(features)):
        for j in range(len(images)):
            vote[i][j] = _get_feature_vote(features[i], images[j])
    '''

    # Update score attribute in all features
    for idx, feature in enumerate(features):
        feature.scores = scores[idx,:]

    # Compute error for each feature (weak learner) and update their threshold and polarity
    if verbose:
        print("\nupdating feature error and threshold ...")
    
    # Update error and threshold of all features (quite time-consuming)
    pool = Pool(processes=NUM_PROCESS)
    updated_features = pool.map(partial(_update_feature, weights=weights , labels=labels), features)
    # TODO: save features in file
 
    # Sort by classification error and retreive indexes
    idxs = [i[0] for i in sorted(enumerate(updated_features), key=lambda x:x[1].error)] 
    
    best_features = list()
    for k in range(n):
        index = idxs[k]
        best_features.append(updated_features[index])

    return best_features, idxs[:n]


def learn(pos_int_img, neg_int_img, num_classifiers=-1, min_feat_width=1, max_feat_width=-1, min_feat_height=1, max_feat_height=-1, verbose=False):
    # select a set of classifiers, iteratively taking the best classifiers based on a weighted error
    # implementation of AdaBoost algorithm (note pos/1 neg/0)
    '''
    : para pos_int_img, neg_int_img: list of pos/neg integral images
    : type pos_int_img, neg_int_img: list[np.ndarray]
    : para num_classifier: number of classifiers to select, default to use all classifier
    : type num_classifier: int
    : para verbose: whether to print
    : type verbose: boolean
    : return: list of selected features (one classifier has only one feature)
    : rtype: list[haar.HaarLikeFeature]
    '''
    num_pos = len(pos_int_img)
    num_neg = len(neg_int_img)
    num_imgs = num_pos + num_neg
    img_height, img_width = pos_int_img[0].shape

    # maximum features width and height default to image width and height
    # note MAX is noted with 'feature' not 'feat'
    max_feature_width = img_width if max_feat_width == -1 else max_feat_width
    max_feature_height = img_height if max_feat_height == -1 else max_feat_height

    # initialise weights and labels (weights of all image samples)
    pos_weights = np.ones(num_pos) * 1. / (2 * num_pos) # w = 1/2m
    neg_weights = np.ones(num_neg) * 1. / (2 * num_neg) # w = 1/2l
    weights = np.hstack((pos_weights, neg_weights)) # concatenated weights of images
    labels = np.hstack((np.ones(num_pos), np.zeros(num_neg))) # concatenated labels (pos/neg)

    # training images list
    images = pos_int_img + neg_int_img # concatenated image samples

    if verbose:
        print("\ncreating haar-like features ...")
    # all the possible features must be quite time consuming
    features = _create_features(img_width, img_height, min_feat_width, max_feature_width, min_feat_height, max_feature_height)

    if verbose:
        print('... done. %d features were created!' % len(features))

    num_features = len(features)
    feature_index = list(range(num_features)) # save manipulation of data

    # preset number of weak learners (classifiers) [under control]
    num_classifiers = num_features if num_classifiers == -1 else num_classifiers

    if verbose:
        print("\ncalculating integral images ...")

    if os.path.exists("./data/scores.txt"):
        scores = load_scores()
    else:

        # 2D numpy.array, each row is a feature with all image scores
        scores = np.zeros((num_features, num_imgs))

        # visualise learning progress with text signals
        bar = progressbar.ProgressBar()
        # pool object to parallelize the execution of a function across multiple input values
        NUM_PROCESS = cpu_count() * 3 # 8 on T580
        pool = Pool(processes=NUM_PROCESS)

        # get all scores for each image and each feature (quite time-consuming)
        for i in bar(range(num_imgs)):
            scores[:, i] = np.array(list(pool.map(partial(_get_feature_score, image=images[i]), features)))

        save_scores(scores)

    '''
    The partial() is used for partial function application which 'freezes' some portion of a function's arguments and/or keywords
    resulting in a new object with a simplified signature
    In the project, (partial(_get_feature_vote, image=images[i]), features), fixed the argument features
    for i in range(len(features)):
        for j in range(len(images)):
            vote[i][j] = _get_feature_vote(features[i], images[j])
    '''

    # TODO: make the function _get_feature_score to update the feature attribute: updated_feature = _get_feature_score()
    # Update score attribute in all features. This is a workaround.
    for idx, feature in enumerate(features):
        feature.scores = scores[idx,:]

    # select classifiers
    classifiers = list() # list of HaarLikeFeature objects

    if verbose:
        print("\nselecting classifiers ...")

    # visualise learning progress with text signals
    bar = progressbar.ProgressBar()

    # iterate all classifier (for t = 1, ..., T)
    for _ in bar(range(num_classifiers)):
        
        classification_errors = np.zeros(len(features)) # epsilon_j

        # 1.- normalize weights (w_t is a probability distribution) [weights of images]
        weights *= 1. / np.sum(weights)

        # 2.- for each feature, j, train a classifier hj (quite time consuming)
        pool = Pool(processes=NUM_PROCESS)
        updated_features = pool.map(partial(_update_feature, weights=weights , labels=labels), features)

        # compute the feature weighted error
        for f_idx, feature  in enumerate(updated_features):
            # f_idx = feature_index[f] #looks like f=f_idx :)
            # classifier error = sum of misclassified image weights
            error = sum(map(lambda img_idx: weights[img_idx] if labels[img_idx] != feature.get_vote(img_idx) else 0, range(num_imgs)))
            classification_errors[f_idx] = error
        
        # 3.- get the best feature (with the smallest error)
        min_error_idx = np.argmin(classification_errors) 
        lowest_error = classification_errors[min_error_idx]
        best_feature_idx = feature_index[min_error_idx]

        # set feature weight (alpha) and add to classifier list
        best_feature = updated_features[best_feature_idx]
        feature_weight = np.log((1 - lowest_error) / lowest_error) # alpha
        best_feature.weight = feature_weight
        classifiers.append(best_feature)

        def new_weights(lowest_error):
            return lowest_error / (1 - lowest_error) 

        # 4.- update image weights (w_(t+1) = w_t * beta_t ^ (1 - e_i)), where e_i = 1 when misclassified
        # map(func_to_apply, list_of_inputs) applies a function to all the items in an input_list
        weights_map = map(lambda img_idx: weights[img_idx] * new_weights(lowest_error) if labels[img_idx] != best_feature.get_vote(img_idx) else weights[img_idx] * 1, range(num_imgs))
        weights = np.array(list(weights_map))

        # remove feature (a feature cannot be selected twice)
        del features[best_feature_idx]

    if verbose:
        print("\nclassified selected ...\nreaching the end of AdaBoost algorithm ...")

    return classifiers