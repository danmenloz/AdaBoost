import src.utils as utils
import src.integral_image as integral
import src.adaboost as ab
import src.haar_features as haar

# Training and test dataset sizes
train_size = 1000
val_size = 100
test_size = 100

# Image resolution
resolution = (20,20)

if __name__ == "__main__":
    # run first 'python ./faceScrub download.py' to generate the actors folder
    
    # Create datasets with equal number of pos and neg classes
    ### The next line creates new datasets with randomly selected images from the actors/ folder
    ### Uncomment to create new datasets or alternatively read the existing datasets
    # train_set, valid_set, test_set = utils.create_datasets(train_size,val_size,test_size,resolution)
    train_set, valid_set, test_set = utils.load_datasets() # read txt files
    # set = [{face: image: 1: 0:},...]

    # Read images from datasets
    pos_train_imgs, neg_train_imgs = utils.read_dataset(train_set)
    pos_valid_imgs, neg_valid_imgs = utils.read_dataset(valid_set)
    pos_test_imgs, neg_test_imgs = utils.read_dataset(test_set)

    # Samples count
    num_train_pos, num_train_neg = len(pos_train_imgs), len(neg_train_imgs)
    num_valid_pos, num_valid_neg = len(pos_valid_imgs), len(neg_valid_imgs)
    num_test_pos, num_test_neg = len(pos_test_imgs), len(neg_test_imgs)
    print('Faces count:     +train=%d  +valid=%d    +test=%d' % (num_train_pos, num_valid_pos, num_test_pos))
    print('Non-faces count: -train=%d  -valid=%d    -test=%d' % (num_train_neg, num_valid_neg, num_test_neg))

    # Integral images
    pos_train_int_imgs = [ integral.IntegralImage(img).int_img for img in pos_train_imgs ]
    neg_train_int_imgs = [ integral.IntegralImage(img).int_img for img in neg_train_imgs ]
    pos_test_int_imgs = [ integral.IntegralImage(img).int_img for img in pos_test_imgs ]
    neg_test_int_imgs = [ integral.IntegralImage(img).int_img for img in neg_test_imgs ]

    # parameters (do not change)
    num_classifier = 10
    min_feature_height = 2
    max_feature_height = 15
    min_feature_width = 2
    max_feature_width = 15
    
    print('\nFinding the top 10 features (before boosting) ...')

    # Create features
    img_height, img_width = pos_train_int_imgs[0].shape
    features = ab._create_features(img_width, img_height, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    print('Total features: %d' % len(features))

    ### Next line executes algorithm to train and find the best 10 features but
    ### it is quite time consuming... it took 3 hrs to train all features
    ### load the found features instead
    # top10_features, idxs = ab.find_best_features(features, pos_train_int_imgs, neg_train_int_imgs, n=10)
    # utils.save_features('top10.txt',top10_features)

    top10_features = utils.load_features('top10.txt')
    utils.plot_features(top10_features,'feature_', resolution)
    utils.plot_features(top10_features,'features_mix', resolution, combined=True)
    utils.plot_features(top10_features,'feature_face_', resolution, face=True)
    best_feature = top10_features[0]

    print('\nBest Feature Training Results:')
    score = utils.test_classifiers(pos_train_int_imgs, neg_train_int_imgs, best_feature, type='weak')
    utils.plot_confusion_matrix(score, len(pos_train_int_imgs), len(neg_train_int_imgs), 'best_train_confmatx')
    utils.build_report(score, len(pos_train_int_imgs), len(neg_train_int_imgs))

    print('\nBest Feature Test Results:')
    score = utils.test_classifiers(pos_test_int_imgs, neg_test_int_imgs, best_feature, type='weak')
    utils.plot_confusion_matrix(score, len(pos_test_int_imgs), len(neg_test_int_imgs), 'best_test_confmatx')
    utils.build_report(score, len(pos_test_int_imgs), len(neg_test_int_imgs))


    print("\nAdaBoost begins ...")

    ### Next line executes the AdaBoost algorithm but it is quite time consuming... 
    ### it took 1 day 8hrs to find 10 clasifiers! Load the found classifiers instead
    # classifiers = ab.learn(pos_train_int_imgs, neg_train_int_imgs, num_classifier, 
    #     min_feature_width, max_feature_width, min_feature_height, max_feature_height, verbose=True)
    # utils.save_features('classifiers.txt', classifiers)

    classifiers = utils.load_features('classifiers.txt')
    utils.plot_features(classifiers, 'classifier_', resolution)
    utils.plot_features(classifiers, 'classifiers_mix', resolution,combined=True, face=True)

    print('\nAdaBoost Training Results:')
    score = utils.test_classifiers(pos_train_int_imgs, neg_train_int_imgs, classifiers)
    utils.plot_confusion_matrix(score, len(pos_train_int_imgs), len(neg_train_int_imgs), 'AdaBoost_train_confmatx')
    utils.plot_roc(pos_train_int_imgs, neg_train_int_imgs, classifiers, 'AdaBoost_train_roc')
    utils.build_report(score, len(pos_train_int_imgs), len(neg_train_int_imgs))

    print('\nAdaBoost Test Results:')
    score = utils.test_classifiers(pos_test_int_imgs, neg_test_int_imgs, classifiers)
    utils.plot_confusion_matrix(score, len(pos_test_int_imgs), len(neg_test_int_imgs), 'AdaBoost_test_confmatx')
    utils.plot_roc(pos_test_int_imgs, neg_test_int_imgs, classifiers, 'AdaBoost_test_roc')
    utils.build_report(score, len(pos_test_int_imgs), len(neg_test_int_imgs))
