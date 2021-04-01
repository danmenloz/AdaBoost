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
    pos_valid_int_imgs = [ integral.IntegralImage(img).int_img for img in pos_valid_imgs ]
    neg_valid_int_imgs = [ integral.IntegralImage(img).int_img for img in neg_valid_imgs ]

    # parameters (do not change)
    num_classifier = 10
    min_feature_height = 2
    max_feature_height = 15
    min_feature_width = 2
    max_feature_width = 15
    
    # Find the top 10 features (before boosting)
    # Create features
    # img_height, img_width = pos_train_int_imgs[0].shape
    # features = ab._create_features(img_width, img_height, min_feature_width, max_feature_width, min_feature_height, max_feature_height)
    # print('Total features: %d' % len(features))
    # top10_features, idxs = ab.find_best_features(features, pos_train_int_imgs, neg_train_int_imgs, n=10)
    # utils.save_features('top10.txt',top10_features)

    # # generate and save image of classsifiers
    # saved_features = utils.load_features('top10.txt')
    # for i, feature in enumerate(saved_features):
    #     img = feature.draw_feature(res=resolution)
    #     img.save( 'images/'+str(i)+'.png', "PNG")
        # print('[%d]: %f, %f, %f\n' %(idxs[i], feature.error, feature.threshold, feature.polarity))
    
    print("\nAdaBoost begins ...")

    classifiers = ab.learn(pos_train_int_imgs, neg_train_int_imgs, num_classifier, 
        min_feature_width, max_feature_width, min_feature_height, max_feature_height, verbose=True)

    utils.save_features('classifiers.txt', classifiers)

    for i, feature in enumerate(classifiers):
        img = feature.draw_feature(res=resolution)
        img.save( 'images/'+'classifier'+str(i)+'.png', "PNG")
