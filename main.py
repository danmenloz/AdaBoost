import src.utils as utils
import src.integral_image as integral

# Training and test dataset sizes
train_size = 10
val_size = 5
test_size = 5

# Image resolution
resolution = (20,20)

if __name__ == "__main__":
    # run first 'python ./faceScrub download.py' to generate the actors folder
    
    # Create datasets with equal number of pos and neg classes
    train_set, valid_set, test_set = utils.create_datasets(train_size, val_size, test_size, resolution)
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

    