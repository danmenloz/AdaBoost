import src.utils as utils

# Training and test dataset sizes
train_size = 10
val_size = 5
test_size = 5

# Image resolution
resolution = (20,20)

if __name__ == "__main__":
    # run first 'python ./faceScrub download.py' to generate the actors folder
    actors_dir = './actors'
    train_set, valid_set, test_set = utils.create_datasets(train_size, val_size, test_size, resolution)