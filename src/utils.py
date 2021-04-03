import csv
import random
import shutil
import os
import numpy as np
from pathlib import Path
from PIL import Image
from random import randrange
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns

from src.haar_features import HaarLikeFeature as haar
from src.haar_features import featureType


path_actors_faces = './actors/faces/'
path_actors_images = './actors/images/'
path_faces_file = './actors/faces.txt'
path_save_features = './data/'
path_save_dir = './images/'



def random_crop(image_path, target_size):
        image = Image.open(image_path)

        if Path(image_path).suffix == '.png':
            image = image.convert('RGB')

        img_size = image.size
        x_max = img_size[0] - target_size[0]
        y_max = img_size[1] - target_size[1]

        random_x = randrange(0, x_max//2 + 1) * 2
        random_y = randrange(0, y_max//2 + 1) * 2

        area = (random_x, random_y, random_x + target_size[0], random_y + target_size[1])
        c_img = image.crop(area)

        return c_img



def create_directory(dataset, dataset_dir, resolution, verbose=1):
    if verbose:
        print('Creating {} directory...'.format(dataset_dir))
    Path(dataset_dir).mkdir(parents=True)
    (Path(dataset_dir) / '0').mkdir()
    (Path(dataset_dir) / '1').mkdir()
    for face in dataset:
        img1 = Image.open(face['face'])
        if verbose==2:
            print('Resizing {} '.format( str(Path(face['face']).name)) )
        if Path(face['face']).suffix != '.jpg' or Path(face['face']).suffix != '.jpeg':
            img1 = img1.convert('RGB')
        img1 = img1.convert('L') # convert to grayscale
        img1 = img1.resize(resolution)
        save_path = str(Path(dataset_dir) / '1' / Path(face['face']).stem ) +  '.png'
        img1.save( save_path, 'PNG')
        face['1'] = save_path
        img0 = random_crop(face['image'],resolution)
        img0 = img0.convert('L') # convert to grayscale
        save_path = str(Path(dataset_dir) / '0'/ Path(face['image']).stem ) +  '.png'
        img0.save( save_path, 'PNG')
        face['0'] = save_path



def create_info_file(dataset, dataset_dir, file_name, verbose=1):
    if verbose:
        print('Creating {} file...'.format(file_name))
    with open( str(Path(dataset_dir) / file_name), 'w', newline='') as file:
        fieldnames = dataset[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for face in dataset:
            writer.writerow(face)
    


def create_datasets(train_size, val_size, test_size, resolution, verbose=1):
    # Read download directory
    if verbose:
        print('\nReading actors folder...')

    faces = [] # this list contains all the download information

    # check if faces.txt file exists
    if Path(path_faces_file).is_file():
        with open(path_faces_file, newline='') as faces_file:
            faces_reader = csv.DictReader(faces_file, delimiter='\t')
            for face in faces_reader:
                faces.append(face)
    else:
        # it takes some minutes to scan the whole directory!
        for actor_entry in tqdm(Path(path_actors_faces).iterdir(), desc ="Reading faces"):
            if actor_entry.is_dir(): # read only directories
                for face_entry in actor_entry.iterdir():
                    faces.append( {'name':actor_entry.name, 'face':str(face_entry)} ) # add info to faces list

        for actor_entry in tqdm(Path(path_actors_images).iterdir(), desc ="Reading images"):
            if actor_entry.is_dir(): # read only directories
                for image_entry in actor_entry.iterdir():
                    # Search for dictionary and add key
                    for face in faces:
                        if face['face'].find(image_entry.stem)>=0: # stem -> name without sufix
                            face['image'] = str(image_entry)
        
        create_info_file(faces, str(Path(path_faces_file).parent), Path(path_faces_file).name) # create faces.txt file

    # Shuffle list
    random.shuffle(faces)

    # clear data dir
    if Path('data').is_dir():
        shutil.rmtree('data')

    if verbose:
        print('\nCreating test set...')
    # Create test set list
    test_set = []
    actors = [] # list to keep track which actors have been used
    for idx, face in enumerate(faces):
        # Examine image and dicard if not RGB, e.g, type L (b/w)
        img = Image.open(face['image'])
        if img.mode == 'RGB':
            test_set.append(face)
            actors.append(face['name'])
        if len(test_set) == test_size:
            break
    actors = list(set(actors)) # delete duplicates
    create_directory(test_set, 'data/test/', resolution, verbose=2)
    create_info_file(test_set, 'data/test/', 'test.txt')

    if verbose:
        print('\nCreating validation set...')
    # Create validation set list, make sure that no actor from test set is here
    validation_set = []
    for idx, face in enumerate(faces[idx+1:], start=idx+1): # continue reading from idx
        img = Image.open(face['image'])
        if face['name'] not in actors and img.mode == 'RGB':
            validation_set.append(face)
        if len(validation_set) == val_size:
            break
    create_directory(validation_set, 'data/valid/', resolution, verbose=2)
    create_info_file(validation_set, 'data/valid/', 'valid.txt')

    if verbose:
        print('\nCreating training set...')
    # Create training set list, make sure that no actor from test set is here
    training_set = []
    for face in faces[idx+1:]:
        img = Image.open(face['image'])
        if face['name'] not in actors and img.mode == 'RGB':
            training_set.append(face)
        if len(training_set) == train_size:
            break
    create_directory(training_set, 'data/train/', resolution, verbose=2)
    create_info_file(training_set, 'data/train/', 'train.txt')

    if verbose:
        print('\nDatasets created sucessfully!\n')

    return training_set, validation_set, test_set



def read_dataset_file(filename):
    actors = list()
    with open(filename, newline='') as dataset:
        dataset_reader = csv.DictReader(dataset, delimiter='\t')
        for actor in dataset_reader:
            actors.append(actor)
    return actors



def load_datasets():
    files = ['data/train/train.txt', 'data/valid/valid.txt', 'data/test/test.txt']
    return map(read_dataset_file, files)



def read_dataset(dataset):
    # dataset is a list of dictionaries: dataset = [{face: image: 1: 0:},...]
    # return list of positive and negative images

    pos_imgs = list()
    neg_imgs = list()

    for actor in dataset:
        pos_imgs.append( np.array(Image.open(actor['1']), dtype=np.float64) )
        neg_imgs.append( np.array(Image.open(actor['0']), dtype=np.float64) )
    
    return pos_imgs, neg_imgs



def save_features(filename, features):
    features_list = [] # list of dictionaries to hold the features
    for feature in features:
        d = {
            'type': feature.type.value,
            'position': feature.top_left,
            'width': feature.width,
            'height': feature.height,
            'error': feature.error,
            'threshold': feature.threshold,
            'polarity': feature.polarity,
            'weight': feature.weight
        }
        features_list.append(d)

    with open(path_save_features+filename, 'w', newline='') as file:
        fieldnames = features_list[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for feature in features_list:
            writer.writerow(feature)
    


def load_features(filename):
    # Read data
    read_features = list()
    with open(path_save_features+filename, newline='') as features:
        features_reader = csv.DictReader(features, delimiter='\t')
        for feature in features_reader:
            read_features.append(feature)

    created_features = [haar(featureType(eval(f['type'])), 
        eval(f['position']), 
        eval(f['width']), 
        eval(f['height']), 
        eval(f['error']), 
        eval(f['threshold']),
        eval(f['polarity']),
        eval(f['weight'])) for f in read_features]

    return created_features


def plot_features(features, name, res, combined=False):
    if combined: # all featuers on one image
        img = Image.new("RGBA", res, (255, 0, 0, 0))
        for feature in features:
            layer = feature.draw_feature(res=res)
            img.paste(layer, (0,0), layer)
        img.save( os.path.join(path_save_dir, name+'.png'), "PNG")
    else:
        for i, feature in enumerate(features):
            img = feature.draw_feature(res=res)
            img.save( os.path.join(path_save_dir, name+str(i)+'.png'), "PNG")


def ensemble_vote(int_img, classifiers, continous=False):
    if continous:
        # compute the vote for the final strong classifier in continous form [0->1]
        vote =  sum([f.weight * f.get_vote(int_img=int_img) for f in classifiers])
                # 0.5*sum([f.weight for f in classifiers]) 
    else:
        # compute the vote for the final strong classifier in discrete form [0,1]
        vote = 1 if sum([f.weight * f.get_vote(int_img=int_img) for f in classifiers]) \
                    >= 0.5*sum([f.weight for f in classifiers]) else 0
    return vote


def test_classifiers(pos_int_imgs, neg_int_imgs, classifiers, continous=False, type='strong'):
    # type: 'weak' or 'strong'
    imgs = pos_int_imgs + neg_int_imgs

    if type=='strong':
        score = [ensemble_vote(img,classifiers,continous) for img in imgs]
    elif type=='weak':
        score = [classifiers.get_vote(int_img=img) for img in imgs]
    
    return score


def plot_roc(pos_int_imgs, neg_int_imgs, classifiers, name):
    score = test_classifiers(pos_int_imgs, neg_int_imgs, classifiers, continous=True)
    labels = [1.0]*len(pos_int_imgs) + [0.0]*len(neg_int_imgs)

    fpr, tpr, threshold = metrics.roc_curve(labels, score)
    roc_auc = metrics.auc(fpr, tpr)

    plt.title('ROC:' + name)
    plt.plot(fpr, tpr, 'b', label='AUC = {:0.2f}'.format(roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(path_save_dir, name+'.jpg'))
    plt.close()


def plot_confusion_matrix(score, num_pos, num_neg, name):
    labels = [1.0]*num_pos + [0.0]*num_neg
    confM = metrics.confusion_matrix(labels, score)
    fig, ax = plt.subplots(figsize=(7,4)) 
    axis_labels = ["non-face","face"]
    sns.heatmap(confM, annot=True,cmap="Greens",fmt='g', xticklabels=axis_labels, yticklabels=axis_labels,ax=ax)
    plt.title('Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(path_save_dir, name+'.jpg'))
    plt.close()


def build_report(score, num_pos, num_neg):
    labels = [1.0]*num_pos + [0.0]*num_neg
    print('\nClassification Report')
    print(metrics.classification_report(labels, score))
