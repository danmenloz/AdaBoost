# Download faceScrub dataset
# Run from the project root folder as: python ./faceScrub download.py

import csv
import random
import subprocess
import shutil
from pathlib import Path
from PIL import Image
from random import randrange
from tqdm import tqdm

# faceScrub data files
actors_file = './faceScrub/facescrub_actors.txt'
actresses_file = './faceScrub/facescrub_actresses.txt'

actors_list = []
actresses_list = []

# Read actors' file
with open(actors_file, newline = '') as actors:                                                                                          
    actors_reader = csv.DictReader(actors, delimiter='\t')
    for actor in actors_reader:
        actors_list.append(actor)

# Read actresses' file
with open(actresses_file, newline = '') as actresses:                                                                                          
    actresses_reader = csv.DictReader(actresses, delimiter='\t')
    for actresses in actresses_reader:
        actresses_list.append(actor)

# Combine and shuffle combined list
actors_list.append(actresses_list)
random.shuffle(actors_list)

# Create source file
with open('./faceScrub/download.txt', 'w', newline='') as file:
    fieldnames = actors_list[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for actor in actors_list:
        try: 
            writer.writerow(actor)
        except:
            print(actor)

# Run fascescrub download tool
cmd = "python3 ./faceScrub/python3_download_facescrub.py ./faceScrub/download.txt ./actors/ \
    --crop_face --logfile=./faceScrub/download.log --timeout=5 --max_retries=3"
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True).wait()
