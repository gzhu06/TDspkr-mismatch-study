import logging
import os
import re
from glob import glob
import json
import numpy as np
import random
import pickle

def array2pickle(data, filename):
    f = open(filename, 'wb')
    pickle.dump(data, f)
    f.close()

def pickle2array(filename):
    f = open(filename, 'rb')
    array = pickle.load(f)
    f.close()
    return array

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_last_checkpoint_if_any(checkpoint_folder):
    os.makedirs(checkpoint_folder, exist_ok=True)
    files = glob('{}/model*.pt'.format(checkpoint_folder), recursive=True)
    if len(files) == 0:
        return None
    return natural_sort(files)[-1]

def create_dir_and_delete_content(directory):
    os.makedirs(directory, exist_ok=True)
    files = sorted(filter(lambda f: os.path.isfile(f) and f.endswith("_ckpt.pt"), 
        map(lambda f: os.path.join(directory, f), os.listdir(directory))),
        key=os.path.getmtime)
    # delete all but most current file to assure the latest model is availabel even if process is killed
    for file in files[:-4]:

        logging.info("removing old model: {}".format(file))
        os.remove(file)
        
class Params():
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        self.update(json_path)

    def save(self, json_path):
        """Saves parameters to json file"""
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']`"""
        return self.__dict__