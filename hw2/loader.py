import os
import json

import numpy as np

class Loader(object):
    def __init__(self):
        pass
    
    def read_data(self, folder, id_to_captions):
        x = []
        x_label = []
        for filename in os.listdir(folder):
            viedo_id = filename.split('.npy')[0]
            full_path = folder + filename
            feature = np.load(full_path)
            x.append(feature)
            x_label.append(id_to_captions[viedo_id])
        x = np.array(x, dtype='float32')
        return x, x_label
    
    def read_test_data(self, id_file, folder):
        x = []
        id_list = []
        with open(id_file, 'r') as file:
            for line in file:
                viedo_id = line.strip()
                full_path = folder + viedo_id + '.npy'
                feature = np.load(full_path)
                x.append(feature)
                id_list.append(viedo_id)
            x = np.array(x, dtype='float32')
        return x, id_list
                
    def read_captions(self, file_name):
        id_to_captions = {}
        with open(file_name, 'r') as file:
            data = json.load(file)
            for entry in data:
                viedo_id = entry['id']
                captions = entry['caption']
                captions = map(lambda x: x.replace('.', ''), captions)
                captions = map(lambda x: x.replace(',', ''), captions)
                captions = map(lambda x: x.replace('"', ''), captions)
                captions = map(lambda x: x.replace('\n', ''), captions)
                captions = map(lambda x: x.replace('?', ''), captions)
                captions = map(lambda x: x.replace('!', ''), captions)
                captions = map(lambda x: x.replace('\\', ''), captions)
                captions = map(lambda x: x.replace('/', ''), captions)
                id_to_captions[viedo_id] = list(captions)

        return id_to_captions