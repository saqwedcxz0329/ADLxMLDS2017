import os
import json

import numpy as np

class Loader(object):
    def __init__(self):
        pass
    
    def read_data(self, folder, id_to_captions=None):
        x = []
        x_label = []
        for filename in os.listdir(folder):
            viedo_id = filename.split('.npy')[0]
            full_path = folder + filename
            feature = np.load(full_path)
            x.append(feature)
            if id_to_captions is not None:
                x_label.append(id_to_captions[viedo_id])
        x = np.array(x, dtype='float32')
        if id_to_captions is not None:
            return x, x_label
        return x


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