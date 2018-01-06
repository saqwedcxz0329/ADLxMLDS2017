import csv
import os

import skimage
import skimage.io
import skimage.transform
from scipy import misc
import numpy as np
import scipy.stats as stats


UNK = '<unk>'
HAIR = 'hair'
EYES = 'eyes'

hair_color = [UNK, 'orange', 'white', 'aqua', 'gray',
'green', 'red', 'purple', 'pink',
'blue', 'black', 'brown', 'blonde']

eyes_color = [UNK, 'gray', 'black', 'orange',
'pink', 'yellow', 'aqua', 'purple',
'green', 'brown', 'red', 'blue']

class Data(object):
    def __init__(self, img_feat, tag_feat, test_tag_feat, z_dim):
        self.eyes_color = eyes_color
        self.hair_color = hair_color
        
        # train data
        self.img_feat = img_feat
        self.tag_feat = tag_feat
        if img_feat is not None and tag_feat is not None:
            self.data_size = len(img_feat)
            self.wrong_idx = np.random.permutation(np.arange(self.data_size))
            self.tag_one_hot = []
            self.gen_info()

        # noise
        self.z_sampler = stats.truncnorm((-1 - 0.) / 1., (1 - 0.) / 1., loc=0., scale=1)

        # batch control
        self.current = 0
        self.epoch = 0

        # test data
        self.test_tag_one_hot = self.gen_test_hot(test_tag_feat)
        self.fixed_z = self.fixed_seed(len(self.test_tag_one_hot), z_dim)

    def gen_test_hot(self, test_intput):
        test_hot = []
        for tags in test_intput:
            eyes_hot = np.zeros([len(self.eyes_color)])
            eyes_hot[self.eyes_color.index(tags[0])] = 1
            hair_hot = np.zeros([len(self.hair_color)])
            hair_hot[self.hair_color.index(tags[1])] = 1
            tag_vec = np.concatenate((eyes_hot, hair_hot))
            test_hot.append(tag_vec)
        return np.array(test_hot)

    def gen_info(self):
        for tags in self.tag_feat:
            eyes_hot = np.zeros([len(self.eyes_color)])
            eyes_hot[self.eyes_color.index(tags[0])] = 1
            hair_hot = np.zeros([len(self.hair_color)])
            hair_hot[self.hair_color.index(tags[1])] = 1
            tag_vec = np.concatenate((eyes_hot, hair_hot))
            self.tag_one_hot.append(tag_vec)
        self.tag_one_hot = np.array(self.tag_one_hot)

    def next_data_batch(self, size):
        if self.current == 0:
            self.epoch += 1
            idx = np.random.permutation(np.arange(self.data_size))
            self.img_feat = self.img_feat[idx]
            self.tag_one_hot = self.tag_one_hot[idx]
            self.wrong_idx = np.random.permutation(np.arange(self.data_size))

        if self.current + size < self.data_size:
            img, tag_one, widx = \
                self.img_feat[self.current:self.current+size], \
                self.tag_one_hot[self.current:self.current+size], \
                self.wrong_idx[self.current:self.current+size]
            self.current += size

        else:
            img, tag_one, widx = \
                self.img_feat[self.current:], \
                self.tag_one_hot[self.current:], \
                self.wrong_idx[self.current:]
            self.current = 0

        return img, tag_one, self.img_feat[widx], self.tag_one_hot[widx]

    def next_noise_batch(self, size, dim):
        return np.random.normal(0.0, 1.0, [size, dim])

    def fixed_seed(sefl, size, dim):
        np.random.seed(0)
        return np.random.normal(0.0, 1.0, [size, dim])

def load_train_data(train_dir, tag_path):
    tag_feat = []
    img_feat = []
    with open(tag_path, 'r') as f:
        for idx, row in enumerate (csv.reader(f)):
            tags = row[1].split('\t')
            text_content = { HAIR: UNK, EYES: UNK }
            for t in tags:
                tag = t.split(':')[0].strip()
                tmp = tag.split()
                tag_length = len(tmp)
                if tag_length > 1:
                    if tmp[1] == HAIR:
                        color = tmp[0]
                        if color in hair_color:
                            text_content[HAIR] = color
                    if tmp[1] == EYES:
                        color = tmp[0]
                        if color in eyes_color:
                            text_content[EYES] = color
            
            if text_content[HAIR] != UNK or text_content[EYES] != UNK:

                row_idx = row[0]
                img_path = os.path.join(train_dir,'{}.jpg'.format(row_idx))
                img = misc.imread(img_path)
                # img = misc.imresize(img, [64, 64, 3])
                tag_feat.append([text_content[EYES], text_content[HAIR]])
                img_feat.append(img)

                '''
                m_img = np.fliplr(img)
                tag_feat.append([text_content[EYES], text_content[HAIR]])
                img_feat.append(m_img)

                img_p5 = misc.imrotate(img, 5)
                tag_feat.append([text_content[EYES], text_content[HAIR]])
                img_feat.append(img_p5)

                img_n5 = misc.imrotate(img, -5)
                tag_feat.append([text_content[EYES], text_content[HAIR]])
                img_feat.append(img_n5)
                '''
                
    img_feat = np.array(img_feat, dtype='float32')/127.5 - 1.
    return img_feat, tag_feat

def load_test(test_path):
    test_tag_feat = []
    with open(test_path, 'r') as f:
        for line in f.readlines():
            text_content = { HAIR: UNK, EYES: UNK }
            line = line.strip().split(',')[1]
            tags = line.split()
            for idx in range(0, len(tags), 2):
                color = tags[idx]
                if tags[idx+1] == EYES:
                    text_content[EYES] = color
                elif tags[idx+1] == HAIR:
                    text_content[HAIR] = color
            test_tag_feat.append([text_content[EYES], text_content[HAIR]])
    return  test_tag_feat

def dump_img(img_dir, img_feats, iters):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)
    resize_img_feats = []
    for img in img_feats:
        resize_img_feats.append(misc.imresize(img, [64, 64, 3]))
    
    path = os.path.join(img_dir, 'iters_{}_sample.jpg'.format(iters))
    
    resize_img_feats = resize_img_feats[:9]
    row = 3
    col = 3
    final_img = None
    for i in range(row):
        row_img = None
        for j in range(col):
            idx = i * row + j
            img = resize_img_feats[idx]
            if row_img is None:
                row_img = img
            else:
                row_img = np.concatenate((row_img, img), axis=1)
        if final_img is None:
            final_img = row_img
        else:
            final_img = np.concatenate((final_img, row_img), axis=0)

    misc.imsave(path, final_img)
    
def dump_test_img(img_dir, img_feats, name):
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_feats = (img_feats + 1.)/2 * 255.
    img_feats = np.array(img_feats, dtype=np.uint8)
    #img_feats = misc.imresize(img_feats, [64, 64, 3])
    
    path = os.path.join(img_dir, '{}_sample.jpg'.format(name))

    final_img = None
    for i in range(img_feats.shape[0]):
        col_img = None
        for img in img_feats[i]:
            col_img = img if col_img is None else np.concatenate((col_img, img), axis=0)
        
        final_img = col_img if final_img is None else np.concatenate((final_img, col_img), axis=1)
    
    misc.imsave(path, final_img)
