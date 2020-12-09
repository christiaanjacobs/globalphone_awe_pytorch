"""
Load GlobalPhone dataset.

Author: Christiaan Jacobs
Contact: christiaanjacobs97@gmail.com
Date: 2020
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint
from os import path
from tqdm import tqdm

import data_io



class GlobalPhone(Dataset):
    def __init__(self, mode, options_dict):        
        self.script = options_dict["script"]        
        self.mode = mode
        data_root = "data"

        if self.script == "train_ae_cae_rnn":
            self.model_type = options_dict['model_type']

        data_dir = []
        if self.mode == "train":
            tag = options_dict["train_tag"]

            if "+" in options_dict["train_lang"]:
                train_languages = options_dict["train_lang"].split("+")
                for train_lang in train_languages:
                    cur_dir = path.join(data_root, train_lang, "train." + tag + ".npz")
                    data_dir.append(cur_dir)       
            else:
                train_lang = options_dict["train_lang"]
                data_dir.append(path.join(data_root, train_lang, "train." + tag + ".npz"))


        if self.mode == "val":
            val_lang = options_dict["val_lang"]
            data_dir.append(path.join(data_root, val_lang, "val.npz"))
            

        self.x = []
        self.labels = []
        self.lengths = []
        self.keys = []
        self.speakers = []
     
        for i, cur_dir in enumerate(data_dir):
            cur_x, cur_labels, cur_lengths, cur_keys, cur_speakers = data_io.load_data_from_npz(cur_dir)
            if self.mode == "train":
                cur_x, cur_labels, cur_lengths, cur_keys, cur_speakers = data_io.filter_data(cur_x, cur_labels, cur_lengths, cur_keys, cur_speakers,
                                                                            n_min_tokens_per_type=options_dict["n_min_tokens_per_type"],
                                                                            n_max_types=options_dict["n_max_types"],
                                                                            n_max_tokens_per_type=options_dict["n_max_tokens_per_type"],
                                                                            n_max_tokens=options_dict["n_max_tokens"])
                                                            
        
            self.x.extend(cur_x)
            self.labels.extend(cur_labels)
            self.lengths.extend(cur_lengths)
            self.keys.extend(cur_keys)
            self.speakers.extend(cur_speakers) # list ['GE034', 'GE045', ..., 'GE12', RU087', 'RU012', 'RU012', ..., 'RU020']
        

        # Convert labels to int for train classifier rnn
        if self.mode == "train":
            self.labels_int = self.convert_labels(options_dict)
        
        self.trunc_and_limit(options_dict)


    def convert_labels(self, options_dict):
        # Convert training labels to integers
        train_label_set = list(set(self.labels))
        label_to_id = {}
        for i, label in enumerate(sorted(train_label_set)):
            label_to_id[label] = i
        train_y = []
        for label in self.labels:
            train_y.append(label_to_id[label])
        train_y = np.array(train_y, dtype=np.int32)
        print(train_y)
        options_dict["n_classes"] = len(label_to_id)
        print("Total no. classes:", options_dict["n_classes"])
        return train_y

    def trunc_and_limit(self, options_dict):
        max_length = options_dict["max_length"]
        d_frame = 13  # None
        options_dict["n_input"] = d_frame
        print("Limiting dimensionality:", d_frame)
        print("Limiting length:", max_length)
        data_io.trunc_and_limit_dim(self.x, self.lengths, d_frame, max_length)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        """
        idx is either int or tuple (x, y) containing word pairs x and y 
        """
        if self.mode == "train":
            if self.script == "train_siamese_rnn":
                list_train = []
                for i in idx:
                    list_train.append((self.x[i], self.labels_int[i]))
                return list_train
            if self.script == "train_contrastive_rnn":
                idx_list = [] 
                for pair in idx:
                    idx_list.extend(list(pair))
                x = []
                y = []
                for i in idx_list:
                    x.append(self.x[i])
                    y.append(self.labels_int[i])
                return x, y
            if self.script == "train_ae_cae_rnn":
                if self.model_type == 'train_ae_rnn':
                    return self.x[idx]
                if self.model_type == 'train_cae_rnn':
                    return self.x[idx[0]], self.x[idx[1]]
    

        if self.mode == "val":
            return self.x[idx], self.labels[idx], self.lengths[idx], self.keys[idx], self.speakers[idx]

