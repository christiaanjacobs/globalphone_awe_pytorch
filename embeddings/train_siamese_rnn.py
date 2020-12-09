"""
Train a Siamese triplets network.

Author: Christiaan Jacobs
Contact: christiaanjacobs97@gmail.com
Date: 2020
"""

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import  DataLoader


from scipy.spatial.distance import pdist
from random import randint
import random

import os
from os import path
from tqdm import tqdm
import sys
import pickle
import timeit
import argparse

from dataset import GlobalPhone
from models import siamese_rnn

sys.path.append(path.join("..", "..", "src", "speech_dtw", "utils"))
import samediff

import triplets
from samplers import SiameseSampler

from link_mfcc import sixteen_languages

from utility_functions import save_options_dict
from utility_functions import save_checkpoint
from utility_functions import load_checkpoint
from utility_functions import save_record_dict
from utility_functions import load_record_dict
from utility_functions import model_folder
from utility_functions import save_best_model


default_options_dict = {
        'train_lang': 'None',                 # language code
        'val_lang': 'None',
        'train_tag': 'utd',                 # 'gt', 'utd', 'rnd'
        'max_length': 100,
        'bidirectional': False,
        'rnn_type': 'gru',                  # 'lstm', 'gru', 'rnn'
        'hidden_size': 400,
        'num_layers': 3,
        'bias': True,
        'batch_first': True,
        'ff_n_hiddens': 130,              # embedding dimensionality
        'learning_rate': 0.001,
        'dropout': .0,
        'n_epochs': 25,
        'batch_size': 300,
        'margin': 0.25, 

        'n_max_pairs': None,
        'n_min_tokens_per_type': None,         # if None, no filter is applied
        'n_max_tokens_per_type': None,
        'n_max_types': None,
        'rnd_seed': 1,
        'distance_metric': 'cosine',
        'load_model': False,
        'adapt_model': "6 lingual",
        'save_best': True

    }

 


# override collate func to return padded sequnce in decending order
def train_collate_fn(batch):
    # print(len(batch))
    # batch is list containing elements of (data, label) 
    # where data is shape (seq_lan, feat_dim) and label is integer
    sorted_batch = sorted(batch[0], key=lambda x: x[0].shape[0], reverse=True) # sort elements according to seq_len from max to min
    sequences = [x[0] for x in sorted_batch] # list containg variable length sequences
    lengths = torch.LongTensor([len(x) for x in sequences]) # contains original length of sequences in batch
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True) # pad variable length sequences to max seq length
    labels = torch.LongTensor([x[1] for x in sorted_batch]) # contains label of each sequence at same index     
    return sequences_padded, lengths, labels

# return self.x[idx], self.labels[idx], self.lengths[idx], self.keys[idx], self.speakers[idx]
def val_collate_fn(batch):
    # batch is list containing elements of (data, label, length, keys, speakers) 
    # where data is shape (seq_len, feat_dim) and label is integer
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True) # sort elements according to seq_len from max to min
    sequences = [x[0] for x in sorted_batch] # list containg variable length sequences
    lengths = torch.LongTensor([len(x) for x in sequences]) # contains original length of sequences in batch
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True) # pad variable length sequences to max seq length
    labels = [x[1] for x in sorted_batch] # contains label of each sequence at same index
    keys =  [x[3] for x in sorted_batch]
    speakers = [x[4] for x in sorted_batch]    
    return sequences_padded, lengths, labels, keys, speakers


# Validation function
def same_diff(np_z, labels_val, speakers_val, keys_val, normalise=True):        
    embed_dict = {}
    for i, utt_key in enumerate(keys_val):
        embed_dict[utt_key] = np_z[i]
    
    if normalise: 
        np_z_normalised = (np_z - np_z.mean(axis=0))/(np_z.std(axis=0))
        distances = pdist(np_z_normalised, metric="cosine")

    word_matches = samediff.generate_matches_array(labels_val)
    speaker_matches = samediff.generate_matches_array(speakers_val)
    sw_ap, sw_prb, swdp_ap, swdp_prb = samediff.average_precision_swdp(
        distances[np.logical_and(word_matches, speaker_matches)],
        distances[np.logical_and(word_matches, speaker_matches == False)],
        distances[word_matches == False]
        )
    # return [sw_prb, -sw_ap, swdp_prb, -swdp_ap]
    return [swdp_prb, -swdp_ap]


def main(options_dict):
    # Random seeds
    random.seed(options_dict["rnd_seed"])
    np.random.seed(options_dict["rnd_seed"])
    torch.manual_seed(options_dict["rnd_seed"])

    # Set operating device
    device = options_dict["device"]

    # Load training data
    dataset_train = GlobalPhone("train", options_dict)

    siamese_sampler = SiameseSampler(dataset_train, options_dict)
    # siame_batch_sampler = torch.utils.data.BatchSampler(siamese_sampler, options_dict['batch_size'], drop_last=True)

    dataloader_train = DataLoader(dataset = dataset_train,
                            batch_size = 1,
                            sampler = siamese_sampler,
                            batch_sampler = None, 
                            shuffle = False,
                            drop_last = False,
                            collate_fn = train_collate_fn)

    # Load validation data
    dataset_val = GlobalPhone("val", options_dict)
    
    dataloader_val = DataLoader(dataset = dataset_val,
                        batch_size = dataset_val.__len__(), 
                        shuffle = False,
                        drop_last = True,
                        collate_fn = val_collate_fn)

     
    if options_dict["load_model"]:
        
        # Initialize model
        model = siamese_rnn(options_dict)
        model.to(device)
        
        pretrained_model_dir = path.join("../train_siamese_rnn/models/RU+CZ+FR+PL+TH+PO.gt/95648ee1f0/")

        model_dir_fn = path.join(pretrained_model_dir, "final_model.pt")

        state = torch.load(model_dir_fn)
        state_dict = state["state_dict"]

        model.load_state_dict(state_dict, strict=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict["learning_rate"])
        # optimizer.load_state_dict(state['optimizer'])

        def freeze(model):
            for name, p in model.named_parameters():
                if "encoder.gru" in name:
                    print(name)
                    p.requires_grad = False
            return None

        # freeze(model)
    else:
        # Initialize model
        model = siamese_rnn(options_dict)
        model.to(device)

        # Loss and optimization function
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict["learning_rate"])

    # Load model from checkpoint
#    if options_dict["resume"] == True:       
#        start_epoch = load_checkpoint(model, optimizer) + 1
#        print(start_epoch)
#    else:
#        start_epoch = 0

    # Save options dictionary before training
#    if not options_dict["resume"]:
    save_options_dict(options_dict)

   # Load record_dict
   # Load record_dict or create new one
    record_dict_fn = path.join(options_dict["model_dir"], "record_dict.pkl")
    
#    if options_dict["resume"]:
#        record_dict = load_record_dict(record_dict_fn)
#    else:
    record_dict = {}
    record_dict["epoch_time"] = []
    record_dict["train_loss"] = []
    record_dict["validation_loss"] = []

    best_val_loss = np.inf
    for epoch in range(options_dict["n_epochs"]):
        start_time = timeit.default_timer()
        # Set model in training mode
        model.train()
        for i, (sequences_padded_train, seq_lengths_train, labels_train) in enumerate(dataloader_train):
            sequences_padded_train, seq_lengths_train, labels_train = sequences_padded_train.to(device), seq_lengths_train.to(device), labels_train.to(device)

            # Apply forward function
            embeddings_train = model(sequences_padded_train, seq_lengths_train)

            # Loss
            triplet_train_loss = triplets.batch_hard_triplet_loss(labels_train, embeddings_train, options_dict["margin"], squared=False, device=device)
            # triplet_train_loss, _ = triplets.batch_all_triplet_loss(labels_train, embeddings_train, options_dict['margin'], squared=False)


            # Backward and optimize
            optimizer.zero_grad()
            triplet_train_loss.to(device).backward()
            # triplet_train_loss.backward()
            optimizer.step() 
            
        end_time = timeit.default_timer()
        epoch_time = end_time - start_time

        print ("Epoch [{}/{}], Loss: {:.4f}, Time: {:.4f}" 
                .format(epoch+1, options_dict["n_epochs"], triplet_train_loss.item(), (epoch_time)))

        # Set model in eval mode
        model.eval()
        with torch.no_grad():
            for i, (sequences_padded_val, lengths_val, labels_val, keys_val, speakers_val) in enumerate(dataloader_val):
                sequences_padded_val, lengths = sequences_padded_val.to(device), lengths_val.to(device)
                
                np_z = model(sequences_padded_val, lengths_val).cpu().detach().numpy()
                
                break # single batch
                
            # Do same_diff eval
            val_loss = same_diff(np_z, labels_val, speakers_val, keys_val)
            current_val_loss = val_loss[-1]
            print(val_loss)

        record_dict["train_loss"].append((epoch, triplet_train_loss.item()))
        record_dict["epoch_time"].append((epoch, epoch_time))    
        record_dict["validation_loss"].append((epoch, val_loss))
        
        save_record_dict(record_dict, record_dict_fn)

        if options_dict["save_best"]:
            # if best modal thus far save mddel
            if current_val_loss < best_val_loss:
                save_best_model(model, optimizer, epoch , options_dict["model_dir"])
                best_val_loss = current_val_loss
        else:
            save_best_model(model, optimizer, epoch , options_dict["model_dir"])


    


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "train_lang", type=str,
        help="GlobalPhone training language {BG, CH, CR, CZ, FR, GE, HA, KO, "
        "PL, PO, RU, SP, SW, TH, TU, VN}",
        )
    parser.add_argument(
        "--val_lang", type=str, help="validation language",
        choices=sixteen_languages, default=None
        )
    parser.add_argument(
        "--n_max_pairs", type=int,
        help="maximum number of same-word pairs to use (default: %(default)s)",
        default=default_options_dict["n_max_pairs"]
        )
    parser.add_argument(
        "--n_min_tokens_per_type", type=int,
        help="minimum number of tokens per type (default: %(default)s)",
        default=default_options_dict["n_min_tokens_per_type"]
        )
    parser.add_argument(
        "--n_max_types", type=int,
        help="maximum number of types per language (default: %(default)s)",
        default=default_options_dict["n_max_types"]
        )
    parser.add_argument(
        "--n_max_tokens_per_type", type=int,
        help="maximum number of tokens per type (default: %(default)s)",
        default=default_options_dict["n_max_tokens_per_type"]
        )
    parser.add_argument(
        "--n_epochs", type=int,
        help="number of epochs of training (default: %(default)s)",
        default=default_options_dict["n_epochs"]
        )
    parser.add_argument(
        "--batch_size", type=int,
        help="size of mini-batch (default: %(default)s)",
        default=default_options_dict["batch_size"]
        )
    parser.add_argument(
        "--train_tag", type=str, choices=["gt", "utd"],
        help="training set tag (default: %(default)s)",
        default=default_options_dict["train_tag"]
        )
    parser.add_argument(
        "--margin", type=float,
        help="margin for contrastive loss (default: %(default)s)",
        default=default_options_dict["margin"]
        )
    parser.add_argument(
        "--rnd_seed", type=int, help="random seed (default: %(default)s)",
        default=default_options_dict["rnd_seed"]
        )
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate(default: %(default)s)",
        default=default_options_dict["learning_rate"]
        )

    parser.add_argument(
        "--save_best", type=bool, help="save best model after epoch(default: %(default)s)",
        default=default_options_dict["save_best"]
        )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    
    args = check_argv()

    # Set options
    options_dict = default_options_dict.copy()
    options_dict["script"] = "train_siamese_rnn"
    options_dict["train_lang"] = args.train_lang
    options_dict["val_lang"] = args.val_lang
    options_dict["n_max_pairs"] = args.n_max_pairs
    options_dict["n_min_tokens_per_type"] = args.n_min_tokens_per_type
    options_dict["n_max_types"] = args.n_max_types
    options_dict["n_epochs"] = args.n_epochs
    options_dict["batch_size"] = args.batch_size
    options_dict["train_tag"] = args.train_tag
    options_dict["margin"] = args.margin
    options_dict["rnd_seed"] = args.rnd_seed
    options_dict["learning_rate"] = args.learning_rate
    options_dict["save_best"] = args.save_best
    
#    print(options_dict['learning_rate'])
#    print(options_dict["save_best"])
    
#Model directory
    model_dir = path.join('models', options_dict['train_lang'] + '.' + options_dict['train_tag'], "train_siamese_rnn")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_dir = path.join(model_dir, model_folder(options_dict))    
    print("Model directory: ", model_dir)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    options_dict['model_dir'] = model_dir
#    options_dict['resume'] = False   

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print('Device:', device)  
    options_dict['device'] = device
    
    # Train RNN
    main(options_dict)
