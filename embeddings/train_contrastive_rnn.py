"""
Train a N-pair contrastive network.

Author: Christiaan Jacobs
Contact: christiaanjacobs97@gmail.com
Date: 2020
"""

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from scipy.spatial.distance import pdist
from random import randint
import random
# from pathlib import Path
import os
from os import path
from tqdm import tqdm
import sys
import pickle
import timeit
import datetime
import argparse

from dataset import GlobalPhone

sys.path.append(path.join("..", "..", "src", "speech_dtw", "utils"))

import samediff
from models import contrastive_rnn

from samplers import N_pair_sampler

sys.path.append(path.join(".."))
from link_mfcc import sixteen_languages

from utility_functions import save_options_dict
from utility_functions import save_checkpoint
from utility_functions import load_checkpoint
from utility_functions import save_record_dict
from utility_functions import load_record_dict
from utility_functions import model_folder
from utility_functions import save_best_model

from closs import contrastive_loss

default_options_dict = {
        'train_lang': 'SP',                 # language code
        'val_lang': 'SP',
        'train_tag': 'gt',                 # 'gt', 'utd', 'rnd'
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
        'n_epochs': 20,
        'batch_size': 600,
        'temperature': 0.1,

        'n_max_pairs': 100000,
        'n_min_tokens_per_type': None,         # if None, no filter is applied
        'n_max_tokens_per_type': None,
        'n_max_types': None,
        'n_minibatches': None,
        'rnd_seed': 1,
        'load_model': False, 
        'shuffle': False,
        'save_best': True

    }


# override collate func to return padded sequnce in decending order
def train_collate_fn(batch):
    # print(len(batch))
    # batch is list containing elements of (data, label) 
    # where data is shape (seq_lan, feat_dim) and label is integer
    
    # sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True) # sort elements according to seq_len from max to min
    sorted_batch = [(i[1], i[0]) for i in sorted(enumerate(batch[0][0]), key=lambda x: x[1].shape[0], reverse=True)]
    sequences_sorted = [x[0] for x in sorted_batch] # list containg variable length sequences
    sorted_idxs = [x[1] for x in sorted_batch]

    lengths = torch.LongTensor([len(x) for x in sequences_sorted]) # contains original length of sequences in batch
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences_sorted, batch_first=True) # pad variable length sequences to max seq length
    
    labels = [label for label in batch[0][1]]
    labels_sorted = [labels[i] for i in sorted_idxs]
    
    return sequences_padded, lengths, labels_sorted, sorted_idxs



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
        distances = pdist(np_z_normalised, metric='cosine')

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
    
    print(datetime.datetime.now())
    sampler = N_pair_sampler(dataset_train, options_dict)
    print(datetime.datetime.now())

    dataloader_train = DataLoader(dataset = dataset_train,
                            batch_size = 1,
                            sampler = sampler,
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
        model = contrastive_rnn(options_dict)
        model.to(device)
        
        pretrained_model_dir = path.join("models/RU+CZ+FR+PL+TH+PO.gt/8142ecc9a7/")
        # pretrained_model_dir = path.join('../train_siamese_rnn/models/RU+CZ+FR.gt/d79375bead/')

        model_dir_fn = path.join(pretrained_model_dir, "final_model.pt")

        state = torch.load(model_dir_fn)
        state_dict = state["state_dict"]
        # state_dict = torch.load(model_dir_fn)

        model.load_state_dict(state_dict, strict=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict["learning_rate"])
        # optimizer.load_state_dict(state['optimizer'])

        def freeze(model):
            for name, p in model.named_parameters():
                if "gru" in name:
                    print(name)
                    p.requires_grad = False
            return None

        # freeze(model)
    else:
        # Initialize model
        model = contrastive_rnn(options_dict)
        model.to(device)

        # Loss and optimization function
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict["learning_rate"])

    # Load model from checkpoint
    # if options_dict['resume'] == True:       
    #     start_epoch = load_checkpoint(model, optimizer) + 1
    #     print(start_epoch)
    # else:
    #     start_epoch = 0

    # Save options dictionary before training
    # if not options_dict['resume']:
    save_options_dict(options_dict)

    # Load record_dict or create new one
    record_dict_fn = path.join(options_dict["model_dir"], "record_dict.pkl")
    # if options_dict['resume']:
    #     record_dict = load_record_dict(record_dict_fn)
    # else:
    record_dict = {}
    record_dict["epoch_time"] = []
    record_dict["train_loss"] = []
    record_dict["validation_loss"] = []

    best_val_loss = np.inf
    for epoch in range(options_dict["n_epochs"]):
        start_time = timeit.default_timer()

        # Set model in training mode
        model.train()
        for i, (sequences_padded_train, seq_lengths_train, labels_train, sorted_idxs) in enumerate(dataloader_train):
            sequences_padded_train, seq_lengths_train = sequences_padded_train.to(device), seq_lengths_train.to(device)

            # Apply forward function
            embeddings_train = model(sequences_padded_train, seq_lengths_train)
            embeddings_train_resorted = torch.zeros_like(embeddings_train)
            embeddings_train_resorted[sorted_idxs] = embeddings_train

            loss = contrastive_loss(embeddings_train_resorted, options_dict["temperature"])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step() 
            
        end_time = timeit.default_timer()
        epoch_time = end_time - start_time

        print ('Epoch [{}/{}], Loss: {:.4f}, Time: {:.4f}' 
                .format(epoch+1, options_dict['n_epochs'], loss.item(), (epoch_time)))

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

        # record_dict['train_loss'].append((epoch, triplet_train_loss.item()))
        record_dict['epoch_time'].append((epoch, epoch_time))    
        record_dict['validation_loss'].append((epoch, val_loss))
        
        save_record_dict(record_dict, record_dict_fn)

         # if best modal thus far save mddel
        if options_dict["save_best"]:
            if current_val_loss < best_val_loss:
                save_best_model(model, optimizer, epoch , options_dict['model_dir'])
                best_val_loss = current_val_loss
        else:
            save_best_model(model, optimizer, epoch , options_dict['model_dir'])


        # Reconstruct minibatches
        print(datetime.datetime.now())
        if options_dict["shuffle"]:
            sampler.minibatch_reconstruction()
        print(datetime.datetime.now())


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
        "--n_max_tokens_per_type", type=int,
        help="maximum number of tokens per type (default: %(default)s)",
        default=default_options_dict["n_max_tokens_per_type"]
        )
    parser.add_argument(
        "--n_max_types", type=int,
        help="maximum number of types per language (default: %(default)s)",
        default=default_options_dict["n_max_types"]
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
        "--rnd_seed", type=int, help="random seed (default: %(default)s)",
        default=default_options_dict["rnd_seed"]
        )
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate (default: %(default)s)",
        default=default_options_dict["learning_rate"]
        )
    parser.add_argument(
        "--temperature", type=float, help="temperature parameter (default: %(defaults)s)",
        default=default_options_dict["temperature"]
        )
    parser.add_argument(
        "--n_minibatches", type=int, help="number of minibatches (default:, %(default)s)",
        default=default_options_dict["n_minibatches"]
        )
    parser.add_argument(
        "--load_model", type=bool, help="load model for adaptation (default:, %(default)s",
        default=default_options_dict["load_model"]
        )
    parser.add_argument(
        "--shuffle", type=bool, help="shuffle minibatches after each epoch (default:, %(default)s",
        default=default_options_dict["shuffle"]
        )
    parser.add_argument(
        "--save_best", type=bool, help="save best model after each epoch (default, %(default)s)",
        default=default_options_dict["save_best"]
        )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    
    # input arguments
    # resume = False
    # # options_dict = {}
    # if resume:
    #     ckpt_dir = 'models/SP.gt/train_siamese_rnn/train3' # get from arguments
    #     if path.isdir(ckpt_dir) and len(os.listdir(ckpt_dir)) != 0:
    #         options_dict_dir = path.join(ckpt_dir, 'options_dict.pkl')
    #         print('Reading:', options_dict_dir)
    #         with open(options_dict_dir, 'rb') as f:
    #             options_dict = pickle.load(f)
    #         options_dict['resume'] = True
    #     else:
    #         print('Path does not exist or does not contain recorded checkpoints. Good luck')
    #         sys.exit()
            
    # else:
    
    args = check_argv()

    # Set options
    options_dict = default_options_dict.copy()
    options_dict["script"] = "train_contrastive_rnn"
    options_dict["train_lang"] = args.train_lang
    options_dict["val_lang"] = args.val_lang
    options_dict["n_max_pairs"] = args.n_max_pairs
    options_dict["n_min_tokens_per_type"] = args.n_min_tokens_per_type
    options_dict["n_max_tokens_per_type"] = args.n_max_tokens_per_type
    options_dict["n_max_types"] = args.n_max_types
    options_dict["n_epochs"] = args.n_epochs
    options_dict["batch_size"] = args.batch_size
    options_dict["train_tag"] = args.train_tag
    options_dict["rnd_seed"] = args.rnd_seed
    options_dict["learning_rate"] = args.learning_rate
    options_dict["temperature"] = args.temperature
    options_dict["n_minibatches"] = args.n_minibatches
    options_dict["load_model"] = args.load_model
    options_dict["save_best"] = args.save_best    
    options_dict["shuffle"] = args.shuffle

    #Model directory
    model_dir = path.join('models', options_dict['train_lang'] + '.' + options_dict['train_tag'], "train_contrastive_rnn")
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_dir = path.join(model_dir, model_folder(options_dict))    
    print("Model directory: ", model_dir)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)
    options_dict['model_dir'] = model_dir

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print('Device:', device)  
    options_dict['device'] = device
    
    # Train RNN
    main(options_dict)
