"""
Train a recurrent correspondence autoencoder.

Author: Christiaan Jacobs
Contact: christiaanjacobs97@gmail.com
Date: 2020
"""

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import  DataLoader

from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from scipy.spatial.distance import pdist
import random
from random import randint

import os
from os import path

from tqdm import tqdm
import sys
import pickle
import timeit
import argparse

from dataset import GlobalPhone
from models import ae_rnn, cae_rnn
import samplers

# utility functions
from utility_functions import save_options_dict
from utility_functions import save_checkpoint
from utility_functions import load_checkpoint
from utility_functions import save_record_dict
from utility_functions import load_record_dict
from utility_functions import model_folder
from utility_functions import save_best_model

sys.path.append(path.join('..' , '..', 'src', 'speech_dtw', 'utils'))
import samediff

# sys.path.append(path.join(".."))
from link_mfcc import sixteen_languages

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
        'ae_learning_rate': 0.001,
        'cae_learning_rate': 0.001,
        'dropout': .0,
        
        'ae_n_epochs': 25,
        'cae_n_epochs': 15,

        'ae_batch_size': 300,
        'cae_batch_size': 300,

        'n_max_pairs': None,                             
        'n_min_tokens_per_type': None,         
        'n_max_types': None,
        'n_max_tokens': None,
        'n_max_tokens_per_type': None,
        'rnd_seed': 1, 

        'pretrain': False,
        'load_model': False
    }

def ae_train_collate_fn(batch):
    # where batch is shape (seq_len, feat_dim) and label is integer
    sorted_batch = sorted(batch, key=lambda x: x.shape[0], reverse=True) # sort elements according to seq_len from max to min
    lengths = torch.FloatTensor([len(x) for x in sorted_batch]) # contains original length of sequences in batch
    sequences_padded = pad_sequence(sorted_batch, batch_first=True) # pad variable length sequences to max seq length
    return sequences_padded, lengths


def cae_train_collate_fn(batch):
    """
    batch is list containing elements of (data, data) where data is (seq_length, n_dim)
    where data is shape (seq_lan, feat_dim) and label is integer
    """
    sorted_input_data = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)  # sort according to input length
   
    input_sequences = [x[0] for x in sorted_input_data]  # list containg variable length input sequences decending order  
    input_seq_lengths = torch.LongTensor([len(x) for x in input_sequences])  # list contains original length of input sequences in batch
    
    corr_sequences = [x[1] for x in sorted_input_data] # list containg variable length correspondence sequences
    corr_seq_lengths = torch.LongTensor([len(x) for x in corr_sequences]) # list contains original length of correspondence sequences in batch
    
    
    input_sequences_padded = pad_sequence(input_sequences, batch_first=True) # pad variable length input sequences to max seq length
    corr_sequences_padded = pad_sequence(corr_sequences, batch_first=True) # pad variable length correspondence sequences to max seq length

    return input_sequences_padded, corr_sequences_padded, input_seq_lengths, corr_seq_lengths


def val_collate_fn(batch):
    # batch is list containing elements of (data, label, length, keys, speakers) 
    # where data is shape (seq_len, feat_dim) and label is integer
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True) # sort elements according to seq_len from max to min
    sequences = [x[0] for x in sorted_batch] # list containg variable length sequences
    lengths = torch.LongTensor([len(x) for x in sequences]) # contains original length of sequences in batch
    sequences_padded = pad_sequence(sequences, batch_first=True) # pad variable length sequences to max seq length
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


# def train_ae(options_dict):
def train(options_dict):

    # Random seeds
    random.seed(options_dict['rnd_seed'])
    np.random.seed(options_dict['rnd_seed'])
    torch.manual_seed(options_dict['rnd_seed'])

    # Set operating device
    device = options_dict['device']

    # Pretrain if True
    if options_dict['pretrain']:
        options_dict['model_type'] = 'train_ae_rnn'
 
        # Load traing data
        dataset_train = GlobalPhone('train', options_dict)
        dataloader_train = DataLoader(dataset = dataset_train,
                                batch_size = options_dict['ae_batch_size'], 
                                shuffle = True,
                                drop_last = True,
                                collate_fn = ae_train_collate_fn)
        # Load validation data
        dataset_val = GlobalPhone('val', options_dict)   
        dataloader_val = DataLoader(dataset = dataset_val,
                                batch_size = dataset_val.__len__(), 
                                shuffle = False,
                                drop_last = True,
                                collate_fn = val_collate_fn)

        # Initialize model
        model = ae_rnn(options_dict)
        model.to(device)

        # Loss and optimization function
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict['ae_learning_rate'])

        # Load model from checkpoint
        # if options_dict['resume'] == True:       
        #     start_epoch = load_checkpoint(model, optimizer) + 1
        #     print("Start from epoch no.: ", start_epoch)
        # else:
        #     start_epoch = 0

        # # Save options dictionary before training
        # if not options_dict['resume']:
        #     save_options_dict(options_dict)

        # # Load record_dict or create new one
        # record_dict_fn = path.join(options_dict['model_dir'], 'record_dict.pkl')
        # if options_dict['resume']:
        #     record_dict = load_record_dict(record_dict_fn)
        # else:
        #     record_dict = {}
        #     record_dict['epoch_time'] = []
        #     record_dict['train_loss'] = []
        #     record_dict['validation_loss'] = []

        best_val_loss = np.inf
        for epoch in range(options_dict['ae_n_epochs']):
            start_time = timeit.default_timer()
            # Set model in training mode
            model.train()
            for i, (sequences_padded, seq_lengths) in enumerate(dataloader_train):
                sequences_padded, seq_lengths = sequences_padded.to(device), seq_lengths.to(device)

                # Apply forward function
                outputs = model(sequences_padded, seq_lengths)

                # Loss
                train_loss = criterion(outputs, sequences_padded) 

                # Backward and optimize
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step() 
                
            end_time = timeit.default_timer()
            epoch_time = end_time - start_time

            print ('Epoch [{}/{}], Loss: {:.4f}, Time: {:.4f} sec' 
                    .format(epoch+1, options_dict['ae_n_epochs'], train_loss.item(), (epoch_time)))  

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

            # End of epoch
            # save record dict
            # record_dict['train_loss'].append((epoch, train_loss.item()))
            # record_dict['epoch_time'].append((epoch, epoch_time))   
            # record_dict['validation_loss'].append((epoch, val_loss))        
            # save_record_dict(record_dict, record_dict_fn)

            # save if best model thus far
            if current_val_loss < best_val_loss:
                save_best_model(model, optimizer, epoch, options_dict['model_dir'], options_dict['model_type'])
                best_val_loss = current_val_loss  


    # Train CAE-RNN
    options_dict['model_type'] = 'train_cae_rnn'
    if not options_dict['pretrain']:
        dataset_train = GlobalPhone('train', options_dict)
    else:
        # Update model type in dataset for __getitem__()
        dataset_train.model_type = options_dict['model_type']
    
    # Use own sampler
    paired_sampler = samplers.PairedSampler(dataset_train, options_dict)
    paired_batch_sampler = torch.utils.data.BatchSampler(paired_sampler, options_dict['cae_batch_size'], drop_last=True)



    dataloader_train = DataLoader(dataset = dataset_train,
                            batch_size = 1,
                            batch_sampler=paired_batch_sampler,
                            shuffle = False,
                            drop_last = False,
                            sampler=None,
                            collate_fn = cae_train_collate_fn)
    
    # Load validation data
    if not options_dict['pretrain']:
        dataset_val = GlobalPhone('val', options_dict)
            
        dataloader_val = DataLoader(dataset = dataset_val,
                                batch_size = dataset_val.__len__(), 
                                shuffle = False,
                                drop_last = True,
                                collate_fn = val_collate_fn)


    # Initialize model 
    model = cae_rnn(options_dict)
    model.to(device)

    

    if options_dict['pretrain']:
        pretrained_model_dir = path.join(options_dict['model_dir'])
        model_dir_fn = path.join(pretrained_model_dir, 'train_ae_rnn.final_model.pt')

        state = torch.load(model_dir_fn)
        state_dict = state['state_dict']
        model.load_state_dict(state_dict, strict=False)

        # Loss and optimization function
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict['cae_learning_rate'])
        # optimizer.load_state_dict(state['optimizer'])

    elif options_dict["load"]:
        pretrained_model_dir = path.join('../models/RU+CZ+FR+PL+TH+PO.gt/train_cae_rnn/8415c0ad94')
        model_dir_fn = path.join(pretrained_model_dir, 'final_model.pt')

        # state = torch.load(model_dir_fn)
        # state_dict = state['state_dict']
        state_dict = torch.load(model_dir_fn)

        keys = [name for name, p in state_dict.items() if 'decoder' in name]
        for key in keys:
            del state_dict[key]

        model.load_state_dict(state_dict, strict=False)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict['cae_learning_rate'])
        # optimizer.load_state_dict(state['optimizer'])

        def freeze(model):
            for name, p in model.named_parameters():
                if 'encoder.gru' in name:
                    print(name)
                    p.requires_grad = False
            return None

        # freeze(model)

    else:
        # Loss and optimization function
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=options_dict['cae_learning_rate'])
        optimizer.load_state_dict(state['optimizer'])



    # # Load model from checkpoint
    # if options_dict['resume'] == True:       
    #     start_epoch = load_checkpoint(model, optimizer) + 1
    # else:
    #     start_epoch = 0

    # Save options dictionary before training
    # if not options_dict['resume']:
    #     save_options_dict(options_dict)

    # Load record_dict or create new one
    record_dict_fn = path.join(options_dict['model_dir'], 'record_dict.pkl')
    # if options_dict['resume']:
    #     record_dict = load_record_dict(record_dict_fn)
    # else:
    record_dict = {}
    record_dict['epoch_time'] = []
    record_dict['train_loss'] = []
    record_dict['validation_loss'] = []

    # Start training
    best_val_loss = np.inf
    for epoch in range(options_dict['cae_n_epochs']):
        start_time = timeit.default_timer()

        # Set model in training mode
        model.train()
        for i, (input_seq_padded, corr_seq_padded, input_seq_len, corr_seq_len) in enumerate(dataloader_train):
            input_seq_padded, corr_seq_padded =  input_seq_padded.to(device), corr_seq_padded.to(device)
            
            # Apply forward function
            decoded_outputs = model(input_seq_padded, input_seq_len, corr_seq_len)

            # Apply loss function
            train_loss = criterion(decoded_outputs, corr_seq_padded) 

            # Backward and optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step() 
            
        end_time = timeit.default_timer()
        epoch_time = end_time - start_time

        print ('Epoch [{}/{}], Loss: {:.4f}, Time: {:.4f}' 
                .format(epoch+1, options_dict['cae_n_epochs'], train_loss.item(), (epoch_time)))
        
        # Set model in eval mode       
        model.eval()
        with torch.no_grad():
            for i, (sequences_padded_val, lengths_val, labels_val, keys_val, speakers_val) in enumerate(dataloader_val):
                sequences_padded_val, lengths_val = sequences_padded_val.to(device), lengths_val.to(device)
                
                np_z = model(sequences_padded_val, lengths_val).cpu().detach().numpy()
                break # single batch
                
            # Do same_diff eval
            val_loss = same_diff(np_z, labels_val, speakers_val, keys_val)
            current_val_lost = val_loss[-1]            
            print(val_loss)
        
        # End of epoch
        record_dict['train_loss'].append((epoch, train_loss.item()))
        record_dict['epoch_time'].append((epoch, epoch_time))    
        record_dict['validation_loss'].append((epoch, val_loss))   
        save_record_dict(record_dict, record_dict_fn)

         # if best modal thus far save mddel
        if current_val_lost < best_val_loss:
            save_best_model(model, optimizer, epoch , options_dict['model_dir'], options_dict['model_type'])
            best_val_loss = current_val_lost 


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "train_lang", type=str,
        help="GlobalPhone training language {BG, CH, CR, CZ, FR, GE, HA, KO, "
        "PL, PO, RU, SP, SW, TH, TU, VN} or combination (e.g. BG+CH)",
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
        "--n_max_tokens", type=int,
        help="maximum number of tokens per language (default: %(default)s)",
        default=default_options_dict["n_max_tokens"]
        )
    parser.add_argument(
        "--n_max_tokens_per_type", type=int,
        help="maximum number of tokens per type per language "
        "(default: %(default)s)",
        default=default_options_dict["n_max_tokens_per_type"]
        )
    parser.add_argument(
        "--ae_n_epochs", type=int,
        help="number of epochs of AE pre-training (default: %(default)s)",
        default=default_options_dict["ae_n_epochs"]
        )
    parser.add_argument(
        "--cae_n_epochs", type=int,
        help="number of epochs of CAE training (default: %(default)s)",
        default=default_options_dict["cae_n_epochs"]
        )
    parser.add_argument(
        "--ae_batch_size", type=int,
        help="size of mini-batch for AE pre-training (default: %(default)s)",
        default=default_options_dict["ae_batch_size"]
        )
    parser.add_argument(
        "--cae_batch_size", type=int,
        help="size of mini-batch for CAE training (default: %(default)s)",
        default=default_options_dict["cae_batch_size"]
        )
    parser.add_argument(
        "--n_hiddens", type=int,
        help="number of hidden units in both the encoder and decoder "
        "(only used if n_layers is also given)"
        )
    parser.add_argument(
        "--dropout", type=float,
        help="dropout probability (default: %(default)s)",
        default=default_options_dict["dropout"]
        )
    parser.add_argument(
        "--train_tag", type=str, choices=["gt", "utd", "rnd"],
        help="training set tag (default: %(default)s)",
        default=default_options_dict["train_tag"]
        )
    parser.add_argument(
        "--cae_learning_rate", type=float,
        help="learning rate (default: %(default)s)",
        default=default_options_dict["cae_learning_rate"]
        )
    # parser.add_argument(
    #     "--pretrain_tag", type=str, choices=["gt", "utd", "rnd"],
    #     help="pretraining set tag (default: %(default)s)",
    #     default=default_options_dict["pretrain_tag"]
    #     )
    parser.add_argument(
        "--bidirectional", action="store_true",
        help="use bidirectional encoder and decoder layers "
        "(default: %(default)s)",
        default=default_options_dict["bidirectional"]
        )
    # parser.add_argument(
    #     "--d_language_embedding", type=int,
    #     help="dimensionality of language embedding (default: %(default)s)",
    #     default=default_options_dict["d_language_embedding"]
    #     )
    parser.add_argument(
        "--rnd_seed", type=int, help="random seed (default: %(default)s)",
        default=default_options_dict["rnd_seed"]
        )
    parser.add_argument(
        "--pretrain", type=bool, help="pretrain using AE-RNN (default: %(default)s)",
        default=default_options_dict["pretrain"]
        )

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':    
    # input arguments
    resume = False
    # options_dict = {}
    # if resume:
    #     ckpt_dir = 'models/SP.gt/train0' # get from arguments
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

    options_dict = default_options_dict.copy()
    options_dict["script"] = "train_ae_cae_rnn"
    options_dict["train_lang"] = args.train_lang
    options_dict["n_max_pairs"] = args.n_max_pairs
    options_dict["n_min_tokens_per_type"] = args.n_min_tokens_per_type
    options_dict["n_max_types"] = args.n_max_types
    options_dict["n_max_tokens"] = args.n_max_tokens
    options_dict["n_max_tokens_per_type"] = args.n_max_tokens_per_type
    options_dict["val_lang"] = args.val_lang
    options_dict["ae_n_epochs"] = args.ae_n_epochs
    options_dict["cae_n_epochs"] = args.cae_n_epochs
    options_dict["dropout"] = args.dropout
    options_dict["ae_batch_size"] = args.ae_batch_size
    options_dict["cae_batch_size"] = args.cae_batch_size
    options_dict["bidirectional"] = args.bidirectional
    options_dict["train_tag"] = args.train_tag
    options_dict["pretrain"] = args.pretrain
    options_dict["rnd_seed"] = args.rnd_seed
    options_dict["cae_learning_rate"] = args.cae_learning_rate

    print(options_dict["pretrain"])

    #Checkpoint directory
    model_dir = path.join('models', options_dict['train_lang'] + '.' + options_dict['train_tag'], 'train_ae_cae_rnn')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    model_dir = path.join(model_dir, model_folder(options_dict))    
    print("Model directory: ", model_dir)

    if not os.path.isdir(model_dir):
        os.makedirs(model_dir)

    options_dict['model_dir'] = model_dir
    options_dict['resume'] = False   

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print('Device:', device)  
    options_dict['device'] = device
    

    # Train AE-RNN
    train(options_dict)
    # train_ae(options_dict)

    # Train CAE-RNN
    # options_dict['model_type'] = 'train_cae_rnn'
    # train_cae(options_dict)
