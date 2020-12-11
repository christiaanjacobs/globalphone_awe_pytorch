import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence 
from scipy.spatial.distance import pdist
import os
from os import path
import sys
import argparse
import pickle
import random

#sys.path.append(path.join(".."))
sys.path.append(path.join("..", "..", "src", "speech_dtw", "utils"))
#sys.path.append(path.join("..", "train_ae_cae_rnn"))
#sys.path.append(path.join("..", "construction"))
import models
from dataset import GlobalPhone
import samediff

#import matplotlib.pyplot as plt
#import seaborn as sns

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


def val_collate_fn(batch):
    """
    batch is list containing elements of (data, label, length, keys, speakers) 
    where data is shape (seq_len, feat_dim) and label is integer
    """
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True) # sort elements according to seq_len from max to min
    sequences = [x[0] for x in sorted_batch] # list containg variable length sequences
    lengths = torch.LongTensor([len(x) for x in sequences]) # contains original length of sequences in batch
    sequences_padded = pad_sequence(sequences, batch_first=True) # pad variable length sequences to max seq length
    labels = [x[1] for x in sorted_batch] # contains label of each sequence at same index
    keys =  [x[3] for x in sorted_batch]
    speakers = [x[4] for x in sorted_batch]    
    return sequences_padded, lengths, labels, keys, speakers


def main(options_dict, model_types):
    random.seed(options_dict['rnd_seed'])
    np.random.seed(options_dict['rnd_seed'])
    torch.manual_seed(options_dict['rnd_seed'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

    dataset_val = GlobalPhone("val", options_dict)    
    dataloader_val = DataLoader(dataset = dataset_val,
                        batch_size = dataset_val.__len__(), 
                        shuffle = False,
                        drop_last = True,
                        collate_fn = val_collate_fn)
    
    model_type = options_dict['model_type']
    if model_type == 'cae_rnn':
        model = models.cae_rnn(options_dict)
        model_dir_fn = path.join(options_dict['model_dir'], 'train_cae_rnn.final_model.pt')
    if model_type == 'ae_rnn':
        model = models.ae_rnn(options_dict)
        model_dir_fn = path.join(options_dict['model_dir'], 'train_ae_rnn.final_model.pt')
    if model_type =='siamese_rnn':
        model = models.siamese_rnn(options_dict)
        model_dir_fn = path.join(options_dict['model_dir'], 'final_model.pt')
    if model_type == 'contrastive_rnn':
        model = models.contrastive_rnn(options_dict)
        model_dir_fn = path.join(options_dict['model_dir'], 'final_model.pt')

    model.load_state_dict(torch.load(model_dir_fn)['state_dict'], strict=True)
    model.to(device)
    
    
    model.eval()
    with torch.no_grad():
        for i, (sequences_padded_val, lengths_val, labels_val, keys_val, speakers_val) in enumerate(dataloader_val):
            sequences_padded_val, lengths_val = sequences_padded_val.to(device), lengths_val.to(device)
            
            np_z = model(sequences_padded_val, lengths_val).cpu().detach().numpy()
               
            break # single batch
            
        # Do same_diff eval
        val_loss = same_diff(np_z, labels_val, speakers_val, keys_val)
        current_val_lost = val_loss[-1]            
        print("Average Precision: {:.8f}".format(-current_val_lost))  


def load_options_dict(options_dict_dir):
    """
        Load model options dictionary
    """
    options_dict_dir_fn = path.join(options_dict_dir, 'options_dict.pkl')
    print('Loading:', options_dict_dir_fn)
    with open(options_dict_dir_fn, '+rb') as f:
        options_dict = pickle.load(f)
    return options_dict
        

if __name__ == '__main__':

    model_types = ["classifier_rnn", "siamese_rnn", "ae_rnn", "cae_rnn", "encoder_rnn"]
    globalphone_langs = ["BG", "CH", "CR", "CZ", "FR", "GE", "HA", "KO","PL", "PO", "RU", "SP", "SW", "TH", "TU", "VN"]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_type', type=str, help="Model types", choices=model_types) # classifier_rnn, cae_rnn. etc
    parser.add_argument('--val_lang', type=str, help="GlobalPhone language", choices=globalphone_langs) 

    args = parser.parse_args()

    model_dir = path.join(args.model_dir) # --model cae_rnn
    models_dir = path.join(args.model_dir)
    options_dict = load_options_dict(model_dir)
#    print(options_dict)
#    options_dict = {'rnd_seed': 1,
#                    'max_length': 100}
#                    
    options_dict['model_dir'] = model_dir
    options_dict['val_lang'] = args.val_lang
    options_dict['model_type'] = args.model_type
    
    main(options_dict, model_types)
