"""
Utility functions.

Author: Christiaan Jacobs
Contact: christiaanjacobs97@gmail.com
Date: 2020
"""

import os
from os import path
import pickle
import torch
import hashlib

def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    """
    Save model state at specified epoch during training
    """
    checkpoint_state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch}
    ckpt_dir = path.join(checkpoint_dir, 'checkpoints') 
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    ckpt_dir = path.join(ckpt_dir, 'model.ckpt.epoch.{}.pt'.format(epoch))
    torch.save(checkpoint_state, ckpt_dir)
    print('Saved checkpoint: {}'.format(ckpt_dir))

def save_best_model(model, optimizer, epoch, model_dir, model_type=None):
    """
    Save best model at time
    """
    if model_type is not None:
        model_dir_fn = path.join(model_dir, model_type+'.final_model.pt')
        state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, model_dir_fn)
        print('Saved new best model: {}'.format(model_dir_fn))
    else:
        model_dir_fn = path.join(model_dir, 'final_model.pt')
        state = {'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        torch.save(state, model_dir_fn)
        print('Saved new best model: {}'.format(model_dir_fn))

def load_checkpoint(model, optimizer):
    """
    Resume model training from last saved checkpoint
    """
    ckpt_dir_fn = path.join(options_dict['checkpoint_dir'], 'checkpoints')
    ckpt_file = path.join(ckpt_dir_fn, sorted(os.listdir(ckpt_dir_fn))[-1])
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    epoch = ckpt['epoch']
    return epoch
    

def save_options_dict(options_dict):
    """
    Save model options dictionary
    """
    option_dict_session = path.join(options_dict['model_dir'], 'options_dict.pkl')
    print('Writing:', option_dict_session)
    with open(option_dict_session, '+wb') as f:
        pickle.dump(options_dict, f, -1)
        
def save_record_dict(record_dict, record_dict_fn):
    """
    Save record dictionary when called
    """
    with open(record_dict_fn, 'wb') as f:
        pickle.dump(record_dict, f, -1)

def load_record_dict(record_dict_fn):
    """
    Load record dictionary when resume == True
    """
    with open(record_dict_fn, 'rb') as f:
        record_dict = pickle.load(f)
    return record_dict

def model_folder(options_dict):
    """
    Create output directory from options dictionary (Herman)
    """
    hasher = hashlib.md5(repr(sorted(options_dict.items())).encode("ascii"))
    hash_str = hasher.hexdigest()[:10]
    return hash_str

