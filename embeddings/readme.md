# Acoustic Word Embedding Models and Evaluation

## Overview

## Data preperation
Create links to the MFCC NumPy archives:

    ./link_mfcc.py SP

You need to run `link_mfcc.py` for all languages; run it without any arguments
to see all 16 language codes. Alternatively, links can be greated for all
languages by passing the "all" argument.

## Correspondance autoencoder RNN
Train a CAE-RNN on Spanish UTD segments:
 
    ./train_ae_cae_rnn.py --pretrain True --ae_n_epochs 25 /
    --cae_n_epochs 10 --ae_batch_size 300 --cae_batch_size 300 /
    --train_tag utd --val_lang SP SP

Evaluate the model:
    
    ./apply_model.py ...

Train a CAE-RNN on Spanish ground truth segments:

    ./train_ae_cae_rnn.py --pretrain True --n_max_pairs 100000 /
        --ae_n_epochs 25 --cae_n_epochs 10 --ae_batch_size 300 /
        --cae_batch_size 300 --train_tag gt --val_lang SP SP
        
 Train a CAE-RNN jointly on multiple languages, limiting the maximum overall number of pairs,
 the maximum number of types per language and requiring a minimum number of tokens per type.
 
    ./train_ae_cae_rnn.py --pretrain True --n_max_pairs 300000 /
        --n_min_tokens_per_type 2 --n_max_types 1000 ---ae_n_epochs 15 /
        --cae_n_epochs 10 --train_tag gt --val_lang GE RU+CZ+FR+PL+TH+PO

## Siamese RNN
Train a Siamese RNN on ground truth segments:

    ./train_siamese_rnn.py --margin 0.25 --n_epochs 25 /
        --n_max_pairs 100000 --learning_rate 0.0005 /
        --train_tag gt --val_lang SP SP
