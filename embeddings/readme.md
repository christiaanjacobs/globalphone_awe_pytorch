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

