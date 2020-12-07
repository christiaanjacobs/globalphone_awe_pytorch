# Multilingual Acoustic Word Embeddings on Globalphone

## Overview

## Disclaimer

## Download datasets

## Install dependencies

## Extract speech features

Update the paths in `paths.py` to point to the data directories. Extract MFCC
features in the `features/` directory as follows:

    cd features
    ./extract_features.py SP

You need to run `extract_features.py` for all languages; run it without any
arguments to see all 16 language codes.

UTD pairs can also be analysed here, by running e.g.:

    ./analyse_utd_pairs.py SP
