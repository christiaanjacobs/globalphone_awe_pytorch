# Multilingual Acoustic Word Embeddings on Globalphone

## Overview

## Disclaimer

## Download datasets

## Install dependencies

You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/)
- [LibROSA](http://librosa.github.io/librosa/)
- [Cython](https://cython.org/)
- [tqdm](https://tqdm.github.io/)
- [speech_dtw](https://github.com/kamperh/speech_dtw/)
- [shorten](http://etree.org/shnutils/shorten/dist/src/shorten-3.6.1.tar.gz)

To install `speech_dtw` (required for same-different evaluation) and `shorten`
(required for processing audio), run `./install_local.sh`.

You can install all the other dependencies in a conda environment by running:

    conda env create -f environment.yml
    conda activate tf1.13

## Extract speech features

Update the paths in `paths.py` to point to the data directories. Extract MFCC
features in the `features/` directory as follows:

    cd features
    ./extract_features.py SP

You need to run `extract_features.py` for all languages; run it without any
arguments to see all 16 language codes.

UTD pairs can also be analysed here, by running e.g.:

    ./analyse_utd_pairs.py SP
