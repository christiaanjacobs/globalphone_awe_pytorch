# Multilingual Acoustic Word Embeddings on Globalphone

## Overview

PyTorch implementation of multilingual acoustic word embedding approaches. The experiments are described in:
- C. Jacobs, H. Kamper, and Y. Matusevych, "Acoustic word embeddings for zero-resource languages using self-supervised contrastive learning and multilingual adaptation," in *Proc. SLT*, 2021. [[arXiv](https://arxiv.org/abs/2103.10731)]

The same Contrastive RNN implementation is used in:
- C. Jacobs and H. Kamper, "Multilingual transfer of acoustic word embeddings improves when training on languages related to the target zero-resource language," in *Proc. Interspeech*, 2021. [[arXiv](https://arxiv.org/abs/2106.12834)]

## Disclaimer
Use code at own risk.

## Download datasets

The [GlobalPhone](https://csl.anthropomatik.kit.edu/english/globalphone.php)
corpus and forced alignments of the data needs to be obtained. GlobalPhone
needs to be paid for. If you have proof of payment, we can give you access to
the forced alignments. Save the data and forced alignments in a separate
directory and update the `paths.py` file to point to the data directories.

## Install dependencies

You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch 1.4](https://pytorch.org/)
- [LibROSA](http://librosa.github.io/librosa/)
- [Cython](https://cython.org/)
- [tqdm](https://tqdm.github.io/)
- [speech_dtw](https://github.com/kamperh/speech_dtw/)
- [shorten](http://etree.org/shnutils/shorten/dist/src/shorten-3.6.1.tar.gz)

To install `speech_dtw` (required for same-different evaluation) and `shorten`
(required for processing audio), run `./install_local.sh`.

You can install all the other dependencies in a conda environment by running:

    conda env create -f environment.yml
    conda activate pyt1.4

## Extract speech features

Update the paths in `paths.py` to point to the data directories. Extract MFCC
features in the `features/` directory as follows:

    cd features
    ./extract_features.py SP

You need to run `extract_features.py` for all languages; run it without any
arguments to see all 16 language codes.

UTD pairs can also be analysed here, by running e.g.:

    ./analyse_utd_pairs.py SP
## Contributors
* Christiaan Jacobs
* [Herman Kamper](http://www.kamperh.com/)
