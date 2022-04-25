"""
Data input and output functions.
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2018, 2019
"""

import numpy as np
import torch
from random import randint
from os import path
from tqdm import tqdm
from collections import Counter
import random

def load_data_from_npz(data_dir, min_length=None):
    print("Reading:", data_dir)
    npz = np.load(data_dir)
    x = []
    labels = []
    speakers = []
    lengths = []
    keys = []
    n_items = 0
    for utt_key in tqdm(sorted(npz)):
        cur_x = npz[utt_key]
        if min_length is not None and len(cur_x) <= min_length:
            continue
        keys.append(utt_key)
        x.append(torch.tensor((cur_x), dtype=torch.float32))
        utt_key_split = utt_key.split("_")
        word = utt_key_split[0]
        speaker = utt_key_split[1]
        labels.append(word)
        speakers.append(speaker)
        lengths.append(len(cur_x))
        n_items += 1
#        if n_items == 2000:
#            break

    print("No. items:", n_items)
    print("E.g. item shape:", x[0].shape)
    return (x, labels, lengths, keys, speakers)


def filter_data(data, labels, lengths, keys, speakers,
        n_min_tokens_per_type=None, n_max_types=None, n_max_tokens=None,
        n_max_tokens_per_type=None):
    """
    Filter the output from `load_data_from_npz` based on specifications.

    Each filter is applied independelty, so they could influence each other.
    E.g. `n_max_tokens` could further reduce the number of types if it is used
    in conjunction with `n_max_types`.

    Return
    ------
    data, labels, lengths keys, speakers : list, list, list, list
        The filtered lists.
    """

    random.seed(1)

    if n_max_types is not None:

        print("Maximum no. of types:", n_max_types)

        # Find valid types
        types = [i[0] for i in Counter(labels).most_common(n_max_types)]

        # Filter
        filtered_data = []
        filtered_labels = []
        filtered_lengths = []
        filtered_keys = []
        filtered_speakers = []
        for i in range(len(data)):
            if labels[i] in types:
                filtered_data.append(data[i])
                filtered_labels.append(labels[i])
                filtered_lengths.append(lengths[i])
                filtered_keys.append(keys[i])
                filtered_speakers.append(speakers[i])

        data = filtered_data
        labels = filtered_labels
        lengths = filtered_lengths
        keys = filtered_keys
        speakers = filtered_speakers

    if n_max_tokens_per_type is not None:

        print("Maximum tokens per type:", n_max_tokens_per_type)

        # Filter
        filtered_data = []
        filtered_labels = []
        filtered_lengths = []
        filtered_keys = []
        filtered_speakers = []
        indices = list(range(len(data)))
        random.shuffle(indices)
        tokens_per_type = Counter()
        for i in indices:
            if tokens_per_type[labels[i]] < n_max_tokens_per_type:
                filtered_data.append(data[i])
                filtered_labels.append(labels[i])
                filtered_lengths.append(lengths[i])
                filtered_keys.append(keys[i])
                filtered_speakers.append(speakers[i])
                tokens_per_type[labels[i]] += 1

        data = filtered_data
        labels = filtered_labels
        lengths = filtered_lengths
        keys = filtered_keys
        speakers = filtered_speakers

    if n_max_tokens is not None:

        print("Maximum no. of tokens:", n_max_tokens)

        # Filter
        filtered_data = []
        filtered_labels = []
        filtered_lengths = []
        filtered_keys = []
        filtered_speakers = []
        indices = list(range(len(data)))
        random.shuffle(indices)
        # for i in range(len(data)):
        for i in indices[:n_max_tokens]:
            filtered_data.append(data[i])
            filtered_labels.append(labels[i])
            filtered_lengths.append(lengths[i])
            filtered_keys.append(keys[i])
            filtered_speakers.append(speakers[i])

        data = filtered_data
        labels = filtered_labels
        lengths = filtered_lengths
        keys = filtered_keys
        speakers = filtered_speakers

    if n_min_tokens_per_type is not None:

        print("Minimum tokens per type:", n_min_tokens_per_type)

        # Find valid types
        types = []
        counts = Counter(labels)
        for key in counts:
            if counts[key] >= n_min_tokens_per_type:
                types.append(key)

        # Filter
        filtered_data = []
        filtered_labels = []
        filtered_lengths = []
        filtered_keys = []
        filtered_speakers = []
        for i in range(len(data)):
            if labels[i] in types:
                filtered_data.append(data[i])
                filtered_labels.append(labels[i])
                filtered_lengths.append(lengths[i])
                filtered_keys.append(keys[i])
                filtered_speakers.append(speakers[i])

        data = filtered_data
        labels = filtered_labels
        lengths = filtered_lengths
        keys = filtered_keys
        speakers = filtered_speakers

    f = open('labels.txt', 'w+')
    for label in labels:
        f.write(str(label)+'\n')
    f.close


    print("No. types:", len(Counter(labels)))
    print("No. tokens:", len(labels))
    return (data, labels, lengths, keys, speakers)


def trunc_and_limit_dim(x, lengths, d_frame, max_length):
    for i, seq in enumerate(x):
        x[i] = x[i][:max_length, :d_frame]
        if max_length is not None:
            lengths[i] = min(lengths[i], max_length)
