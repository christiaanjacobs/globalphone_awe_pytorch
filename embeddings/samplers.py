"""
Sampling methods for batch construction.

Author: Christiaan Jacobs
Contact: christiaanjacobs97@gmail.com
Date: 2020
"""

from torch.utils.data import Sampler
import numpy as np
import random


class PairedSampler(Sampler):
    # TODO shuffle pairs
    """
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, options_dict):
        self.data_source = data_source

        # get_pair_list shuffles pairs given n_max_pairs
        self.pair_list = get_pair_list(
            data_source.labels, both_directions=True,
            n_max_pairs=options_dict["n_max_pairs"]
            )
        if options_dict["n_max_pairs"] is None:
            random.seed(options_dict['rnd_seed'])
            random.shuffle(self.pair_list)

        print("Total pairs: ", len(self.pair_list))
    
    def __len__(self):
        return len(self.pair_list)

    def __iter__(self):
        return iter(self.pair_list)




class SiameseSampler(Sampler):
    # TODO shuffle pairs
    """
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, options_dict):
        self.data_source = data_source

        # get_pair_list shuffles pairs given n_max_pairs
        self.pair_list = get_pair_list(
            data_source.labels, both_directions=True,
            n_max_pairs=options_dict["n_max_pairs"]
            )
        if options_dict["n_max_pairs"] is None:
            random.seed(options_dict['rnd_seed'])
            random.shuffle(self.pair_list)

        print(len(self.pair_list))
        self.flat_list = [item for sublist in self.pair_list for item in sublist]
        
        self.flat_list = self.flat_list[:len(self.flat_list)//options_dict['batch_size']*options_dict['batch_size']]
        self.flat_list = np.reshape(self.flat_list, (-1, options_dict['batch_size']))
        print(self.flat_list.shape)
            
    def __len__(self):
        return len(self.flat_list)

    def __iter__(self):
        return iter(self.flat_list)

class N_pair_sampler(Sampler):

    def __init__(self, data_source, options_dict):
        self.labels = data_source.labels
        self.speakers = data_source.speakers
        self.batch_size = options_dict["batch_size"]
        self.pair_list = get_pair_list(
            self.labels, both_directions=True, n_max_pairs=options_dict["n_max_pairs"]
            )
        if options_dict["n_max_pairs"] is None:
            random.seed(options_dict["rnd_seed"])
            random.shuffle(self.pair_list)

        self.n_minibatches = options_dict["n_minibatches"]
        
        # self.tuplet_list_idxs, self.tuplet_list_hot = get_tuplet_lists(self.pair_list, self.labels, 5)
        # self.tuplet_list_idxs, self.tuplet_list_hot = tuplets_shuffle(self.tuplet_list_idxs, self.tuplet_list_hot)
        original = self.pair_list.copy()
        self.batch = get_minibatch(original, self.labels, self.batch_size, self.n_minibatches)

    def __iter__(self):
        """
        returns tuplet of idxs and 2-hot eg. [2, 45, 67, 102, 43], [0, 1, 0, 0, 1]
        """
        return iter(self.batch)




#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#


def get_pair_list(labels, both_directions=True, n_max_pairs=None):
    """Return a list of tuples giving indices of matching types."""
    N = len(labels)
    match_list = []
    for n in range(N - 1):
        cur_label = labels[n]
        for cur_match_i in (n + 1 + np.where(np.asarray(labels[n + 1:]) ==
                cur_label)[0]):
            match_list.append((n, cur_match_i))
            # if both_directions:
            #     match_list.append((cur_match_i, n))
    if n_max_pairs is not None:
        random.seed(1)  # use the same list across different models
        random.shuffle(match_list)
        match_list = match_list[:n_max_pairs]
    if both_directions:
        return match_list + [(i[1], i[0]) for i in match_list]
    else:
        return match_list


def get_minibatch(pairs, labels, N, n_minibatches):
    batch = []
    for i, positive_pair in enumerate(pairs):
        # print("len pairs", len(pairs))
        minibatch = []
        minibatch.append(positive_pair)
        # print(positive_pair)
        positive_label = labels[positive_pair[0]]
        # print(positive_label)

        minibatch_labels = []
        for pair in pairs[i+1:]:
            # print(pair)
            pair_label = labels[pair[0]]
            # print(pair_label)
            if pair_label != positive_label and pair_label not in minibatch_labels:
                # print(pair)
                # print(pair_label)
                minibatch.append(pair)
                # print(minibatch)
                minibatch_labels.append(pair_label)
                # print(minibatch_labels)
                pairs.remove(pair)
                # del pairs[labels.index(pair)]
            if len(minibatch) == N:
                break

        if len(minibatch) == N:
            batch.append(minibatch)
        if n_minibatches is not None:
            if len(batch) == n_minibatches:
                break

    print("No. minibatches:", len(batch))
    return batch
