"""
Neural embedding models.

Author: Christiaan Jacobs
Contact: christiaanjacobs97@gmail.com
Date: 2020
"""

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class encoder_rnn(nn.Module):
    def __init__(self, options_dict, batch_size):
        super(encoder_rnn, self).__init__()
        self.input_size = options_dict['n_input']
        self.hidden_size = options_dict['hidden_size']
        self.num_layers = options_dict['num_layers']
        self.bias = options_dict['bias']
        self.batch_first = options_dict['batch_first']
        self.dropout = options_dict['dropout']
        self.bidirectional = options_dict['bidirectional']
        self.batch_size = batch_size
        self.device = options_dict['device']
        self.embedding_dim = options_dict['ff_n_hiddens']
        self.rnn_type = options_dict['rnn_type']
        
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        # Encoding
        if self.rnn_type == 'lstm': 
            self.lstm = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.gru = nn.GRU(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        
        self.fc = nn.Linear(self.hidden_size*self.num_directions, self.embedding_dim)  

           
    def forward(self, x, lengths):            
        # Training mode
        if self.training:
            # Enforce batch first 
            if not self.batch_first:
                x = permute(1,0,2)   # change x from shape (seq_len, batch, input_size) to shape (batch_size, seq_len, input_size)
            
            # Pack padded sequences
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
            # RNN 
            if self.rnn_type == 'lstm':
                output_lstm, (hidden, _) = self.lstm(packed) # manually set init cell state at init_hidden() - default zero
            elif self.rnn_type == 'gru':
                output_gru, hidden = self.gru(packed)

            # Embedded layer
            out = self.fc(hidden[-1].squeeze())

        # Validation mode
        else:
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

            if self.rnn_type == 'lstm':
                output_lstm, (hidden, _) = self.lstm(packed) # adjust batch size when calling init_hidden() when train and val batch sizes not equal
            elif self.rnn_type == 'gru':
                output_rnn, hidden = self.gru(packed)
                
                # Embedded layer   
            out = self.fc(hidden[-1])

        return out


class decoder_rnn(nn.Module):
    def __init__(self, options_dict, batch_size):
        super(decoder_rnn, self).__init__()
        self.input_size = options_dict['n_input']
        self.hidden_size = options_dict['hidden_size']
        self.num_layers = options_dict['num_layers']
        self.bias = options_dict['bias']
        self.batch_first = options_dict['batch_first']
        self.dropout = options_dict['dropout']
        self.bidirectional = options_dict['bidirectional']
        self.batch_size = batch_size
        self.device = options_dict['device']
        self.embedding_dim = options_dict['ff_n_hiddens']
        self.rnn_type = options_dict['rnn_type'] 

        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1
        
        # Recurrent layers (decoding)
        if self.rnn_type == 'lstm':                                                                                  # create sequence of length equal to encoder input sequence length
            self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                                batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        elif self.rnn_type == 'gru':
            self.gru = nn.GRU(self.embedding_dim, hidden_size=self.hidden_size, num_layers=self.num_layers, bias=self.bias, 
                              batch_first=self.batch_first, dropout=self.dropout, bidirectional=self.bidirectional)
        
        # Final decoded layer
        self.fc = nn.Linear(self.hidden_size*self.num_directions, self.input_size)

    
    def forward(self, x, lengths):

        if self.training:
            packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False) # x contains embedded layer repeated to
            
            if self.rnn_type == 'lstm':                                                                                  # create sequence of length equal to encoder input sequence length
                decoder_rnn_output, (_, _) = self.lstm(packed)
            elif self.rnn_type == 'gru':
                decoder_rnn_output, _ = self.gru(packed)

            unpacked_rnn_output, _ = pad_packed_sequence(decoder_rnn_output, batch_first=True) # (batch, max_seq_length, hidden_size)

            final_output = self.fc(unpacked_rnn_output) # (batch, max_seq_len, 13)

            # Create mask that replace values in rows wirh zeros at indices greater that original seq_len
            mask = [torch.ones(lengths[i],final_output.size(-1)) for i in range(final_output.size(0))]
            mask_padded = torch.nn.utils.rnn.pad_sequence(mask, batch_first=True).to(self.device)

            # Apply mask to final_output
            final_output_masked = torch.mul(final_output, mask_padded)
           
        return final_output_masked


class ae_rnn(nn.Module):
    def __init__(self, options_dict):
        super(ae_rnn, self).__init__()
        self.encoder = encoder_rnn(options_dict, options_dict['ae_batch_size'])
        self.decoder = decoder_rnn(options_dict, options_dict['ae_batch_size'])

    def forward(self, x, lengths):

        if self.training:
            # Encode padded sequences
            encoded_x = self.encoder(x, lengths) # shape (batch, embed_dim)

            # Apply activation on embeded layer (necessary?)
            encoded_x = torch.nn.functional.relu_(encoded_x)
            
            lengths = [int(length.tolist()) for length in lengths]

            # Decoding
            # need to repeat latent embedding as input to the rnn up to original sequence length (lengths are already sorted in decending)
            sequences = [z.unsqueeze(0).expand(lengths[i],-1) for i, z in enumerate(encoded_x)] # [batch, seq_len_i, hidden_dim]
            sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True) # pad variable length sequences to max seq length

            # Decode padded sequences
            decoded_x = self.decoder(sequences_padded, lengths)

            return decoded_x

        # Only return encoded embeddings
        else:
            return self.encoder(x, lengths)


class cae_rnn(nn.Module):
    def __init__(self, options_dict):
        super(cae_rnn, self).__init__()
        self.encoder = encoder_rnn(options_dict, options_dict['cae_batch_size'])
        self.decoder = decoder_rnn(options_dict, options_dict['cae_batch_size'])

    def forward(self, x, input_lengths, corr_lengths=None):
        
        if self.training:
            # Encode padded sequence, x
            encoded_x = self.encoder(x, input_lengths)
            
            # apply activation on embeded layer (necessary?)
            # encoded_x = torch.nn.functional.relu_(encoded_x)
            
            # Decoding
            # repeat latent embedding as input to the rnn up to corresponding output sequence length (corresponding lengths are not sorted)
            lengths_sorted = [(length,idx)  for (idx,length) in sorted(enumerate(corr_lengths), key=lambda x:x[1], reverse=True)]
            corr_lengths_sorted = [x[0].tolist() for x in lengths_sorted]
            corr_sorting_indices = [x[1] for x in lengths_sorted] # use to rearange corr_lengths to orignal sequence

            encoded_x[range(len(encoded_x))] = encoded_x[corr_sorting_indices]

            corr_sequences = [z.unsqueeze(0).expand(corr_lengths_sorted[i],-1) for i, z in enumerate(encoded_x)] # [batch, seq_len_i, hidden_dim]
            corr_sequences_padded = torch.nn.utils.rnn.pad_sequence(corr_sequences, batch_first=True) # pad variable length sequences to max seq length

            # Decode padded sequences
            decoded_x = self.decoder(corr_sequences_padded, corr_lengths_sorted)

            # Reorder output in orignal order
            final_decoded_x = torch.zeros_like(decoded_x)
            for i in range(len(decoded_x)):                
                final_decoded_x[corr_sorting_indices[i]] = decoded_x[i]

            return final_decoded_x

        else:
            encoded_x = self.encoder(x, input_lengths)
            return encoded_x


class siamese_rnn(nn.Module):
    def __init__(self, options_dict):
        super(siamese_rnn, self).__init__()
        self.encoder = encoder_rnn(options_dict, options_dict["batch_size"])

    def forward(self, x, lengths):
        encoder_rnn_output = self.encoder(x, lengths)
        return encoder_rnn_output

class contrastive_rnn(nn.Module):
    def __init__(self, options_dict):
        super(contrastive_rnn, self).__init__()
        self.encoder = encoder_rnn(options_dict, options_dict["batch_size"])

    def forward(self, x, lengths):
        encoder_rnn_output = self.encoder(x, lengths)
        return encoder_rnn_output


