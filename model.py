import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from config import ACTIVATION_FUNCTIONS, device, PERSONALISATION

RNN_MODEL = 'rnn'
TRANSFORMER_MODEL = 'transformer'

class TransformerEncoderCustom(nn.Module):
    """
    Transformer encoder
    """

    def __init__(self, params):
        super().__init__()

        self.transformer_layer = nn.TransformerEncoderLayer(d_model=params.model_dim, nhead=params.trf_num_heads,
                                                            batch_first=True, dropout=params.trf_dropout,
                                                            dim_feedforward=params.d_fc_out)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer,
                                                         num_layers=params.trf_num_layers)
        self.n_to_1 = params.n_to_1
        self.num_heads = params.trf_num_heads
        self.window_mask = params.trf_mask_windowing

    def forward(self, x, x_len):

        attn_mask = get_3d_masks(x_len, x.shape[1], self.num_heads).to(device)
        if self.window_mask > 0:
            attn_mask = window_mask(attn_mask, self.window_mask)
        # print(attn_mask)
        encodings = self.transformer_encoder(x, mask=attn_mask)

        if self.n_to_1:
            encodings = torch.mean(encodings, dim=1)
            # TODO squeeze?
            pass
            #raise NotImplementedError()
        return encodings


def create_2d_mask(length, max_length):
    """
    Auxiliary mask to create a 2D mask
    @param length: length of the sequence to consider
    @param max_length: length of the sequence
    @return: 2D mask tensor
    """
    m = torch.zeros((length, length))
    m = F.pad(m, pad=(0, max_length - length, 0, max_length - length), value=1.)
    m = m - torch.eye(max_length)
    m = torch.clip(m, min=0)
    return m.bool()


def get_3d_masks(lengths, max_length, num_heads):
    lengths = [int(l) for l in lengths.detach().cpu().numpy()]
    masks = []
    for l in lengths:
        masks.extend([create_2d_mask(l, max_length) for _ in range(num_heads)])
    return torch.stack(masks)


def window_mask(mask, neighbors=3):
    m = np.zeros((mask.shape[1], mask.shape[2]))
    for i in range(mask.shape[1]):
        m[i][max(0, i - neighbors):i + neighbors + 1] = 1
    # false -> attend, true -> do not attend
    m = np.invert(m.astype(bool))
    return torch.logical_or(torch.Tensor(m).to(device), mask)


class RNN(nn.Module):
    # TODO parameters just as args object
    def __init__(self, d_in, d_out, n_layers=1, bi=True, dropout=0.2, n_to_1=False):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=d_in, hidden_size=d_out, bidirectional=bi, num_layers=n_layers, dropout=dropout)
        self.n_layers = n_layers
        self.d_out = d_out
        self.n_directions = 2 if bi else 1
        self.n_to_1 = n_to_1

    def forward(self, x, x_len):
        x_packed = pack_padded_sequence(x, x_len.cpu(), batch_first=True, enforce_sorted=False)
        rnn_enc = self.rnn(x_packed)

        if self.n_to_1:
            # hiddenstates, h_n, only last layer
            return last_item_from_packed(rnn_enc[0], x_len)
            #batch_size = x.shape[0]
            #h_n = h_n.view(self.n_layers, self.n_directions, batch_size, self.d_out) # (NL, ND, BS, dim)
            #last_layer = h_n[-1].permute(1,0,2) # (BS, ND, dim)
            #x_out = last_layer.reshape(batch_size, self.n_directions * self.d_out) # (BS, ND*dim)

        else:
            x_out = rnn_enc[0]
            x_out = pad_packed_sequence(x_out, total_length=x.size(1), batch_first=True)[0]

        return x_out

#https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/7
def last_item_from_packed(packed, lengths):
    sum_batch_sizes = torch.cat((
        torch.zeros(2, dtype=torch.int64),
        torch.cumsum(packed.batch_sizes, 0)
    )).to(device)
    sorted_lengths = lengths[packed.sorted_indices].to(device)
    last_seq_idxs = sum_batch_sizes[sorted_lengths] + torch.arange(lengths.size(0)).to(device)
    last_seq_items = packed.data[last_seq_idxs]
    last_seq_items = last_seq_items[packed.unsorted_indices]
    return last_seq_items

class OutLayer(nn.Module):
    def __init__(self, d_in, d_hidden, d_out, dropout=.0, bias=.0):
        super(OutLayer, self).__init__()
        self.fc_1 = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(True), nn.Dropout(dropout))
        self.fc_2 = nn.Linear(d_hidden, d_out)
        nn.init.constant_(self.fc_2.bias.data, bias)

    def forward(self, x):
        y = self.fc_2(self.fc_1(x))
        return y


class Model(nn.Module):
    def __init__(self, params):
        super(Model, self).__init__()
        self.params = params

        # TODO rename parameter
        self.inp = nn.Linear(params.d_in, params.model_dim, bias=False)

        if params.model_type == RNN_MODEL:
            self.encoder = RNN(params.model_dim, params.model_dim, n_layers=params.rnn_n_layers, bi=params.rnn_bi,
                           dropout=params.rnn_dropout, n_to_1=params.n_to_1)
        elif params.model_type == TRANSFORMER_MODEL:
            self.encoder = TransformerEncoderCustom(params)

        d_rnn_out = params.model_dim * 2 if params.rnn_bi and params.rnn_n_layers > 0 else params.model_dim
        self.out = OutLayer(d_rnn_out, params.d_fc_out, params.n_targets, dropout=params.linear_dropout)
        self.final_activation = ACTIVATION_FUNCTIONS[params.task]()

    def forward(self, x, x_len):
        x = self.inp(x)
        x = self.encoder(x, x_len)
        y = self.out(x)
        activation = self.final_activation(y)
        return activation, x

    def set_n_to_1(self, n_to_1):
        self.encoder.n_to_1 = n_to_1


class PersonalisedModel(nn.Module):
    def __init__(self, wrapped_model:Model, hidden_size=64):
        super(PersonalisedModel, self).__init__()
        self.wrapped_model = wrapped_model
        # freeze
        for param in self.wrapped_model.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(0.5)
        # TODO parameter
        self.hidden = nn.Linear(64, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

        self.final_activation = ACTIVATION_FUNCTIONS[PERSONALISATION]()

    def forward(self, x, x_len):
        wrapped_pred, wrapped_enc = self.wrapped_model(x, x_len)
        pers_pred = self.final_activation(self.out(self.hidden(self.dropout(wrapped_enc))))
        return pers_pred, wrapped_pred
