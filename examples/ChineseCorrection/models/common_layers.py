# config:utf-8
import torch
import torch.nn as nn
import numpy as np


def get_window_mask(pad_input, window_size):
    seq_len = pad_input.size(1)
    batch_size = pad_input.size(0)
    mask = torch.ones(seq_len, seq_len).type_as(pad_input)
    for i in range(seq_len):
        if i < window_size:
            zero_index = (0, i + window_size)
        else:
            zero_index = (i - window_size, i + window_size)
        shape = mask[i][zero_index[0]: zero_index[1] + 1].shape
        mask[i][zero_index[0]: zero_index[1] + 1] = torch.zeros(shape)

    mask = mask.repeat(batch_size, 1, 1).bool()
    return mask


def get_select_mask(pad_input, window_size):
    seq_len = pad_input.size(1)
    batch_size = pad_input.size(0)
    mask = torch.zeros(seq_len, seq_len).type_as(pad_input)
    for i in range(seq_len):
        if i < window_size:
            ones_index = (0, window_size*2)
        elif i >= seq_len-window_size:
            ones_index = (seq_len-2*window_size-1, seq_len-1)
        else:
            ones_index = (i - window_size, i + window_size)
        shape = mask[i][ones_index[0]: ones_index[1] + 1].shape
        mask[i][ones_index[0]: ones_index[1] + 1] = torch.ones(shape)

    mask = mask.repeat(batch_size, 1, 1).bool()
    return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, dim_model, dim_key, dropout=0.1, window_size=2):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.dim_key = dim_key
        # dim_model是输入的最后一个维度，num_heads*dim_key是输出的最后一个维度，前面的维度保持不变
        self.query_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.key_linear = nn.Linear(dim_model, num_heads * dim_key)
        self.window_size = window_size

        nn.init.normal_(self.query_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))
        nn.init.normal_(self.key_linear.weight, mean=0, std=np.sqrt(2.0 / (self.dim_model + self.dim_key)))

        self.attention = ScaledDotProductAttention(temperature=np.power(dim_key, 0.5), attn_dropout=dropout)

    def forward(self, query, key):
        """
        dim_model = H
        query: B x T_Q x H, key: B x T_K x H
        mask: B x T x T (attention mask)
        return:
            output: [batch_size, T, dim_model]
            attention: [batch_size*num_heads, T, T]
        """
        batch_size, len_query, _ = query.size()  # batch_size, T, dim_model
        batch_size, len_key, _ = key.size()
        mask = get_window_mask(query, self.window_size)

        # B x T_Q x num_heads x dim_key
        query = self.query_linear(query).view(batch_size, len_query, self.num_heads, self.dim_key)
        # B x T_K x num_heads x dim_key
        key = self.key_linear(key).view(batch_size, len_key, self.num_heads, self.dim_key)

        # query.permute(2, 0, 1, 3)就是更换维度位置[num_heads, batch_size, T, dim_query]
        # view后就是[num_heads*batch_size, T, dim_query]
        query = query.permute(2, 0, 1, 3).contiguous().view(-1, len_query, self.dim_key)  # (num_heads * B) x T_Q x H_K
        key = key.permute(2, 0, 1, 3).contiguous().view(-1, len_key, self.dim_key)  # (num_heads * B) x T_K x H_K

        # 扩充mask，将mask的第一维扩充成[batch_size*num_heads, T, T]
        if mask is not None:
            mask = mask.repeat(self.num_heads, 1, 1)  # (B * num_head) x T x T
        attn = self.attention(query, key, mask=mask).view(self.num_heads, batch_size, len_query, len_query)
        attn = attn.transpose(0, 1)  # (B, num_head, T, T)

        attn = attn.reshape(-1, len_query, len_query)
        select_mask = get_select_mask(attn, self.window_size)

        attn = attn.masked_select(select_mask)

        attn = attn.reshape(batch_size, self.num_heads, len_query, -1)

        return attn


class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, mask=None):
        """
        q, k, v :[num_heads*batch_size, T, dim_query]
        mask: [num_heads*batch_size, T, dim_query]
        return:
            output: [batch_size*num_heads, T, dim_value]
            attention: [batch_size*num_heads, T, T]
        """
        attn = torch.bmm(q, k.transpose(1, 2))  # [num_heads*batch_size, T, T]
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)  # mask为True的地方用-np.inf填充，其他不变
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        return attn
