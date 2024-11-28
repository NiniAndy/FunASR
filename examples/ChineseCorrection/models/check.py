import torch
from torch import nn
from transformers import BertModel, AutoConfig
from collections import OrderedDict
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertOnlyMLMHead
from models.common_layers import MultiHeadAttention
import torch.nn.functional as F
import json
from torch import nn


# class Check(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.bert_config = AutoConfig.from_pretrained(config.PRETRAINED.config)
#         self.embeddings = BertEmbeddings(self.bert_config)
#         self.load_from_transformers_state_dict(config.PRETRAINED.model)
#         self.window_size = config.MODEL.window_size
#         self.local_att = MultiHeadAttention(12, 768, 64, dropout=0.1, window_size=self.window_size)
#         self.conv = nn.Conv2d(12, 36, (1, self.window_size * 2 + 1))
#
#         self.fc1 = nn.Linear(36, 1)
#
#     def forward(self, x):
#         emb = self.embeddings(input_ids=x)
#         local_attn = self.local_att(emb, emb)
#         out = self.conv(local_attn)
#         out = out.squeeze(-1)
#         out = out.transpose(1, 2)
#         out = torch.sigmoid(self.fc1(out)).squeeze(-1)
#         return out
#
#     def load_from_transformers_state_dict(self, gen_fp):
#         state_dict = OrderedDict()
#         gen_state_dict = BertModel.from_pretrained(gen_fp).state_dict()
#         for k, v in gen_state_dict.items():
#             name = k
#             if name.startswith('bert'):
#                 name = name[5:]
#             if name.startswith('encoder'):
#                 name = f'corrector.{name[8:]}'
#             if 'gamma' in name:
#                 name = name.replace('gamma', 'weight')
#             if 'beta' in name:
#                 name = name.replace('beta', 'bias')
#             state_dict[name] = v
#         self.load_state_dict(state_dict, strict=False)


class Check(nn.Module):
    def __init__(self, config):
        super(Check, self).__init__()

        self.embedding_dim = config.MODEL.embedding_dim
        self.lstm_hidden_dim = config.MODEL.lstm_hidden_dim
        self.fc_hidden = config.MODEL.fc_hidden
        self.vocab_size = config.TOKENIZER.vocab_size

        self.char_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.forward_lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, batch_first=True)
        self.backward_lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, batch_first=True)
        self.drop_out = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.lstm_hidden_dim*2, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 1)
        self.window_size = config.MODEL.window_size

    def forward(self, x):
        batch_size, seq_len = x.size(0), x.size(1)
        emb = self.char_embedding(x)
        emb = F.pad(emb.transpose(1, 2), (self.window_size, self.window_size), "constant", 0).transpose(1, 2)
        emb = emb.contiguous()  # 强制内存连续化，否则后续会出错
        emb = self.drop_out(emb)
        # 前向
        forward_x = emb.as_strided((batch_size, seq_len, self.window_size+1, self.embedding_dim), ((seq_len+2*self.window_size)*self.embedding_dim, self.embedding_dim, self.embedding_dim, 1))

        # 后向
        emb_back = emb.flip(1).contiguous()
        backward_x = emb_back.as_strided((batch_size, seq_len, self.window_size+1, self.embedding_dim), ((seq_len+2*self.window_size)*self.embedding_dim, self.embedding_dim, self.embedding_dim, 1))
        backward_x = backward_x.flip(1)

        forward_x = forward_x.reshape(batch_size*seq_len, self.window_size+1, self.embedding_dim)
        backward_x = backward_x.reshape(batch_size * seq_len, self.window_size + 1, self.embedding_dim)

        out_forward, (h_forward, c_forward) = self.forward_lstm(forward_x)

        out_backward, (h_backward, c_backward) = self.backward_lstm(backward_x)

        bi_local_lstm_hidden = torch.cat((h_forward[0], h_backward[0]), dim=1)

        bi_local_lstm_hidden = bi_local_lstm_hidden.reshape(batch_size, seq_len, self.lstm_hidden_dim*2)

        bi_local_lstm_hidden = F.leaky_relu(bi_local_lstm_hidden)

        out = self.fc1(bi_local_lstm_hidden)

        out = F.leaky_relu(out)

        out = F.sigmoid(self.fc2(out)).squeeze(-1)

        return out




class Single_Check(nn.Module):
    def __init__(self, kwargs):
        super(Single_Check, self).__init__()

        self.embedding_dim = kwargs["model_conf"]["embedding_dim"]
        self.lstm_hidden_dim = kwargs["model_conf"]["lstm_hidden_dim"]
        self.fc_hidden =  kwargs["model_conf"]["fc_hidden"]
        self.vocab_size = kwargs["tokenizer_conf"]["vocab_size"]
        self.window_size = kwargs["model_conf"]["window_size"]

        self.char_embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)
        self.bi_lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.drop_out = nn.Dropout(0.1)
        self.fc1 = nn.Linear(self.lstm_hidden_dim * 2, self.fc_hidden)
        self.fc2 = nn.Linear(self.fc_hidden, 1)

        max_batch_size, max_seq_len = 512, 128
        row_idx = torch.arange(max_seq_len).unsqueeze(-1).expand(-1, self.window_size + 1).clone()
        col_idx = torch.arange(self.window_size + 1).unsqueeze(0) + torch.arange(max_seq_len).unsqueeze(-1)
        col_idx = col_idx.unsqueeze(0).expand(max_batch_size, -1, -1).clone()  # [batch_size, seq_len, window_size+1]
        batch_idx = torch.arange(max_batch_size).unsqueeze(-1).unsqueeze(-1).expand(-1, max_seq_len, self.window_size + 1).clone() # [batch_size, seq_len, window_size+1]
        A = torch.zeros(max_batch_size, max_batch_size, max_batch_size + self.window_size)

        self.register_buffer("row_idx", row_idx)
        self.register_buffer("col_idx", col_idx)
        self.register_buffer("batch_idx", batch_idx)
        self.register_buffer("A", A)


    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)

        emb = self.char_embedding(input)
        emb = F.pad(emb.transpose(1, 2), (self.window_size, self.window_size), "constant", 0).transpose(1, 2)
        emb = emb.contiguous()  # 强制内存连续化，否则后续会出错
        emb = self.drop_out(emb)
        # 分窗口
        x = emb.as_strided((batch_size, seq_len, self.window_size+1, self.embedding_dim),
                           ((seq_len+2*self.window_size)*self.embedding_dim, self.embedding_dim, self.embedding_dim, 1))
        x = x.reshape(batch_size*seq_len, self.window_size+1, self.embedding_dim)
        out, _ = self.bi_lstm(x)
        out = out.reshape(batch_size, seq_len, self.window_size+1, -1)
        out = F.leaky_relu(out)
        out = self.fc1(out)
        out = F.leaky_relu(out)
        scores = self.fc2(out).squeeze(-1)
        A = self.A[:batch_size, :seq_len, :seq_len + self.window_size].clone()
        batch_idx = self.batch_idx[:batch_size, :seq_len, :]
        row_idx = self.row_idx[:seq_len, :]
        col_idx = self.col_idx[:batch_size, :seq_len, :]
        A[batch_idx, row_idx, col_idx] = scores.squeeze(-1)
        scores_out = A.sum(dim=1)[:, self.window_size:].contiguous()
        scores_out = F.sigmoid(scores_out)
        return scores_out