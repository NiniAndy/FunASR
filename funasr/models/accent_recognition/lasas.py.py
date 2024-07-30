import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d

from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.encoder import TransformerEncoder

class LASAS(nn.Module):

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            spurious_label_num: int = 500,
            num_spaces:int =  8,
            loss_type: str = "amsoftmax",
            encoder_conf: dict =  {},
            **kwargs,
    ):

        super(LASAS, self).__init__()

        self.input_dim = input_dim
        self.embed = nn.Embedding(spurious_label_num, input_dim)
        self.lasas = LASASBlock(text_dim=input_dim, acoustic_dim=input_dim, hidden_dim=input_dim, num_spaces=num_spaces)
        self.encoder = TransformerEncoder(input_dim, global_cmvn=None, **encoder_conf)
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim * 2),
                                    nn.Dropout(0.1),
                                    nn.Linear(input_dim * 2, input_dim),
                                    )
        self.pooling = Pooling(input_dim = input_dim, output_dim=input_dim)
        self.loss = LASASLoss(input_dim=input_dim, num_classes=output_dim, type=loss_type)


    def forward(self, Xa, Xt, mask):
        '''
        :param Xa: [i, j, k]层的Xse的Cat，shape[batch_size, seq_len, en_output_dim * num_share_layers]
        :param Xt: regular_greedy_ctc_decode()的输出，shape[batch_size, seq_len]
        :return:
        '''
        batch_size, seq_len, _ = Xa.shape
        # 1. 将Xt转换为文本向量
        Xt = self.embed(Xt)
        # 2. 将Xt和Xa转换为双模态表示
        Xbm = self.lasas(Xt, Xa, mask)
        # 3. 将Xbm输入到TransformerEncoder中
        Xbm_len =torch.sum(mask, dim=-1).squeeze(-1)
        Xbm, _ = self.encoder(Xbm, Xbm_len) # [batch_size,seq_len, hidden_dim]
        # 4. 将Xbm输入到全连接层中
        Xdnn = self.linear(Xbm)
        Xar = self.pooling(Xdnn)
        return Xar, Xdnn

    def calculate_loss(self, Xar, Yar):
        '''
        :param Yar_hat: [batch_size, acc_size]
        :param Yar: [batch_size, acc_size]
        :return:
        '''
        Yar_hat, loss, acc = self.loss(Xar, Yar)
        return Yar_hat, loss, acc



class LASASBlock(nn.Module):
    def __init__(self, text_dim, acoustic_dim, hidden_dim, num_spaces):
        super(LASASBlock, self).__init__()
        self.num_spaces = num_spaces
        self.scale = (hidden_dim / num_spaces) ** 0.5
        # 不同空间的文本映射矩阵
        self.Wt = nn.ModuleList([nn.Linear(text_dim, hidden_dim) for _ in range(num_spaces)])
        # 不同空间的声学映射矩阵
        self.Wa = nn.ModuleList([nn.Linear(acoustic_dim, hidden_dim) for _ in range(num_spaces)])
        # 文本向量的降维矩阵
        self.Wtd = nn.Linear(text_dim, hidden_dim - num_spaces)

    def forward(self, Xt, Xa, mask):
        # Xt, Xa: [batch_size, seq_len, feature_dim]
        # mask: bool [batch_size, 1, seq_len] 为False的位置为padding
        batch_size, seq_len, _ = Xt.shape
        S = []
        mask = mask.unsqueeze(-1).squeeze(1) # [B, max_seqlen, 1]

        for Wa_i, Wt_i in zip(self.Wa, self.Wt):
            Vt = Wt_i(Xt) * mask  # [batch_size, seq_len, hidden_dim]
            Va = Wa_i(Xa) * mask  # [batch_size, seq_len, hidden_dim]
            # 扩展维度以进行批量点积运算
            Si = (Vt * Va).sum(dim=-1, keepdim=True) / self.scale  # [batch_size, seq_len, 1]
            S.append(Si)

        # 合并所有相似度形成口音偏移
        S = torch.cat(S, dim=-1)  # [batch_size, seq_len, num_spaces]
        # 降低文本OneHot向量的维度
        Vtd = self.Wtd(Xt) * mask  # [batch_size, seq_len, hidden_dim]
        # 形成双模态表示
        Xbm = torch.cat([S, Vtd], dim=-1)  # [batch_size, seq_len, hidden_dim]

        return Xbm






class Pooling(nn.Module):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
    ):
        super(Pooling, self).__init__()
        self.pooling = AttentiveStatisticsPooling(input_dim)
        self.bn = BatchNorm1d(input_size=input_dim * 2)
        self.fc = nn.Linear(input_dim * 2, output_dim)

    def forward(self, concat_encoder_output):
        # concat_encoder_output: [B, T, D*num_blocks]
        concat_encoder_output = concat_encoder_output.permute(0, 2, 1)  # [B, D*num_blocks, T]
        concat_encoder_output = self.pooling(concat_encoder_output)  # [B, D*num_blocks*2, 1]
        concat_encoder_output = self.bn(concat_encoder_output)  # [B, D*num_blocks*2, 1]
        concat_encoder_output = concat_encoder_output.permute(0, 2, 1)  # [B, D*num_blocks*2, T]
        concat_encoder_output = self.fc(concat_encoder_output)
        prompt = concat_encoder_output.squeeze(1)
        return prompt




class LASASLoss(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_classes=7,
            type="amsoftmax",
    ):
        super(LASASLoss, self).__init__()

        self.type = type

        if type == "amsoftmax":
            self.m = 0.2
            self.s = 30
            self.in_feats = input_dim
            self.W = torch.nn.Parameter(torch.randn(input_dim, num_classes), requires_grad=True)
            self.ce = nn.CrossEntropyLoss()
            nn.init.xavier_normal_(self.W, gain=1)

            print('Initialised AM-Softmax m=%.3f s=%.3f' % (self.m, self.s))
            print('Embedding dim is {}, number of speakers is {}'.format(input_dim, num_classes))

        else:
            self.embedding_dim = input_dim
            self.fc = nn.Linear(input_dim, num_classes)
            self.criertion = nn.CrossEntropyLoss()

            print('init softmax')
            print('Embedding dim is {}, number of speakers is {}'.format(input_dim, num_classes))

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


    def forward(self, x, label=None):
        if self.type == "amsoftmax":
            assert x.size()[0] == label.size()[0]
            assert x.size()[1] == self.in_feats

            x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x, x_norm)
            w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.W, w_norm)
            costh = torch.mm(x_norm, w_norm)
            label_view = label.view(-1, 1)
            if label_view.is_cuda: label_view = label_view.cpu()
            delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
            if x.is_cuda: delt_costh = delt_costh.cuda()
            costh_m = costh - delt_costh
            costh_m_s = self.s * costh_m
            output = torch.argmax(costh_m_s, dim=1)
            loss = self.ce(costh_m_s, label)
            acc = self.accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]

        else:
            assert x.size()[0] == label.size()[0]
            assert x.size()[1] == self.embedding_dim

            x = F.normalize(x, dim=1)
            x = self.fc(x)
            output = torch.argmax(x, dim=1)
            loss = self.criertion(x, label)
            acc = self.accuracy(x.detach(), label.detach(), topk=(1,))[0]

        return loss, acc, output

