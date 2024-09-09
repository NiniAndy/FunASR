import torch
import torch.nn as nn
import torch.nn.functional as F

from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.lobes.models.ECAPA_TDNN import BatchNorm1d

from funasr.models.transformer.encoder import EncoderLayer, QKVEncoderLayer
from funasr.models.transformer.attention import MultiHeadedAttention
from funasr.models.transformer.embedding import PositionalEncoding
from funasr.models.transformer.layer_norm import LayerNorm
from funasr.models.transformer.utils.multi_layer_conv import Conv1dLinear
from funasr.models.transformer.utils.multi_layer_conv import MultiLayeredConv1d
from funasr.models.transformer.utils.nets_utils import make_pad_mask
from funasr.models.transformer.positionwise_feed_forward import PositionwiseFeedForward
from funasr.models.transformer.utils.repeat import repeat


class ASD(nn.Module):

    def __init__(
            self,
            input_dim: int,
            acc_num: int,
            num_spaces: int = 1,
            loss_type: str = "amsoftmax",
            encoder_conf: dict =  {},
            **kwargs,
    ):

        '''

        Args:
            input_dim: 输入的encoder combine的维度
            output_dim: accent字典的维度
            spurious_label_num: 伪标签的数量
            num_spaces: 求相似空间的数量
        '''

        super(ASD, self).__init__()

        self.input_dim = input_dim
        self.asd_block = ASDBlock(text_dim=input_dim, acoustic_dim=input_dim, hidden_dim=input_dim, num_spaces=num_spaces, **encoder_conf)

        # Transformer encoder parameters
        num_blocks = encoder_conf.get("num_blocks", 2)
        encoder_hidden_size = input_dim * num_spaces
        output_size = encoder_conf.get("output_size", 256)
        attention_heads = encoder_conf.get("attention_heads", 4)
        attention_dropout_rate = encoder_conf.get("attention_dropout_rate", 0.1)
        positionwise_layer_type = encoder_conf.get("positionwise_layer", "linear")
        linear_units = encoder_conf.get("linear_units", 2048)
        dropout_rate = encoder_conf.get("dropout_rate", 0.1)
        normalize_before = encoder_conf.get("normalize_before", True)
        concat_after = encoder_conf.get("concat_after", False)

        self._output_size = output_size


        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                encoder_hidden_size,
                linear_units,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")

        self.encoders = repeat(
            num_blocks,
            lambda lnum: EncoderLayer(
                encoder_hidden_size,
                MultiHeadedAttention(attention_heads, encoder_hidden_size, attention_dropout_rate),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.encoder_proj = torch.nn.Linear(encoder_hidden_size, output_size)

        self.gru = nn.GRU(input_size=output_size, hidden_size=output_size, num_layers=1, batch_first=True, bidirectional=False)

        self.linear = nn.Sequential(nn.Linear(output_size, output_size * 2),
                                    nn.Dropout(0.1),
                                    nn.Linear(output_size * 2, output_size),
                                    )

        # self.pooling = Pooling(input_dim = output_size, output_dim=output_size)
        self.loss = LASASLoss(input_dim=output_size, num_classes=acc_num, type=loss_type)

    def output_size(self):
        return self._output_size


    def forward(self, X_text, X_pron, X_len):
        '''
        :param X_text: tensor 指定层的文字级别输出表征 [batch_size, num_spaces, seq_len, hidden_dim]
        :param X_pron: tensor 指定层的发音级别输出表征 [batch_size, num_spaces, seq_len, hidden_dim]
        :return:
        '''
        mask = (~make_pad_mask(X_len)[:, None, :]).to(X_text.device)
        batch_size, num_space, seq_len, _ = X_text.shape
        # 2. 将Xt和Xa转换为双模态表示
        Xbm = self.asd_block(X_text, X_pron, mask)
        # 3. 将Xbm输入到TransformerEncoder中
        Xbm, _ = self.encoders(Xbm, mask) # [batch_size,seq_len, hidden_dim]
        Xdnn = self.encoder_proj(Xbm)
        _, h =self.gru(Xdnn)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        Xar = h.permute(1, 0, 2).reshape(batch_size, -1)
        # 4. 将Xbm输入到全连接层中
        Xar = self.linear(Xar)
        return Xar, Xdnn

    def calculate_loss(self, Xar, Yar):
        '''
        :param Yar_hat: [batch_size, acc_size]
        :param Yar: [batch_size, acc_size]
        :return:
        '''
        loss, acc, Yar_hat = self.loss(Xar, Yar)
        return loss, acc, Yar_hat



class ASDBlock(nn.Module):
    def __init__(self, text_dim, acoustic_dim, hidden_dim, num_spaces, **encoder_conf):
        super(ASDBlock, self).__init__()
        self.num_spaces = num_spaces
        self.scale = (hidden_dim / num_spaces) ** 0.5
        # 不同空间的文本映射矩阵
        self.Wt = nn.ModuleList([nn.Linear(text_dim, hidden_dim) for _ in range(num_spaces)])
        # 不同空间的声学映射矩阵
        self.Wa = nn.ModuleList([nn.Linear(acoustic_dim, hidden_dim) for _ in range(num_spaces)])

        # Transformer encoder parameters
        num_blocks = encoder_conf.get("num_blocks", 2)
        output_size = hidden_dim
        attention_heads = encoder_conf.get("attention_heads", 4)
        attention_dropout_rate = encoder_conf.get("attention_dropout_rate", 0.1)

        dropout_rate = encoder_conf.get("dropout_rate", 0.1)
        normalize_before = encoder_conf.get("normalize_before", True)
        concat_after = encoder_conf.get("concat_after", False)

        positionwise_layer_args = (
            output_size,
            output_size*4,
            dropout_rate,
        )


        self.encoders = QKVEncoderLayer(
                hidden_dim,
                MultiHeadedAttention(n_head=attention_heads, n_feat=hidden_dim, dropout_rate=attention_dropout_rate),
                PositionwiseFeedForward(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            )

    def forward(self, X_text, X_pron, mask):
        '''
        Args:
            X_pron: tensor 指定层的文字级别输出表征 [[batch_size, seq_len, hidden_dim], [batch_size, seq_len, hidden_dim], ...]
            X_text: tensor 指定层的发音级别输出表征 [[batch_size, seq_len, hidden_dim], [batch_size, seq_len, hidden_dim], ...]
            mask: bool [batch_size, 1, seq_len] 为False的位置为padding
        Returns:
        '''

        batch_size, num_spaces, seq_len, _ = X_text.shape
        mask = mask.unsqueeze(-1).squeeze(1) # [B, max_seqlen, 1]
        mask_expand = mask.unsqueeze(1).expand(-1, num_spaces, -1, -1)  # [B, num_spaces, max_seqlen, 1]

        X_text_new = torch.zeros_like(X_text) # 确保不在原始张量上进行就地操作

        for i in range(num_spaces):
            Vt = self.Wt[i](X_text[:, i, :, :]) * mask
            Va = self.Wa[i](X_pron[:, i, :, :]) * mask
            # 计算相似度
            sim = torch.cosine_similarity(Vt, Va, dim=-1)  # [batch_size, seq_len]
            # 相似缩放
            sim_score = 1 - sim
            X_text_new[:, i, :, :] = X_text[:, i, :, :] * sim_score.unsqueeze(-1)

        X_text_new = X_text_new.reshape(batch_size*num_spaces, seq_len, -1)  # [batch_size*num_spaces, seq_len, hidden_dim]
        X_pron = X_pron.reshape(batch_size*num_spaces, seq_len, -1)  # [batch_size*num_spaces, seq_len, hidden_dim]
        mask = mask_expand.reshape(batch_size*num_spaces, seq_len, -1)  # [batch_size*num_spaces, seq_len, 1]
        Xbm, _ = self.encoders(X_pron, X_text_new, X_text_new, mask) # [batch_size*num_spaces,seq_len, hidden_dim]

        Xbm = Xbm.reshape(batch_size, num_spaces, seq_len, -1)  # [batch_size, num_spaces, seq_len, hidden_dim]
        # reshape成[batch_size, seq_len, hidden_dim*num_spaces]
        Xbm = Xbm.permute(0, 2, 3, 1).reshape(batch_size, seq_len, -1)  # [batch_size, seq_len, hidden_dim*num_spaces]

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
            self.ce = nn.CrossEntropyLoss(ignore_index=-1)
            nn.init.xavier_normal_(self.W, gain=1)

            print('Initialised AM-Softmax m=%.3f s=%.3f' % (self.m, self.s))
            print('Embedding dim is {}, number of speakers is {}'.format(input_dim, num_classes))

        else:
            self.embedding_dim = input_dim
            self.fc = nn.Linear(input_dim, num_classes)
            self.ce = nn.CrossEntropyLoss(ignore_index=-1)

            print('init softmax')
            print('Embedding dim is {}, number of speakers is {}'.format(input_dim, num_classes))

    def accuracy(self, output, target):
        # 过滤掉 target 中为 -1 的位置
        valid_mask = (target != -1)

        # 仅计算有效样本的准确率
        valid_output = output[valid_mask]
        valid_target = target[valid_mask]

        # 如果没有有效样本，返回0或其他默认值
        if valid_target.numel() == 0:
            return torch.tensor(0.0).to(output.device)

        # 计算正确预测的数量
        correct = (valid_output == valid_target).sum().item()
        total = valid_target.size(0)  # 有效样本总数
        if total == 0:
            return torch.tensor(0.0).to(output.device)
        else:
            accuracy = correct / total  # 计算准确率
            return accuracy


    def forward(self, x, label=None):
        label = label.squeeze(-1).long()
        if self.type == "amsoftmax":
            assert x.size()[0] == label.size()[0]
            assert x.size()[1] == self.in_feats

            # x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            # x_norm = torch.div(x, x_norm)
            # w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            # w_norm = torch.div(self.W, w_norm)
            # costh = torch.mm(x_norm, w_norm)
            # label_view = label.view(-1, 1)
            # if label_view.is_cuda: label_view = label_view.cpu()
            # delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
            # if x.is_cuda: delt_costh = delt_costh.cuda()
            # costh_m = costh - delt_costh
            # costh_m_s = self.s * costh_m
            # output = torch.argmax(costh_m_s, dim=1)
            # loss = self.ce(costh_m_s, label)
            # acc = self.accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]

            valid_mask = (label != -1)

            # 仅选择有效的样本
            x_valid = x[valid_mask]
            label_valid = label[valid_mask]

            # 如果没有有效样本，返回默认的值（如零损失和零准确率）
            if label_valid.numel() == 0:
                return torch.tensor(0.0), torch.tensor(0.0), torch.empty(0, dtype=torch.long)

            # 计算 x_norm 和 w_norm 的归一化
            x_norm = torch.norm(x_valid, p=2, dim=1, keepdim=True).clamp(min=1e-12)
            x_norm = torch.div(x_valid, x_norm)
            w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
            w_norm = torch.div(self.W, w_norm)

            # 计算 cos(theta)
            costh = torch.mm(x_norm, w_norm)

            # 计算 delt_costh 和 cos(theta) - m
            label_view = label_valid.view(-1, 1)
            delt_costh = torch.zeros(costh.size()).scatter_(1, label_view.cpu(), self.m)
            costh_m = costh - delt_costh.to(costh.device)

            # 计算调整后的 cos(theta) * s
            costh_m_s = self.s * costh_m

            # 计算输出和损失
            output = torch.argmax(costh_m_s, dim=1)
            loss = self.ce(costh_m_s, label_valid)

            # 计算准确率
            acc = self.accuracy(output.detach(), label_valid.detach())


        else:
            assert x.size()[0] == label.size()[0]
            assert x.size()[1] == self.embedding_dim

            x = F.normalize(x, dim=1)
            x = self.fc(x)
            _, output = torch.max(x, dim=1)
            # pro_lable把label所有-1的位置换成0
            pro_lable = label.clone()
            pro_lable[pro_lable == -1] = 0
            loss = self.ce(x, pro_lable)
            acc = self.accuracy(output, label)

        return loss, acc, output

