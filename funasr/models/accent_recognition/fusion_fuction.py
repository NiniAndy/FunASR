import torch.nn as nn
import torch

class FusionFunction(nn.Module):
    def __init__(self, ar_encoder_input_size, accent_size):
        super(FusionFunction, self).__init__()
        self.ar_linear = nn.Sequential(
            nn.Linear(ar_encoder_input_size, ar_encoder_input_size * 2),
            nn.GELU(),
            nn.Linear(ar_encoder_input_size * 2, ar_encoder_input_size * 2),
            nn.GELU(),
            nn.Linear(ar_encoder_input_size * 2, ar_encoder_input_size),
            nn.GELU()
        )

        self.ar_output_layer = nn.Linear(ar_encoder_input_size, accent_size)

        class_count_dict =  {"Mandarin": 370819, "Northeastern": 4843, "Jiang - Huai": 27586, "Southwestern": 45359, "Jiao - Liao": 20268,
                             "Beijing": 2237, "Zhongyuan": 48590, "Ji - Lu": 33861, "Lan - Yin": 20549}
        total = sum(class_count_dict.values())
        class_weights = [total / (accent_size*class_count_dict[k]) for k in class_count_dict.keys()]

        self.ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).cuda(), ignore_index=-1)

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

    def forward(self, x_hyp, x_a, y_dal):
        x_f = self.fusion(x_hyp, x_a)
        ar_dmm = self.ar_linear(x_f)
        y_dal_hat = self.ar_output_layer(ar_dmm)
        y_dal = y_dal.squeeze(1).long()
        loss_dal = self.ce_loss(y_dal_hat, y_dal)

        _, output = torch.max(y_dal_hat, dim=1)
        acc_dal = self.accuracy(output, y_dal)

        return ar_dmm, y_dal_hat, loss_dal, acc_dal

    def inference(self, x_hyp, x_a):
        x_f = self.fusion(x_hyp, x_a)
        ar_dmm = self.ar_linear(x_f)
        y_dal_hat = self.ar_output_layer(ar_dmm)
        _, output = torch.max(y_dal_hat, dim=1)
        return ar_dmm, output


class AddFusionFunction(FusionFunction):
    def __init__(self, x_hyp_size, x_a_size, ar_encoder_input_size, accent_size):
        super(AddFusionFunction, self).__init__(ar_encoder_input_size, accent_size)

        self.hyp_gru = nn.GRU(input_size=x_hyp_size, hidden_size=ar_encoder_input_size, num_layers=1, batch_first=True, bidirectional=False)
        self.a_gru = nn.GRU(input_size=x_a_size, hidden_size=ar_encoder_input_size, num_layers=1, batch_first=True, bidirectional=False)

    def fusion(self, x_hyp, x_a):
        # 只取最后一个时间步的输出
        _, x_hyp = self.hyp_gru(x_hyp)
        _, x_a = self.a_gru(x_a) # [layer_num, batch_size, hidden_size]
        x_hyp = x_hyp.squeeze(0)
        x_a = x_a.squeeze(0)
        return x_hyp + x_a

class ConcatFusionFunction(FusionFunction):
    def __init__(self, x_hyp_size, x_a_size, ar_encoder_input_size, accent_size):
        super(ConcatFusionFunction, self).__init__(ar_encoder_input_size, accent_size)
        cat_size = x_hyp_size + x_a_size
        self.gru = nn.GRU(cat_size, ar_encoder_input_size)

    def fusion(self, x_hyp, x_a):
        x = torch.cat((x_hyp, x_a), dim=1)
        _, x = self.gru(x)
        return x