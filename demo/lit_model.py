import pytorch_lightning as pl
import torch
import torch.optim as optim
from torch import nn
import os

from wenet.utils.init_model import init_model
# from utils.token_utils import CalculatePerformance
from wenet.utils.scheduler import WarmupLR


class LitModel(pl.LightningModule):
    """pytorch lightning 模型"""

    def __init__(
            self,
            args: object,
            configs: dict):
        super(LitModel, self).__init__()
        # 初始化参数
        self.configs = configs

        self.val_cer = []
        self.val_loss = []

        torch.set_float32_matmul_precision(args.SOLVER.precision)
        self.save_hyperparameters(configs)
        # self.calculate_performance = CalculatePerformance(configs)

        # 初始化模型
        self.model = init_model(configs)
        self.step = 0
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print(self.model)

        scheduler_conf = {
            "warmup_steps": configs["scheduler_conf"]["warmup_steps"],
            "base_lrs": [configs["optim_conf"]["lr"]]
        }
        self.lr_scheduler = WarmupLR(**scheduler_conf)

        # 初始化模型参数
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def share_step(self, batch, batch_idx):
        key, feats, target, feats_lengths, target_lengths, detail_info = batch
        loss_dict = self.model(feats, feats_lengths, target, target_lengths, detail_info)
        loss = loss_dict['loss']
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.share_step(batch, batch_idx)
        cer = loss_dict['cer_att']
        self.log("loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("cer", cer, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss}
        # self.log("cer", cer * 100, prog_bar=True, logger=True, sync_dist=True)
        # return {"loss": loss, "train_cer": cer}

    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.share_step(batch, batch_idx)
        cer = loss_dict['cer_att']
        self.val_loss.append(loss)
        self.val_cer.append(cer)
        # self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        # return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = 0
        avg_cer = 0
        for i in range(len(self.val_loss)):
            avg_cer += self.val_cer[i]
            avg_loss += self.val_loss[i]
        avg_loss = avg_loss / len(self.val_loss)
        avg_cer = avg_cer / len(self.val_cer)
        self.print('dev: loss: {:.4f}, cer: {:.4f} '.format(avg_loss, avg_cer))
        self.log("val_loss", avg_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_cer", avg_cer, prog_bar=False, logger=True, sync_dist=True)
        return {"val_loss": avg_loss, "val_cer": avg_cer}

    def test_step(self, batch, batch_idx):
        self.step += 1
        inputs, input_len, tgt, tgt_len = batch
        hyps, _ = self.model.ctc_prefix_beam_search(
            inputs,
            input_len,
            beam_size=10,
            decoding_chunk_size=-1,
            num_decoding_left_chunks=-1,
            simulate_streaming=False,
            context_graph=None)
        hyps = list(hyps)
        cer, sub_dict = self.calculate_performance.calculate_cer(hyps, tgt[0].tolist(), self.config.eos_id)
        # self.print('测试集结果: cer: {:.2f}%'.format(cer * 100))
        self.log("test_cer", cer, on_epoch=True, prog_bar=True)
        return cer

    def forward(self, audio_path, inference_data):
        """
        推理部分
        Args:
            inference_data: 音频信号
            audio_path: 音频文件
            para_args: 载入参数
        Returns:
        """
        pass

    def configure_optimizers(self):
        if self.configs['optim'] == 'adamw':
            optimizer = optim.AdamW(self.parameters(), **self.configs['optim_conf'])
        else:
            optimizer = optim.Adam(self.parameters(), **self.configs['optim_conf'])
        return optimizer

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_closure=None):
        step = self.trainer.global_step + 1
        rate = self.lr_scheduler.get_lr(step)[0]
        for p in optimizer.param_groups:
            p['lr'] = rate
        self.log("lr", rate, prog_bar=False, logger=True, sync_dist=True)
        optimizer.step(closure=optimizer_closure)

    def load_model(self, path):
        # ckpt = torch.load(path, map_location='cpu')
        # self.load_state_dict(ckpt["state_dict"])
        ckpt = torch.load(path, map_location='cpu')
        self.model.load_state_dict(ckpt)
