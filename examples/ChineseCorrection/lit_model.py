import torch
import pytorch_lightning as pl
from models import Check, FocalLoss, Single_Check
import torch.nn.functional as F
from examples.ChineseCorrection.utils import CalculatePerformance, cfg2dict, draw_sores

import pytorch_lightning as pl
import torch

import os
import json
import logging

from funasr.register import tables
from funasr.optimizers import optim_classes
from funasr.train_utils.trainer import Trainer
from funasr.schedulers import scheduler_classes
from funasr.train_utils.initialize import initialize
from funasr.download.download_model_from_hub import download_model
from funasr.models.lora.utils import mark_only_lora_as_trainable, lora_summary, loar_wrapper
from funasr.train_utils.set_all_random_seed import set_all_random_seed
from funasr.train_utils.load_pretrained_model import load_pretrained_model
from funasr.utils.misc import prepare_model_dir
from funasr.train_utils.model_summary import model_summary
from funasr import AutoModel

from itertools import chain
from pypinyin import lazy_pinyin, INITIALS, FINALS_TONE3
from  examples.ChineseCorrection.data.dataset import make_pny
from examples.ChineseCorrection.data.data import collator



class LitModel(pl.LightningModule):
    """pytorch lightning 模型"""

    def __init__(self, kwargs: dict):
        super(LitModel, self).__init__()
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        # 初始化参数
        self.kwargs = kwargs

        self.val_f1 = []
        self.val_loss = []
        self.val_asr_acc = []

        self.test_token_num = 0
        self.test_error_num = 0

        torch.set_float32_matmul_precision(kwargs["solver_conf"].get("precision", "medium"))
        self.save_hyperparameters(kwargs)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # 初始化模型
        device = kwargs.get("device", "cuda")
        kwargs["device"] = "cpu"

        self.calculate_performance = CalculatePerformance(kwargs["confusion_conf"]["threshold"])

        # 是否使用lora
        use_lora = kwargs.get("use_lora", False)
        lora_details = kwargs.get("lora_details", None)
        output_dir = kwargs.get("output_dir", "./exp")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if lora_details is not None and use_lora:
            lora_config = os.path.join(output_dir, "lora_config.json")
            with open(lora_details, 'r') as file:
                lora_details = json.load(file)
            lora_exception = lora_details.get("lora_exception", [])
            with open(lora_config, "w") as f:
                json.dump(lora_details, f, indent=4)
            kwargs["lora_details"] = lora_config

        # model = Single_Check(kwargs)

        model = AutoModel(**kwargs)
        # parse kwargs
        kwargs = model.kwargs
        kwargs["device"] = device
        tokenizer = kwargs["tokenizer"]
        frontend = kwargs["frontend"]
        model = model.model
        del kwargs["model"]

        '''freeze_param'''
        freeze_param = kwargs.get("freeze_param", None)
        if freeze_param is not None:
            if "," in freeze_param:
                freeze_param = eval(freeze_param)
            if not isinstance(freeze_param, (list, tuple)):
                freeze_param = (freeze_param,)

            if local_rank == 0:
                print("freeze_param is not None: %s", freeze_param)

            for t in freeze_param:
                for k, p in model.named_parameters():
                    if k.startswith(t + ".") or k == t:
                        p.requires_grad = False

                        if local_rank == 0:
                            print(f"Setting {k}.requires_grad = False")

        ''' mark_only_lora_as_trainable'''
        if use_lora:
            lora_bias = kwargs.get("lora_bias", "none")
            mark_only_lora_as_trainable(model, lora_bias, lora_exception)

        ''' print INFO'''
        if local_rank == 0:
            print(f"{model_summary(model)}")
            if use_lora:
                print(f"{lora_summary(model)}")

        # optim
        if local_rank == 0:
            print("Build optim")
        optim = kwargs.get("optim", "adam")
        assert optim in optim_classes
        optim_class = optim_classes.get(optim)
        # part optim
        if kwargs["scheduler"] == "partition_warmuplr":
            optim_groups = []
            encoder_params = []
            base_params = []
            for name, param in model.named_parameters():
                if 'encoder' in name:
                    encoder_params.append(param)
                else:
                    base_params.append(param)

            assert encoder_params, "Encoder parameters are empty!"
            optim_groups = [
                {'params': encoder_params, 'lr': 1e-5},
                {'params': base_params, 'lr': kwargs.get("optim_conf", {}).get("lr", 1e-3)}
            ]
        else:
            optim_groups = model.parameters()
        optim = optim_class(optim_groups, **kwargs.get("optim_conf"))

        # scheduler
        if local_rank == 0:
            print("Build scheduler")
        scheduler = kwargs.get("scheduler", "warmuplr")
        assert scheduler in scheduler_classes
        scheduler_class = scheduler_classes.get(scheduler)
        scheduler = scheduler_class(optim, **kwargs.get("scheduler_conf"))

        self.model = model
        self.optim = optim
        self.scheduler = scheduler

    def share_step(self, batch, batch_idx):
        loss, result_dict, device_id = self.model(**batch)
        return loss, result_dict

    def training_step(self, batch, batch_idx):
        loss, result_dict = self.share_step(batch, batch_idx)
        asr_acc = result_dict['acc']
        self.log("loss", loss, prog_bar=True, logger=True, sync_dist=True)
        self.log("asr_acc", asr_acc, prog_bar=True, logger=True, sync_dist=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        loss, result_dict = self.share_step(batch, batch_idx)
        asr_acc = result_dict['acc']
        self.val_loss.append(loss)
        self.val_asr_acc.append(asr_acc)
        # self.log("val_loss", loss, prog_bar=True, logger=True, sync_dist=True)
        # return {"val_loss": loss}

    def on_validation_epoch_end(self):
        avg_loss = 0
        avg_asr_acc = 0
        for i in range(len(self.val_loss)):
            avg_asr_acc += self.val_asr_acc[i][0]
            avg_loss += self.val_loss[i][0]
        avg_loss = avg_loss / len(self.val_loss)
        avg_asr_acc = avg_asr_acc / len(self.val_asr_acc)
        self.print('dev: loss: {:.4f}, acc: {:.4f} '.format(avg_loss, avg_asr_acc))
        self.log("val_loss", avg_loss, prog_bar=False, logger=True, sync_dist=True)
        self.log("val_asr_acc", avg_asr_acc, prog_bar=False, logger=True, sync_dist=True)
        return {"val_loss": avg_loss, "val_asr_acc": avg_asr_acc}

    def test_step(self, batch, batch_idx):
        results = self.model.test(**batch)
        total_len = results["total_len"]
        error_num = results["error_num"]

        self.test_token_num += total_len
        self.test_error_num += error_num

        cer = self.test_error_num / self.test_token_num

        # self.print('测试集结果: cer: {:.2f}%'.format(cer * 100))
        self.log("test_cer", cer, on_epoch=True, prog_bar=True)
        return cer

    def forward(self, inference_data):

        ids = self.model.tokenizer.encode(inference_data)
        text = torch.tensor(ids, dtype=torch.int64)

        wrong_pny_list = make_pny(inference_data)
        blank = ['<blank>']
        wrong_pny_list = list(chain(*zip(wrong_pny_list, blank*(len(wrong_pny_list)-1)), [wrong_pny_list[-1]]))
        wrong_pny = self.model.pny_tokenizer.tokens2ids(wrong_pny_list)

        wrong_pny = torch.tensor(wrong_pny, dtype=torch.int64)
        pny_lengths =  torch.tensor([wrong_pny.size(0)], dtype=torch.int32)
        text_lengths =  torch.tensor([text.size(0)], dtype=torch.int32)

        sample = {
            "pny": wrong_pny,
            "pny_lengths": pny_lengths,
            "text": text,
            "text_lengths": text_lengths,
        }

        sample = collator([sample])

        results = self.model.test(**sample)

        return results

    def configure_optimizers(self):
        return {
            "optimizer": self.optim,
            "lr_scheduler": self.scheduler  # 可选
        }

    def optimizer_step(self, epoch=None, batch_idx=None, optimizer=None, optimizer_closure=None):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        lr = self.scheduler.get_lr()[0]
        self.log("lr", lr, prog_bar=False, logger=True, sync_dist=True)
        self.lr_schedulers().step()  # 这会调用 self.scheduler.step()
        # step = self.trainer.global_step + 1
        # rate = self.scheduler.get_lr()[0]
        # for p in optimizer.param_groups:
        #     p['lr'] = rate
        # self.log("lr", rate, prog_bar=False, logger=True, sync_dist=True)
        # optimizer.step(closure=optimizer_closure)

    def load_model(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.load_state_dict(ckpt["state_dict"])
        # ckpt = torch.load(path, map_location='cpu')
        # self.model.load_state_dict(ckpt)

