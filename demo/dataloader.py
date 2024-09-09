import copy
import os
import logging

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from funasr.register import tables




class MyDataModule(pl.LightningDataModule):
    def __init__(self, kwargs):
        super(MyDataModule, self).__init__()

        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # build tokenizer
        tokenizer = kwargs.get("tokenizer", None)
        if tokenizer is not None:
            tokenizer_class = tables.tokenizer_classes.get(tokenizer)
            tokenizer = tokenizer_class(**kwargs.get("tokenizer_conf", {}))
            kwargs["token_list"] = (
                tokenizer.token_list if hasattr(tokenizer, "token_list") else None
            )
            kwargs["token_list"] = (
                tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else kwargs["token_list"]
            )
            vocab_size = len(kwargs["token_list"]) if kwargs["token_list"] is not None else -1
            if vocab_size == -1 and hasattr(tokenizer, "get_vocab_size"):
                vocab_size = tokenizer.get_vocab_size()
        else:
            vocab_size = -1
        kwargs["tokenizer"] = tokenizer

        # build frontend
        frontend = kwargs.get("frontend", None)
        kwargs["input_size"] = None
        if frontend is not None:
            frontend_class = tables.frontend_classes.get(frontend)
            frontend = frontend_class(**kwargs.get("frontend_conf", {}))
            kwargs["input_size"] = (
                frontend.output_size() if hasattr(frontend, "output_size") else None
            )
        kwargs["frontend"] = frontend

        if local_rank == 0:
            print ("Build dataloader")

        self.dataset_class = tables.dataset_classes.get(kwargs.get("dataset", "AudioDataset"))
        self.kwargs = kwargs


    def prepare_data(self):
        # 通过URL下载数据
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = self.dataset_class(
                self.kwargs.get("train_data_set_list"),
                is_training=True,
                **self.kwargs.get("dataset_conf"),
                **self.kwargs
            )

            self.val_dataset = self.dataset_class(
                self.kwargs.get("valid_data_set_list"),
                is_training=False,
                **self.kwargs.get("dataset_conf"),
                **self.kwargs
            )

            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print('训练集长度: ', len(self.train_dataset))
                print('验证集长度：', len(self.val_dataset))

        if stage == 'test' or stage is None:
            self.test_dataset = self.dataset_class(
                self.kwargs.get("test_data_set_list"),
                is_training=False,
                **self.kwargs.get("dataset_conf"),
                **self.kwargs
            )


    def train_dataloader(self):
        batch_sampler = "BatchSampler"
        batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
        batch_sampler = batch_sampler_class(self.train_dataset, start_step=0, **self.kwargs.get("dataset_conf"))
        return DataLoader(self.train_dataset,
                          collate_fn=self.train_dataset.collator,
                          batch_size=self.kwargs["solver_conf"]["train_batch_size"],
                          num_workers=self.kwargs["solver_conf"]["num_workers"],)

    def val_dataloader(self):
        batch_sampler = "BatchSampler"
        batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
        batch_sampler = batch_sampler_class(self.val_dataset, start_step=0, **self.kwargs.get("dataset_conf"))
        return DataLoader(self.val_dataset,
                          collate_fn=self.val_dataset.collator,
                          batch_size=self.kwargs["solver_conf"]["valid_batch_size"],
                          num_workers=self.kwargs["solver_conf"]["num_workers"],)


    def test_dataloader(self):
        batch_sampler = "BatchSampler"
        batch_sampler_class = tables.batch_sampler_classes.get(batch_sampler)
        self.kwargs["dataset_conf"]["batch_size"] = 1
        batch_sampler = batch_sampler_class(self.test_dataset, start_step=0, **self.kwargs.get("dataset_conf"))
        return DataLoader(self.test_dataset,
                          collate_fn=self.test_dataset.collator,
                          batch_size=1,
                          num_workers=1,)
