import copy
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from data import dataset
from data.dataset import collate_fn
from wenet.utils.file_utils import read_lists, read_non_lang_symbols
from utils.functions import init_vocab


class MyDataModule(pl.LightningDataModule):
    def __init__(self, args, configs):
        super(MyDataModule, self).__init__()

        self.train_conf = configs['dataset_conf']
        self.cv_conf = copy.deepcopy(self.train_conf)
        self.cv_conf['speed_perturb'] = False
        self.cv_conf['spec_aug'] = False
        self.cv_conf['spec_sub'] = False
        self.cv_conf['spec_trim'] = False
        self.cv_conf['shuffle'] = False

        self.test_conf = copy.deepcopy(self.cv_conf)

        self.args = args
        self.symbol_table, _, self.acc_table, _, self.pny_table, _ = init_vocab(args)

        self.bpe_model = args.DATA.bpe_model
        self.non_lang_syms = read_non_lang_symbols(args.DATA.non_lang_syms)

    def prepare_data(self):
        # 通过URL下载数据
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            train_lists = read_lists(self.args.DATA.train_data)
            val_lists = read_lists(self.args.DATA.cv_data)

            self.train_dataset = dataset.SpectrogramDataset(
                self.train_conf,
                train_lists,
                self.symbol_table,
                self.pny_table,
                self.acc_table,
                self.bpe_model,
                self.non_lang_syms)
            self.val_dataset = dataset.SpectrogramDataset(
                self.cv_conf,
                val_lists,
                self.symbol_table,
                self.pny_table,
                self.acc_table,
                self.bpe_model,
                self.non_lang_syms)
            if int(os.environ.get('LOCAL_RANK', 0)) == 0:
                print('训练集长度: ', len(self.train_dataset))
                print('验证集长度：', len(self.val_dataset))

        if stage == 'test' or stage is None:

            test_lists = read_lists(self.args.DATA.test)
            self.test_dataset = dataset.SpectrogramDataset(
                self.test_conf,
                test_lists,
                self.symbol_table,
                self.pny_table,
                self.acc_table,
                self.bpe_model,
                self.non_lang_syms)

            print('测试集长度：', len(self.test_file_list))


    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.args.SOLVER.train_batch_size,
                          num_workers=24,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.args.SOLVER.valid_batch_size,
                          num_workers=24,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.args.SOLVER.test_batch_size,
                          num_workers=24,
                          collate_fn=collate_fn)
