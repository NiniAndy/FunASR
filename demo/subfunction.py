import glob
import os
import yaml


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from yacs.config import CfgNode as CN

from data.dataloader import MyDataModule
from lit_model import LitModel
from utils.functions import calculate_config, init_vocab


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super(CustomProgressBar, self).__init__()

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def inference(check_point, config_file, inference_data):
    if os.path.exists(config_file):
        print("加载配置文件{}".format(config_file))
        cfgf = open(config_file)
        config = CN().load_cfg(cfgf)
    else:
        print("没有找到目标配置，将加载默认配置")
        from args import _C as config
    config.merge_from_file("args.yml")  # 可以修改测试数据集噪音等
    model = LitModel(config)
    model.load_model(check_point)

    hyp = model(inference_data)
    print(hyp)


def auto_load(ckpt_dir, monitor, mode):
    config_file = glob.glob(f"{ckpt_dir}/*.yaml")
    assert len(config_file) > 0, f"No config file in {ckpt_dir}"
    config_file = config_file[0]
    check_point = glob.glob(f"{ckpt_dir}/checkpoints/*.ckpt")
    if len(check_point) > 1:
        ckpt_score_list = []
        # 自动寻找最优加载
        for ckpt_name in check_point:
            assert monitor in ckpt_name, f"{ckpt_name} dose not have {monitor}"
            ckpt_score = ckpt_name.split(f"{monitor}=")[1].split("-")[0]
            ckpt_score = float(ckpt_score.split(".ckpt")[0])
            ckpt_score_list.append(ckpt_score)
        if mode == "min":
            check_point = [ckpt_name for ckpt_name in check_point if str(min(ckpt_score_list)) in ckpt_name]
        else:
            check_point = [ckpt_name for ckpt_name in check_point if str(max(ckpt_score_list)) in ckpt_name]
        check_point = check_point[0]
    else:
        check_point = check_point[0]

    return config_file, check_point


def resume_train(args, check_point, config_file):
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print("Resume training from {}".format(config_file))
        print("Resume training from {}".format(check_point))
    with open(config_file, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    args.merge_from_file("args.yml")

    args.symbol_table = configs["symbol_table"]
    args.pny_table = configs["pny_table"]
    args.acc_table = configs["acc_table"]

    model = LitModel(args, configs)
    data_module = MyDataModule(args, configs)
    bar = CustomProgressBar()
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.4f}-{val_cer:.4f}',
                                          monitor='val_loss',
                                          save_top_k=args.SAVE.save_top_k,
                                          mode='min',
                                          every_n_epochs=args.SAVE.every_n_epochs)
    trainer = pl.Trainer(devices=args.SOLVER.devices,
                         accelerator=args.SOLVER.accelerator,
                         accumulate_grad_batches=args.SOLVER.accumulate_grad_batches,
                         strategy=args.SOLVER.strategy,
                         max_epochs=args.SOLVER.max_epochs,
                         callbacks=[checkpoint_callback, bar],
                         logger=TensorBoardLogger(args.SAVE.save_dir, name=args.SAVE.name))
    trainer.fit(model, data_module, ckpt_path=check_point)


def train(args, configs):
    configs = calculate_config(args, configs)
    model = LitModel(args, configs)
    data_module = MyDataModule(args, configs)

    bar = CustomProgressBar()
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.4f}-{val_cer:.4f}',
                                          monitor='val_loss',
                                          save_top_k=args.SAVE.save_top_k,
                                          mode='min',
                                          every_n_epochs=args.SAVE.every_n_epochs)
    trainer = pl.Trainer(devices=args.SOLVER.devices,
                         accelerator=args.SOLVER.accelerator,
                         accumulate_grad_batches=args.SOLVER.accumulate_grad_batches,
                         strategy=args.SOLVER.strategy,
                         max_epochs=args.SOLVER.max_epochs,
                         callbacks=[checkpoint_callback, bar],
                         logger=TensorBoardLogger(args.SAVE.save_dir, name=args.SAVE.name))
    trainer.fit(model, data_module)


def test(check_point, config_file):
    bar = CustomProgressBar()
    if config_file is not None:
        if os.path.exists(config_file):
            # print("加载配置文件{}".format(config_file))
            cfgf = open(config_file)
            config = CN().load_cfg(cfgf)
        else:
            print("没有找到目标配置，将加载默认配置")
            from args import _C as config
            config = calculate_config(config)
    else:
        print("没有找到目标配置，将加载默认配置")
        from args import _C as config
        config = calculate_config(config)
    config.merge_from_file("args.yml")  # 可以修改测试数据集噪音等
    model = LitModel(config)
    data_module = MyDataModule(config)
    model.load_model(check_point)
    trainer = pl.Trainer(devices=[3], accelerator="gpu", logger=False, callbacks=[bar], max_epochs=-1)
    cer = trainer.test(model, data_module)
    print("cer", cer)
