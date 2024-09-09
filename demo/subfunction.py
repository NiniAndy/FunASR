import glob
import os
import yaml


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from yacs.config import CfgNode as CN

from lit_model import LitModel
from dataloader import MyDataModule



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
        kwargs = yaml.load(fin, Loader=yaml.FullLoader)
    args.merge_from_file("args.yml")

    args.symbol_table = kwargs["symbol_table"]
    args.pny_table = kwargs["pny_table"]
    args.acc_table = kwargs["acc_table"]

    model = LitModel(args, kwargs)
    data_module = MyDataModule(args, kwargs)
    bar = CustomProgressBar()
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.4f}-{val_asr_acc:.4f}',
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


def train(kwargs):
    model = LitModel(kwargs)
    data_module = MyDataModule(kwargs)

    bar = CustomProgressBar()
    save_conf = kwargs["save_conf"]
    solver_conf = kwargs["solver_conf"]

    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.4f}-{val_asr_acc:.4f}',
                                          monitor='val_loss',
                                          save_top_k=save_conf.get("save_top_k", 10),
                                          mode='min',
                                          every_n_epochs=save_conf.get("every_n_epochs", 1)
                                          )

    trainer = pl.Trainer(devices=solver_conf.get("devices", [0]),
                         accelerator=solver_conf.get("accelerator", "gpu"),
                         accumulate_grad_batches=solver_conf.get("accumulate_grad_batches", 1),
                         strategy=solver_conf.get("strategy", "ddp"),
                         max_epochs=solver_conf.get("max_epochs", 100),
                         callbacks=[checkpoint_callback, bar],
                         logger=TensorBoardLogger(os.path.join("/ssd/zhuang/code/FunASR/demo/", save_conf.get("save_dir", "tb_logs")),
                                                  name=save_conf.get("name", "default"))
                         )

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
