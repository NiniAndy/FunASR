import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from yacs.config import CfgNode as CN

from examples.ChineseCorrection.data.data import MyDataModule
from examples.ChineseCorrection.lit_model import LitModel
from examples.ChineseCorrection.utils import init_tokenizer


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
        from config import _C as config
    config.merge_from_file("config.yml")  # 可以修改测试数据集等
    model = LitModel(config)
    model.load_model(check_point)
    # model = model.cuda(config.INFERENCE.gpu) if config.INFERENCE.use_cuda else model
    wrong_index = model(inference_data)
    wrong_index = (wrong_index[:, 1] - 1).tolist() if wrong_index.size(0) != 0 else []
    wrong_index = [i for i in wrong_index if i < len(inference_data)]
    result = []
    for i in range(len(inference_data)):
        if i in wrong_index:
            result.append('\033[31m{}\033[0m'.format(inference_data[i]))
        else:
            result.append(inference_data[i])
    print("".join(result))



def train(kwargs):
    data_module = MyDataModule(kwargs)
    model = LitModel(kwargs)

    bar = CustomProgressBar()

    save_conf = kwargs["save_conf"]
    solver_conf = kwargs["solver_conf"]
    save_file = save_conf.get("save_dir", "tb_logs")
    output_dir = kwargs.get("output_dir", "/ssd/zhuang/code/FunASR/demo/")
    save_path = os.path.join(output_dir, save_file)

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
                         logger=TensorBoardLogger(save_dir = save_path, name=save_conf.get("name", "default"))
                         )

    trainer.fit(model, data_module)


def test(check_point, config_file):
    bar = CustomProgressBar()
    if os.path.exists(config_file):
        print("加载配置文件{}".format(config_file))
        cfgf = open(config_file)
        config = CN().load_cfg(cfgf)
    else:
        print("没有找到目标配置，将加载默认配置")
        from config import _C as config
    config.merge_from_file("config.yml")  # 可以修改测试数据集等
    trainer = pl.Trainer(gpus=config.SOLVER.gpus, strategy=config.SOLVER.accelerator, logger=False,
                         callbacks=[bar])
    # f1_list_all = []
    # precision_list_all = []
    # recall_list_all = []
    # for k in range(10, 80, 10):
    #     config.CONFUSION.max_mask_prob = k / 100
    # f1_list = []
    # precision_list = []
    # recall_list = []
    config, tokenizer = init_tokenizer(config)
    data_module = MyDataModule(config, tokenizer)
    # for i in range(0, 100, 10):
    #     config.SOLVER.threshold = i / 100
    model = LitModel(config)
    model.load_model(check_point)
    result = trainer.test(model, data_module)[0]
    # precision_list.append(result["precision"])
    # recall_list.append(result["recall"])
    # f1_list.append(result["f1"])
    # f1_list_all.append(f1_list)
    # precision_list_all.append(precision_list)
    # recall_list_all.append(recall_list)
    # x = [i / 100 for i in range(0, 100, 10)]
    # if trainer.local_rank == 0:
    #     for f1_list in f1_list_all:
    #         print(f1_list)
    #         plt.plot(x, f1_list)
    #     plt.title("F1-threshold lv1")
    #     plt.xlabel("Threshold")
    #     plt.ylabel("F1")
    #     plt.legend(labels=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
    #     plt.show()
    #
    #     for i in range(7):
    #         plt.plot(recall_list_all[i], precision_list_all[i])
    #     plt.title("P-R lv1")
    #     plt.xlabel("Recall")
    #     plt.ylabel("Precision")
    #     plt.legend(labels=["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7"])
    #     plt.show()
