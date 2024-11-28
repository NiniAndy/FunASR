import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from examples.pny2hanSLM.data.data import MyDataModule
from examples.pny2hanSLM.lit_model import LitModel


class CustomProgressBar(TQDMProgressBar):
    def __init__(self):
        super(CustomProgressBar, self).__init__()

    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items


def inference(inference_data, kwargs):
    data_module = MyDataModule(kwargs)
    model = LitModel(kwargs)
    model.load_model(kwargs["ckpt"])
    output = model(inference_data)
    print (output)



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


def test(kwargs):

    data_module = MyDataModule(kwargs)
    model = LitModel(kwargs)

    bar = CustomProgressBar()

    save_conf = kwargs["save_conf"]
    solver_conf = kwargs["solver_conf"]
    save_file = save_conf.get("save_dir", "tb_logs")
    output_dir = kwargs.get("output_dir", "/ssd/zhuang/code/FunASR/demo/")
    save_path = os.path.join(output_dir, save_file)

    trainer = pl.Trainer(devices=solver_conf.get("devices", [0]),
                         accelerator=solver_conf.get("accelerator", "gpu"),
                         logger=False,
                         callbacks=[bar])

    model.load_model(kwargs["ckpt"])
    result = trainer.test(model, data_module)[0]

