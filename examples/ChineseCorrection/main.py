from __future__ import print_function

import yaml

from examples.ChineseCorrection.subfunction import inference, test, train



if __name__ == '__main__':
    stage = "train"

    arg = "/ssd/zhuang/code/FunASR/examples/ChineseCorrection/args.yaml"
    with open(arg, 'r', encoding='utf-8') as file:
        args = yaml.safe_load(file)

    if stage == "train":
        with open(args["config"], 'r', encoding='utf-8') as file:
            model_configs = yaml.safe_load(file)

        configs = {**model_configs, **args}

        train(configs)

    elif stage == "test":
        with open(args["config"], 'r', encoding='utf-8') as file:
            model_configs = yaml.safe_load(file)

        configs = {**model_configs, **args}
        # configs["ckpt"] = "/ssd/zhuang/code/FunASR/examples/ChineseCorrection/tb_logs/stage_1/version_0/checkpoints/model.pt.avg10"

        test(configs)

    elif stage == "inference":
        with open(args["config"], 'r', encoding='utf-8') as file:
            model_configs = yaml.safe_load(file)

        configs = {**model_configs, **args}
        # configs["ckpt"] = "/ssd/zhuang/code/FunASR/examples/ChineseCorrection/tb_logs/pny2han/version_0/checkpoints/epoch=88-val_loss=7.8599-val_asr_acc=0.9454.ckpt"

        test_data = "那她杀的阿姨在机场的照片"

        inference(test_data, configs)

    pass
