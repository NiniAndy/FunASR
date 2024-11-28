from __future__ import print_function

import yaml

from examples.ChineseCorrection.subfunction import inference, test, train


if __name__ == '__main__':
    stage = "train"
    ckpt_dir = "/data/NAS_PLUS/zhuang/CodeBase/AR_test/Transformer_PNY/tb_logs/hubert_extractor_pny/version_0/"
    # 如果是inference/test，必须提供以下两个文件
    # ckpt_dir = "/data/NAS_PLUS/zhuang/CodeBase/Conformer_Baseline/tb_logs/aishell_conformer/version_3/"
    monitor = "val_loss"
    mode = "min"
    resume = False

    arg = "/ssd/zhuang/code/FunASR/examples/ChineseCorrection/args.yaml"
    with open(arg, 'r', encoding='utf-8') as file:
        args = yaml.safe_load(file)

    model_config = args["config"]

    with open(model_config, 'r', encoding='utf-8') as file:
        model_configs = yaml.safe_load(file)

    configs = {**model_configs, **args}

    if stage == "train":
        train(configs)

    pass
