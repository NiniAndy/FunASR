from __future__ import print_function

import yaml


from subfunction import inference, test, train, auto_load, resume_train

# 测试直接使用文字作为token

if __name__ == '__main__':
    stage = "train"
    ckpt_dir = "/data/NAS_PLUS/zhuang/CodeBase/AR_test/Transformer_PNY/tb_logs/hubert_extractor_pny/version_0/"
    # 如果是inference/test，必须提供以下两个文件
    # ckpt_dir = "/data/NAS_PLUS/zhuang/CodeBase/Conformer_Baseline/tb_logs/aishell_conformer/version_3/"
    monitor = "val_loss"
    mode = "min"
    resume = False

    arg = "/ssd/zhuang/code/FunASR/demo/args.yaml"
    with open(arg, 'r', encoding='utf-8') as file:
        args = yaml.safe_load(file)

    model_config = args["config"]

    with open(model_config, 'r', encoding='utf-8') as file:
        model_configs = yaml.safe_load(file)

    configs = {**model_configs, **args}

    if stage == "train":
        train(configs)

    pass

    # if stage == "train":
    #     args.merge_from_file("args.yml")
    #     with open(args.DATA.config, 'r') as fin:
    #         kwargs = yaml.load(fin, Loader=yaml.FullLoader)
    #
    #     if resume:
    #         config_file, check_point = auto_load(ckpt_dir, monitor, mode)
    #         resume_train(args, check_point, config_file)
    #     else:
    #         train(args, kwargs)
    #
    #
    # elif stage == "test":
    #     # config_file, check_point = auto_load(ckpt_dir, monitor, mode)
    #     config_file = None
    #     check_point = ckpt_dir
    #     test(check_point, config_file)
    # elif stage == "inference":
    #     config_file, check_point = auto_load(ckpt_dir, monitor, mode)
    #     inference(check_point, config_file, "1.wav")
    # else:
    #     print("Unsupported {} stage".format(kwargs.stage))
