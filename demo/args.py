import os
import time

from yacs.config import CfgNode as CN

root_dir = os.getcwd()
date = time.strftime("%Y-%m-%d", time.localtime())

# 这一部分是默认参数，临时修改参数请到yml文件里指定
_C = CN()

_C.DATA = CN()
_C.DATA.config = "/data/NAS_PLUS/zhuang/CodeBase/AR_test/ASR_with_prompt/conf/train_transformer.yaml"
_C.DATA.train_data = '/ssd/zhuang/CodeStorage/DARNet/data/ASR/train_phase1/data.list'
_C.DATA.cv_data = '/ssd/zhuang/CodeStorage/DARNet/data/ASR/dev_phase1/data.list'
_C.DATA.test_data = '/ssd/zhuang/CodeStorage/DARNet/data/ASR/dev_phase1/data.list'
_C.DATA.symbol_table = '/ssd/zhuang/CodeStorage/DARNet/data/lang_char.txt'
_C.DATA.pny_table = '/ssd/zhuang/CodeStorage/DARNet/data/lang_pny.txt'
_C.DATA.acc_table = '/ssd/zhuang/CodeStorage/DARNet/data/lang_acc.txt'
_C.DATA.cmvn = None
_C.DATA.non_lang_syms = None
_C.DATA.bpe_model = None
_C.DATA.lfmmi_dir = ''


_C.SOLVER = CN()
# 学习率和损失函数
_C.SOLVER.clip = False
# 训练设置
_C.SOLVER.devices = [2, 3]
_C.SOLVER.precision = "medium"
_C.SOLVER.strategy = 'ddp'  # dp
_C.SOLVER.accelerator = "gpu"
_C.SOLVER.accumulate_grad_batches = 4
_C.SOLVER.gradient_clip = 0.5
_C.SOLVER.max_epochs = 100
_C.SOLVER.train_batch_size = 32
_C.SOLVER.valid_batch_size = 32
_C.SOLVER.test_batch_size = 32

# 保存参数
_C.SAVE = CN()
_C.SAVE.name = date
_C.SAVE.save_top_k = 20
_C.SAVE.every_n_epochs = 1
_C.SAVE.save_dir = "tb_logs"

# 推理参数
_C.INFERENCE = CN()
_C.INFERENCE.use_cuda = True
_C.INFERENCE.inference_type = 'greedy_search'# 'greedy_search' 'beam_search' 对inference起作用, 目前beam search非并行，test会很慢
_C.INFERENCE.beam_width = 3  # beam search 搜索宽度
_C.INFERENCE.beam_nbest = 3  # beam search 保留结果，test阶段强制为1
_C.INFERENCE.c_weight = 1  # 长度惩罚系数
_C.INFERENCE.view_nbest = False  # inference 阶段展示nbest个结果

