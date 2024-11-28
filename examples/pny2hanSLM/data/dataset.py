import os
import random
from functools import partial
from itertools import chain

import numpy as np
import torch
from pypinyin import lazy_pinyin, INITIALS, FINALS_TONE3
from torch.utils.data import Dataset
from tqdm import tqdm

from examples.ChineseCorrection.utils import GenerateConfusion


def make_pny(sentence):
    origin_initials = lazy_pinyin(sentence, INITIALS)  # 声母
    origin_vowel = lazy_pinyin(sentence, FINALS_TONE3)  # 韵母和音调
    origin_pinyin = [origin_initials[j] + origin_vowel[j] for j in range(len(origin_initials))]
    return origin_pinyin




class ConfusionDataSet(Dataset):
    def __init__(self, tokenizer, correct_path, confusion_config, mode):
        super(ConfusionDataSet, self).__init__()
        self.int_pad_value = -1
        self.float_pad_value = 0.0

        self.tokenizer = tokenizer
        self.generate = GenerateConfusion(confusion_config)
        self.correct_path = correct_path

        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print("加载", mode, "数据集：", correct_path)

        buffer = 1024 * 1024
        with open(correct_path) as f:
            length = sum(x.count('\n') for x in iter(partial(f.read, buffer), ''))

        self.correct_data = np.empty(length, dtype="<U64")
        # self.wrong_data = np.empty(length, dtype="<U64")
        # self.wrong_ids_data = np.empty(length, dtype="<U32")

        with open(correct_path, "r", encoding="utf-8") as f:
            i = 0
            for line in tqdm(f, total=length, desc="Processing lines"):
                if line != "\n" and len(line[:-1]) < 64:
                    self.correct_data[i] = line[:-1]
                    i += 1
        self.correct_data = self.correct_data[:i]

    def __len__(self):
        return self.correct_data.size

    def __getitem__(self, item):
        correct_text = self.correct_data[item]
        if len(correct_text.split()) >1:
            correct_text = correct_text.split()[1 :]
            correct_text = "".join(correct_text)

        try:
            wrong_text, wrong_ids = self.generate.generate(correct_text)
        except:
            wrong_text = correct_text
            wrong_ids = []
        else:
            pass

        ids = self.tokenizer.encode(correct_text)
        text = torch.tensor(ids, dtype=torch.int64)

        wrong_pny_list = make_pny(wrong_text)

        # 随机进入删除
        delete_flag = random.random()
        if delete_flag > 0.9 and len(wrong_pny_list) > 5:
            wrong_pny_list = wrong_pny_list[:-1]
            num_to_delete = random.randint(1, 2)
            indices_to_delete = random.sample(range(len(wrong_pny_list)), num_to_delete)
            # 按索引删除元素，从较大索引开始，避免索引偏移问题
            for index in sorted(indices_to_delete, reverse=True):
                del wrong_pny_list[index]

        blank = ['<blank>']
        wrong_pny_list = list(chain(*zip(wrong_pny_list, blank*(len(wrong_pny_list)-1)), [wrong_pny_list[-1]]))
        wrong_pny = self.tokenizer.tokens2ids(wrong_pny_list)

        wrong_pny = torch.tensor(wrong_pny, dtype=torch.int64)
        pny_lengths =  torch.tensor([wrong_pny.size(0)], dtype=torch.int32)
        text_lengths =  torch.tensor([text.size(0)], dtype=torch.int32)

        sample = {
            "pny": wrong_pny,
            "pny_lengths": pny_lengths,
            "text": text,
            "text_lengths": text_lengths,
        }

        return sample


