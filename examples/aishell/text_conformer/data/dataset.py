from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch
from examples.ChineseCorrection.utils import GenerateConfusion
from functools import partial
import json
import numpy as np
from tqdm import tqdm
import os
from itertools import chain

from pypinyin import lazy_pinyin, INITIALS, FINALS_TONE3

def make_pny(sentence):
    origin_initials = lazy_pinyin(sentence, INITIALS)  # 声母
    origin_vowel = lazy_pinyin(sentence, FINALS_TONE3)  # 韵母和音调
    origin_pinyin = [origin_initials[j] + origin_vowel[j] for j in range(len(origin_initials))]
    return origin_pinyin


class OrdinaryDataSet(Dataset):
    def __init__(self, pre_trained_name, data_path):
        super(OrdinaryDataSet, self).__init__()

        self.tokenizer = BertTokenizer.from_pretrained(pre_trained_name)
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        wrong_text = self.data[item]["original_text"]
        correct_text = self.data[item]["correct_text"]
        wrong_ids = self.data[item]["wrong_ids"]

        if wrong_text[-1] == "\u3000":
            wrong_text = wrong_text[:-1]
            correct_text = correct_text[:-1]

        sentence_len = len(correct_text)
        correct_text = " ".join(correct_text)
        wrong_text = " ".join(wrong_text)

        correct_text_ids = self.tokenizer(correct_text, return_tensors="pt")["input_ids"]
        wrong_text_ids = self.tokenizer(wrong_text, return_tensors="pt")["input_ids"]

        wrong_list = [0] * sentence_len
        for index in wrong_ids:
            wrong_list[index] = 1
        wrong_list.insert(0, 0)
        wrong_list.append(0)
        wrong_list = torch.tensor(wrong_list).type_as(correct_text_ids)

        if wrong_text_ids.size(1) != wrong_list.size(0):
            print(wrong_text)

        return wrong_text_ids, wrong_list, correct_text_ids


class ConfusionDataSet(Dataset):
    def __init__(self, tokenizer, pny_tokenizer, correct_path, confusion_config, mode):
        super(ConfusionDataSet, self).__init__()
        self.int_pad_value = -1
        self.float_pad_value = 0.0

        self.tokenizer = tokenizer
        self.pny_tokenizer  = pny_tokenizer
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

        sentence_len = len(correct_text)
        ids = self.tokenizer.encode(correct_text)
        text = torch.tensor(ids, dtype=torch.int64)

        wrong_pny_list = make_pny(wrong_text)
        blank = ['<blank>']
        wrong_pny_list = list(chain(*zip(wrong_pny_list, blank*(len(wrong_pny_list)-1)), [wrong_pny_list[-1]]))
        wrong_pny = self.pny_tokenizer.tokens2ids(wrong_pny_list)

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





class ContrastDataSet_test(Dataset):
    def __init__(self, tokenizer, correct_path, error_path, mode):
        super(ContrastDataSet_test, self).__init__()
        self.tokenizer = tokenizer
        self.correct_path = correct_path
        self.error_path = error_path

        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            print("加载", mode, "数据集：", correct_path)

        buffer = 1024 * 1024

        with open(correct_path) as f:
            length = sum(x.count('\n') for x in iter(partial(f.read, buffer), ''))
        self.correct_data = np.empty(length, dtype="<U64")
        with open(correct_path, "r", encoding="utf-8") as f:
            i = 0
            for line in tqdm(f, total=length, desc="Processing lines"):
                if line != "\n" and len(line[:-1]) < 64:
                    self.correct_data[i] = line[:-1]
                    i += 1
        self.correct_data = self.correct_data[:i]

        with open(error_path, "r", encoding="utf-8") as f:
            length = sum(x.count('\n') for x in iter(partial(f.read, buffer), ''))
        self.error_data = np.empty(length, dtype="<U64")
        with open(error_path, "r", encoding="utf-8") as f:
            i = 0
            for line in tqdm(f, total=length, desc="Processing lines"):
                if line != "\n" and len(line[:-1]) < 64:
                    self.error_data[i] = line[:-1]
                    i += 1
        self.error_data = self.error_data[:i]

    def __len__(self):
        return self.correct_data.size

    def __getitem__(self, item):
        correct_text = self.correct_data[item]
        wrong_text = self.error_data[item]
        wrong_list = [0 if char1 == char2 else 1 for char1, char2 in zip(correct_text, wrong_text)]

        correct_text_tokens, correct_text_ids = self.tokenizer.tokenize(correct_text)
        wrong_text_tokens, wrong_text_ids = self.tokenizer.tokenize(wrong_text)
        correct_text_ids = torch.tensor(correct_text_ids).unsqueeze(0)
        wrong_text_ids = torch.tensor(wrong_text_ids).unsqueeze(0)

        wrong_list = torch.tensor(wrong_list).type_as(correct_text_ids)

        return wrong_text_ids, wrong_list, correct_text_ids
