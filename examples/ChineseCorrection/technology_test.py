from transformers import BertModel, BertTokenizer, AutoConfig, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertEmbeddings, BertEncoder, BertPooler, BertOnlyMLMHead
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import OrderedDict
from config import _C as config
import json
from pypinyin import lazy_pinyin, INITIALS, FINALS_TONE3, FINALS
from tqdm import tqdm
from multiprocessing import Pool, Queue
from collections import Counter
# import Levenshtein as Lev
from utils import GenerateConfusion

premodel = BertModel.from_pretrained("save/PreTrained/model/chinese-roberta-wwm-ext/")
tokenizer = BertTokenizer.from_pretrained("save/PreTrained/tokenizer/chinese-roberta-wwm-ext/")
sentence_1 = "我们去了天安门"
sentence_2 = "这个数据集有问题"
out_1 = tokenizer(sentence_1, return_tensors="pt")
out_2 = tokenizer(sentence_2, return_tensors="pt")
bert_config = AutoConfig.from_pretrained(config.PRETRAINED.config)
a = premodel(out_1["input_ids"], output_attentions=True)
b = premodel(out_2["input_ids"], output_attentions=True)
print(out_1["input_ids"])
# 统计同音字及频率


# with open("/data/nas/qian/News2016/train_lab.txt", "r") as f:
#     data = f.readlines()
#     # data = data[:1000000]
#
#
# def get_tone_set(tmp_data):
#     initials_list = set()
#     vowel_list = set()
#     for i in tqdm(tmp_data):
#         if i != "\n":
#             i = i[:-1]
#             Initials = lazy_pinyin(i, INITIALS, errors="")
#             Vowel = lazy_pinyin(i, FINALS, errors="")
#             # tone = [Initials[j] + Vowel[j] for j in range(len(Initials))]
#             # i = [j for j in i]
#             # assert len(i) == len(tone)
#             # for j in range(len(tone)):
#             #     try:
#             #         tone_set[tone[j]][i[j]] += 1
#             #     except KeyError:
#             #         try:
#             #             a = tone_set[tone[j]]
#             #         except KeyError:
#             #             tone_set[tone[j]] = {}
#             #             tone_set[tone[j]][i[j]] = 1
#             #         else:
#             #             tone_set[tone[j]][i[j]] = 1
#
#             # tone_set[tone[j]] = tone_set[tone[j]] + [i[j]]
#             initials_list |= set(Initials)
#             vowel_list |= set(Vowel)
#
#             # tone_set[tone[j]] = list(set(tone_set[tone[j]]))
#             pass
#     # print(initials_list)
#     # print(vowel_list)
#     return initials_list, vowel_list
#
#
# total_number = 30
# p = Pool(total_number)
# per_number = len(data) // total_number + 1
# result = []
# for i in range(total_number):
#     tmp_data = data[i * per_number:(i + 1) * per_number]
#     result.append(p.apply_async(get_tone_set, args=(tmp_data,)))
#
# p.close()
# p.join()
#
# initials_list = set()
# vowel_list = set()
# for i in result:
#     i, j = i.get()
#     initials_list |= i
#     vowel_list |= j
#
# print(len(initials_list))
# print(len(vowel_list))
# print(list(initials_list))
# print(list(vowel_list))

# with open("dataset/pinyin/initials_list.txt", "w", encoding="utf-8") as f:
#     for i in initials_list:
#         f.write(i + "\n")
# with open("dataset/pinyin/vowel_list.txt", "w", encoding="utf-8") as f:
#     for i in vowel_list:
#         f.write(i + "\n")

# tone_set = {}
# for i in result:
#     i = i.get()
#     for j in i:
#         pass
#         for k in i[j]:
#             try:
#                 tone_set[j][k] = tone_set[j][k] + i[j][k]
#             except KeyError:
#                 try:
#                     a = tone_set[j]
#                 except KeyError:
#                     tone_set[j] = {}
#                     tone_set[j][k] = i[j][k]
#                 else:
#                     tone_set[j][k] = i[j][k]
# print(len(tone_set))
# new_set = {}
# for i, j in tone_set.items():
#     word_sorted = sorted(j.items(), key=lambda item: item[1], reverse=True)
#     word_list = [i[0] for i in word_sorted]
#     freq_list = [i[1] for i in word_sorted]
#     total_num = sum(freq_list)
#     freq_list = [u / total_num for u in freq_list]
#     new_set[i] = {}
#     new_set[i]["word_list"] = word_list
#     new_set[i]["freq_list"] = freq_list
#     new_set[i]["total_num"] = total_num
#
#
# with open("initials_list.json", "w", encoding="utf-8") as f:
#     json.dump(new_set, f, indent=4, ensure_ascii=False)

# initials_list = ["b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "zh", "ch", "sh", "r", "z",
# "c", "s"]


# with open("dataset/pinyin/initials_list.txt", "r", encoding="utf-8") as f:
#     initials_list = f.readlines()
# with open("dataset/pinyin/vowel_list.txt", "r", encoding="utf-8") as f:
#     vowel_list = f.readlines()
#
# initials_list = [i[:-1] for i in initials_list]
# vowel_list = [i[:-1] for i in vowel_list]
#
# initials_distence_matrix = {}
# for i in initials_list:
#     initials_distence_matrix[i] = {}
#     distance_1 = []
#     distance_other = []
#     for j in initials_list:
#         if Lev.distance(i, j) == 1:
#             distance_1.append(j)
#         elif Lev.distance(i, j) > 1:
#             distance_other.append(j)
#     initials_distence_matrix[i]["1"] = distance_1
#     initials_distence_matrix[i]["other"] = distance_other
# print(initials_distence_matrix)
#
# vowel_distence_matrix = {}
# for i in vowel_list:
#     vowel_distence_matrix[i] = {}
#     distance_1 = []
#     distance_other = []
#     for j in vowel_list:
#         if Lev.distance(i, j) == 1:
#             distance_1.append(j)
#         elif Lev.distance(i, j) > 1:
#             distance_other.append(j)
#     vowel_distence_matrix[i]["1"] = distance_1
#     vowel_distence_matrix[i]["other"] = distance_other
# print(vowel_distence_matrix)
#
# with open("dataset/pinyin/initials_distance.json", "w", encoding="utf-8") as f:
#     json.dump(initials_distence_matrix, f, indent=4, ensure_ascii=False)
#
# with open("dataset/pinyin/vowel_distance.json", "w", encoding="utf-8") as f:
#     json.dump(vowel_distence_matrix, f, indent=4, ensure_ascii=False)


# generate = GenerateConfusion(config.CONFUSION)

# with open("/nvme0/qian/News2016/valid_correct.txt", "r") as f:
#     data = f.readlines()
#     data = [i[:-1] for i in data if i != "\n"]
#     data = data[:10000]
#
#
# def get_pinyin(data):
#     initials_list = []
#     vowel_list = []
#     for i in tqdm(data):
#         if len(i) < 64:
#             initials_list.append(np.array(lazy_pinyin(i, INITIALS)))
#             vowel_list.append(np.array(lazy_pinyin(i, FINALS_TONE3)))
#     initials_list = np.array(initials_list)
#     vowel_list = np.array(vowel_list)
#     return initials_list, vowel_list
#
#
# total_number = 40
# p = Pool(total_number)
# per_number = len(data) // total_number + 1
# result_list = []
# for i in range(total_number):
#     tmp_data = data[i * per_number:(i + 1) * per_number]
#     result_list.append(p.apply_async(get_pinyin, args=(tmp_data,)))
#
# p.close()
# p.join()
#
# initials_list = []
# vowel_list = []
# for result in result_list:
#     i, j = result.get()
#     initials_list += i
#     vowel_list += j
#
# assert len(initials_list) == len(data)
# assert len(initials_list) == len(vowel_list)
# initials_list = np.array(initials_list)
# vowel_list = np.array(vowel_list)
#
# np.save("/nvme0/qian/News2016/train_test_initials.npy", initials_list)
# np.save("/nvme0/qian/News2016/train_test_vowel.npy", vowel_list)
#
# initials = np.load("/nvme0/qian/News2016/valid_test_initials.npy", allow_pickle=True)
# vowel = np.load("/nvme0/qian/News2016/valid_test_vowel.npy", allow_pickle=True)
#
# print(len(initials))
