import json

type = "acc"

data_list_path = "/ssd/zhuang/CodeStorage/DIMNet/data/data_train_phase1.list"
with open(data_list_path, 'r', encoding='utf-8') as file:
    data_list = file.readlines()

if type == "char":
    threshold = 10
    char_count = {}

    for line in data_list:
        data = json.loads(line)
        text = data['text']
        for char in text:
            if char not in char_count:
                char_count[char] = 1
            else:
                char_count[char] += 1

    frequent_chars = [char for char, count in char_count.items() if count > threshold]
    with open("lang_char.txt", 'w', encoding='utf-8') as file:
        file.write("<blank> 0\n<unk> 1\n")
        for i, char in enumerate(frequent_chars, start=2):
            file.write(f"{char} {i}\n")
        file.write(f"<sos/eos> {i+1}\n")

if type == "pny":
    pinyin_list = []
    for line in data_list:
        data = json.loads(line)
        pinyin = data['pinyin']
        pinyin = pinyin.split(" ")
        for p in pinyin:
            if len(p) == 0:
                continue
            if p not in pinyin_list:
                pinyin_list.append(p)

    with open("lang_pny.txt", 'w', encoding='utf-8') as file:
        file.write("<blank> 0\n<unk> 1\n")
        for i, pny in enumerate(pinyin_list, start=2):
            file.write(f"{pny} {i}\n")
        file.write(f"<sos/eos> {i + 1}\n")


if type == "acc":
    acc_list = []
    for line in data_list:
        data = json.loads(line)
        acc = data['acc']
        if acc not in acc_list:
            acc_list.append(acc)

    with open("lang_acc.txt", 'w', encoding='utf-8') as file:
        for i, acc in enumerate(acc_list, start=0):
            file.write(f"{acc} {i}\n")

print ("Done!")

