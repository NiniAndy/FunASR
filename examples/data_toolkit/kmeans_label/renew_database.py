import os
import json
from tqdm import tqdm

def get_label_iterator(spurious_label_path):
    with open(spurious_label_path, "r") as f:
        lines = f.readlines()
        def iterate():
            for item in lines:
                item = item.strip()
                item = json.loads(item)
                yield item
        return iterate, len(lines)



root = "/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data3"
spurious_label_name = "kmeans_label"
# 找到所有audio_datasets.jsonl文件
deal_files = []
for root, dirs, files in os.walk(root):
    for file in files:
        if file == "audio_datasets.jsonl":
            audio_datasets = os.path.join(root, file)
            deal_files.append(audio_datasets)

# deal_files = ["/ssd/zhuang/code/FunASR/examples/kespeech/DATA/data3/AS/test/audio_datasets.jsonl"]

for audio_datasets in deal_files:

    print ("processing file: ", audio_datasets)
    update_file = []


    with open(audio_datasets, "r") as f:
        lines = f.readlines()

    spurious_label_path = audio_datasets.replace("audio_datasets.jsonl", spurious_label_name)

    with open(spurious_label_path, "r") as f:
        spurious_label_lines = f.readlines()

    for i in tqdm(range(len(spurious_label_lines))):

        spurious_label = spurious_label_lines[i].strip()
        len_spurious_label = len(spurious_label.split(" "))
        item = json.loads(lines[i])
        if len_spurious_label != item["source_len"]:
            print ("error: ", item["source"])
            continue

        item["spurious_label"] = spurious_label
        update_file.append(item)

    os.remove(audio_datasets)
    with open(audio_datasets, "w") as f:
        for item in update_file:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print ("finish processing file: ", audio_datasets)




