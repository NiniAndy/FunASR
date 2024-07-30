# 把有重复的文本和对应的id提取出来
from collections import defaultdict

text_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/Conbinedata/dev/text"

with open(text_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Create a dictionary to hold the texts and their associated ids
text_to_ids = defaultdict(list)

# Populate the dictionary
for line in lines:
    id_, text = line.strip().split(' ', 1)
    text_to_ids[text].append(id_)

# Filter texts that appear more than once and prepare them in the desired format
repeated_texts = ["{} {}".format(' '.join(ids), text) for text, ids in text_to_ids.items() if len(ids) > 1]

# Write the repeated texts to a new file
with open('text_002', 'w', encoding='utf-8') as out_file:
    for line in repeated_texts:
        out_file.write(line + '\n')

print(f"Repeated texts have been written to text_002")
