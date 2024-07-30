# 按照规则对锚点和样本进行排序
from tqdm import tqdm
import json

id2accent = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/text_007"
with open(id2accent, 'r', encoding='utf-8') as file:
    idNaccent = file.readlines()

formatted_lines = []

for line in tqdm(idNaccent):
    parts = line.split()
    id1, accent1, id2, accent2, *text = parts

    # 应用规则
    if 'p2_' in id1 and 'p2_' not in id2:
        anchor, sample = id1, id2
        anchor_acc, sample_acc = accent1, accent2
    elif 'p2_' in id2 and 'p2_' not in id1:
        anchor, sample = id2, id1
        anchor_acc, sample_acc = accent2, accent1
    elif 'p2_' not in id1 and 'p2_' not in id2:
        if accent1 == 'Mandarin' and accent2 != 'Mandarin':
            anchor, sample = id1, id2
            anchor_acc, sample_acc = accent1, accent2
        elif accent2 == 'Mandarin' and accent1 != 'Mandarin':
            anchor, sample = id2, id1
            anchor_acc, sample_acc = accent2, accent1
        else:
            continue  # 跳过规则3的情况
    else:
        # 对于规则4和5，比较ID的16进制部分
        key1, key2 = id1.split('_')[-1], id2.split('_')[-1]
        anchor, sample = (id1, id2) if key1 > key2 else (id2, id1)
        anchor_acc, sample_acc = (accent1, accent2) if key1 > key2 else (accent2, accent1)

    formatted_line = json.dumps({"anchor": anchor,
                                 "anchor_acc": anchor_acc,
                                 "sample": sample,
                                 "sample_acc": sample_acc,
                                 "txt": ' '.join(text)
                                 }, ensure_ascii=False)
    formatted_lines.append(formatted_line)

# 将处理后的行写入新文件
with open('text_008', 'w', encoding='utf-8') as out_file:
    for line in formatted_lines:
        out_file.write(line + '\n')

print("处理后的行已保存到text_008")
