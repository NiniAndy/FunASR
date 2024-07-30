# 对配对的语音标注口音
utt2subdialect_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/Conbinedata/dev/utt2subdialect"
with open(utt2subdialect_path, 'r', encoding='utf-8') as file:
    utt_lines = file.readlines()

# 将utt2subdialect的内容处理成字典，键为id，值为口音
id_to_dialect = {}
for line in utt_lines:
    parts = line.strip().split()
    if len(parts) == 2:
        id_, dialect = parts
        id_to_dialect[id_] = dialect

text_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/text_006"
# 打开并读取text文件的内容
with open(text_path, 'r', encoding='utf-8') as file:
    text_lines = file.readlines()

# 处理text文件，为每个id添加对应的口音
result_lines = []
for line in text_lines:
    parts = line.strip().split()
    ids = parts[:-1]
    text = parts[-1]

    # 为每个id添加对应的口音
    line_with_dialect = []
    for id_ in ids:
        dialect = id_to_dialect.get(id_, "Unknown")  # 如果id不存在于字典中，则返回"Unknown"
        line_with_dialect.append(f"{id_} {dialect}")

    line_with_dialect.append(text)
    result_lines.append(' '.join(line_with_dialect))

# 将处理结果写入新文件
with open('text_007', 'w', encoding='utf-8') as out_file:
    for line in result_lines:
        out_file.write(line + '\n')

print("处理结果已保存到text_007")
