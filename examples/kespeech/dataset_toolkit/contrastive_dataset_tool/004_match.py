# 将text_003B中同一说话人不同口音的语音对应的文本提取出来
with open('text_003B', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 处理每一行以找到匹配的speakerID
matching_lines = []

for line in lines:
    parts = line.split()
    text = parts[-1]
    ids = parts[:-1]

    # 创建一个字典来跟踪speakerID
    speaker_dict = {}
    for id_ in ids:
        speaker_id = id_.split('_')[0]
        if speaker_id not in speaker_dict:
            speaker_dict[speaker_id] = []
        speaker_dict[speaker_id].append(id_)

    # 查找匹配的speakerID
    for speaker_id, id_list in speaker_dict.items():
        if len(id_list) > 1:
            matching_line = ' '.join(id_list + [text])
            matching_lines.append(matching_line)

# 将匹配的行写入新文件
with open('text_004', 'w', encoding='utf-8') as out_file:
    for line in matching_lines:
        out_file.write(line + '\n')

print("匹配的行已保存到text_004")
