# 清洗数据使得数据只有一个锚点和一个样本
with open('text_005', 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 过滤只包含两个ID的行
filtered_lines = []

for line in lines:
    parts = line.split()
    # 假设ID包含下划线，而文本不包含
    id_count = sum(1 for part in parts if '_' in part)

    if id_count == 2:
        filtered_lines.append(line.strip())

# 将过滤后的行写入新文件
with open('text_006', 'w', encoding='utf-8') as out_file:
    for line in filtered_lines:
        out_file.write(line + '\n')

print("只包含两个ID的行已保存到text_006")
