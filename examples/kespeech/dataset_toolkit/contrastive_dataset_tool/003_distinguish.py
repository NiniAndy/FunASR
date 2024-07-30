# 先把只包含2个id的保存至一个文件中，把超过两个id的保存至另一个文件中
from tqdm import tqdm

with open('text_002', 'r', encoding='utf-8') as file:
    lines = file.readlines()

conform_lines = []
other_lines = []

# Process each line
for line in tqdm(lines):
    # Count the number of IDs in the line (assuming IDs contain an underscore)
    id_count = sum(1 for part in line.split() if '_' in part)

    # Classify the line based on the number of IDs
    if id_count == 2:
        conform_lines.append(line.strip())
    else:
        other_lines.append(line.strip())

# Write the lines with exactly two IDs to a new file
with open('text_003A', 'w', encoding='utf-8') as out_file:
    out_file.write('\n'.join(conform_lines))

# Write the other lines to another file
with open('text_003B', 'w', encoding='utf-8') as out_file:
    out_file.write('\n'.join(other_lines))

print("Lines with two IDs saved to text_003A")
print("Other lines saved to text_003B")
