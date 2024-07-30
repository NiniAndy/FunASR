# 计算有多少文本重合的语音
from collections import Counter

text_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/Conbinedata/dev/text"
with open(text_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# Extract the text from each line, assuming the format is 'id text'.
texts = [line.split(' ', 1)[1].strip() for line in lines if len(line.split(' ', 1)) > 1]
text_counter = Counter(texts)

# Count how many texts appear more than once.
repeated_texts_count = sum(1 for count in text_counter.values() if count > 1)

print(f'Number of texts that appear more than once: {repeated_texts_count}')
