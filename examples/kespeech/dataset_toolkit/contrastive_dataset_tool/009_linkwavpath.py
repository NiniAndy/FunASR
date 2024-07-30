import json
from tqdm import tqdm

ordered_data = '/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/text_008'
wav_scp = '/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/Conbinedata/dev/wav.scp'

formatted_lines = []

id_to_path = {}
with open(wav_scp, 'r', encoding='utf-8') as file:
    for line in file:
        id_, path = line.strip().split(maxsplit=1)
        id_to_path[id_] = path


with open(ordered_data, 'r', encoding='utf-8') as file:
    for line in tqdm(file):
        data = json.loads(line)
        anchor_id = data["anchor"]
        anchor_acc = data["anchor_acc"]
        sample_id = data["sample"]
        sample_acc = data["sample_acc"]
        txt = data["txt"]


        # 检查是否找到了anchor和sample对应的路径
        if anchor_id in id_to_path and sample_id in id_to_path:
            anchor_wav = id_to_path[anchor_id]
            sample_wav = id_to_path[sample_id]
            formatted_line = json.dumps({"anchor": anchor_id,
                                         "anchor_acc": anchor_acc,
                                         "anchor_wav": anchor_wav,
                                         "sample": sample_id,
                                         "sample_acc": sample_acc,
                                         "sample_wav": sample_wav,
                                         "txt": txt}, ensure_ascii=False)
            formatted_lines.append(formatted_line)

output_dir = '/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/text_009'
with open(output_dir, 'w', encoding='utf-8') as file:
    for line in formatted_lines:
        file.write(line + '\n')

print(f"Updated data saved to {output_dir}")
