# 对齐anchor和sample音频长度

# 单线程
import json
import os
from pydub import AudioSegment
import subprocess
from tqdm import tqdm

data_list_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/train_data.list"
root = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech"

with open(data_list_path, 'r', encoding='utf-8') as file:
    data_list = file.readlines()

# 遍历数据，处理音频
for entry in tqdm(data_list):
    entry = json.loads(entry)
    anchor_path = os.path.join(root, entry['anchor_wav'])
    sample_path = os.path.join(root, entry['sample_wav'])


    # 加载音频并获取时长
    anchor_audio = AudioSegment.from_wav(anchor_path)
    sample_audio = AudioSegment.from_wav(sample_path)
    anchor_duration = len(anchor_audio)
    sample_duration = len(sample_audio)

    # 确定哪个音频更长
    longer_audio_path = anchor_path if anchor_duration > sample_duration else sample_path
    shorter_audio_path = anchor_path if anchor_duration <= sample_duration else sample_path
    shorter_duration = min(anchor_duration, sample_duration)
    shorter_audio = AudioSegment.from_wav(shorter_audio_path)

    # 使用 sox 进行加速处理
    subprocess.run(['sox', longer_audio_path, 'temp/temp.wav', 'gain', '-n', '-3', 'tempo', '-s', str(max(anchor_duration, sample_duration) / shorter_duration)])


    # 如果需要，进一步裁剪音频
    adjusted_audio = AudioSegment.from_wav('temp/temp.wav')
    if len(adjusted_audio) > shorter_duration:
        adjusted_audio = adjusted_audio[:shorter_duration]

    nas_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio_aligned"
    ssd_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Audio_aligned"

    long_save_nas_path = longer_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', nas_path)
    long_save_ssd_path = longer_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', ssd_path)
    short_save_nas_path = shorter_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', nas_path)
    short_save_ssd_path = shorter_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', ssd_path)

    if not os.path.exists(os.path.dirname(long_save_nas_path)):
        os.makedirs(os.path.dirname(long_save_nas_path))
    if not os.path.exists(os.path.dirname(long_save_ssd_path)):
        os.makedirs(os.path.dirname(long_save_ssd_path))
    if not os.path.exists(os.path.dirname(short_save_nas_path)):
        os.makedirs(os.path.dirname(short_save_nas_path))
    if not os.path.exists(os.path.dirname(short_save_ssd_path)):
        os.makedirs(os.path.dirname(short_save_ssd_path))

    # 保存调整后的音频
    adjusted_audio.export(long_save_nas_path, format='wav')
    adjusted_audio.export(long_save_ssd_path, format='wav')
    shorter_audio.export(short_save_nas_path, format='wav')
    shorter_audio.export(short_save_ssd_path, format='wav')



'''
# 多线程
from multiprocessing import Pool
from itertools import repeat
import json
import os
from pydub import AudioSegment
import subprocess


def process_audio(entry, each_process_len):

    process_id, entry = entry
    id = process_id // each_process_len
    if id == 0:
        print(f"\rProgress: {process_id}/{each_process_len}", end="")
    entry = json.loads(entry)
    root = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech"
    anchor_path = os.path.join(root, entry['anchor_wav'])
    sample_path = os.path.join(root, entry['sample_wav'])

    # 加载音频并获取时长
    anchor_audio = AudioSegment.from_wav(anchor_path)
    sample_audio = AudioSegment.from_wav(sample_path)
    anchor_duration = len(anchor_audio)
    sample_duration = len(sample_audio)

    # 确定哪个音频更长
    longer_audio_path = anchor_path if anchor_duration > sample_duration else sample_path
    shorter_audio_path = anchor_path if anchor_duration <= sample_duration else sample_path
    shorter_duration = min(anchor_duration, sample_duration)
    shorter_audio = AudioSegment.from_wav(shorter_audio_path)

    temp_dir = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/tools/temp/temp_{}.wav".format(id)
    if os.path.exists(temp_dir):
        os.remove(temp_dir)
    # 使用 sox 进行加速处理
    subprocess.run(['sox', longer_audio_path, temp_dir, 'gain', '-n', '-3', 'tempo', '-s', str(max(anchor_duration, sample_duration) / shorter_duration)])

    # 如果需要，进一步裁剪音频
    adjusted_audio = AudioSegment.from_wav(temp_dir)
    if len(adjusted_audio) > shorter_duration:
        adjusted_audio = adjusted_audio[:shorter_duration]

    nas_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio_aligned"
    ssd_path = "/ssd/zhuang/dataset/data_KeSpeech/KeSpeech/Audio_aligned"

    long_save_nas_path = longer_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', nas_path)
    long_save_ssd_path = longer_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', ssd_path)
    short_save_nas_path = shorter_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', nas_path)
    short_save_ssd_path = shorter_audio_path.replace('/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Audio', ssd_path)

    if not os.path.exists(os.path.dirname(long_save_nas_path)):
        os.makedirs(os.path.dirname(long_save_nas_path))
    if not os.path.exists(os.path.dirname(long_save_ssd_path)):
        os.makedirs(os.path.dirname(long_save_ssd_path))
    if not os.path.exists(os.path.dirname(short_save_nas_path)):
        os.makedirs(os.path.dirname(short_save_nas_path))
    if not os.path.exists(os.path.dirname(short_save_ssd_path)):
        os.makedirs(os.path.dirname(short_save_ssd_path))

    # 保存调整后的音频
    adjusted_audio.export(long_save_nas_path, format='wav')
    adjusted_audio.export(long_save_ssd_path, format='wav')
    shorter_audio.export(short_save_nas_path, format='wav')
    shorter_audio.export(short_save_ssd_path, format='wav')

def main():
    data_list_path = "/data/NAS_PLUS/zhuang/dataset/data_KeSpeech/KeSpeech/Tasks/ASContrastive/train_data.list"

    with open(data_list_path, 'r', encoding='utf-8') as file:
        data_list = file.readlines()

    num_processes = 64
    each_process_len = len(data_list) // num_processes

    with Pool(processes=num_processes) as pool:
        args = zip(enumerate(data_list), repeat(each_process_len))
        # 创建一个包含数据和进程索引的元组
        pool.starmap(process_audio, args)


if __name__ == "__main__":
    main()

'''