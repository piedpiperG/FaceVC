import os
import subprocess
from glob import glob

if __name__ == "__main__":
    # 设定HDTF文件夹路径
    dataset_root = '/home/gyz/facevc/Dataset/HDTF'


    # 为每个子文件夹（RD, WDA, WRA）创建对应的WAV文件夹
    types = ["RD", "WDA", "WRA"]
    for type_ in types:
        wav_root = os.path.join(dataset_root, f"{type_}_wav")
        os.makedirs(wav_root, exist_ok=True)  # 如果目录不存在，则创建目录
        
        # 获取当前子文件夹中所有MP4文件的路径
        video_paths = glob(os.path.join(dataset_root, type_, "*.mp4"))
        for video_path in video_paths:
            # 为每个视频文件设置一个对应的WAV保存路径
            save_path = os.path.join(wav_root, video_path.split("/")[-1].replace(".mp4", ".wav"))
            
            # 构建并执行ffmpeg命令来转换视频为音频
            command = f"ffmpeg -i '{video_path}' -vn -acodec pcm_s16le -ar 16000 -ac 1 '{save_path}'"
            res = subprocess.call(command, shell=True)
            if res == 0:
                print(f"Converted {video_path} to {save_path}")
            else:
                print(f"Failed to convert {video_path}")
