from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2
import numpy as np
from PIL import Image
import os
from glob import glob
from joblib import Parallel, delayed
from tqdm import tqdm
import torch

# 检查是否有可用的GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_face_embs(video_path, mtcnn, resnet, save_root):
    face_imgs = []
    face_embs = []
    capture = cv2.VideoCapture(video_path)
    if capture.isOpened():
        while True:
            ret, img = capture.read()
            if not ret:
                break
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_cropped = mtcnn(img)
                if img_cropped is not None:
                    img_cropped = img_cropped.to(device)  # 将裁剪后的图像移动到GPU

                    # 将 img_cropped 转换为 NumPy 数组，然后转换为 PIL 图像
                    img_cropped_np = img_cropped.detach().cpu().numpy().transpose(1, 2, 0) * 255
                    img_pil = Image.fromarray(img_cropped_np.astype(np.uint8))
                    # 创建保存图像的路径，并以帧的索引命名文件
                    img_save_dir = os.path.join(save_root, 'facesjpg')
                    os.makedirs(img_save_dir, exist_ok=True)
                    img_save_path = os.path.join(img_save_dir, f"{video_path.split('/')[-1][:-4]}.jpg")
                    img_pil.save(img_save_path, format='JPEG')

                    img_embedding = resnet(img_cropped.unsqueeze(0).to(device))  # 将图像输入到resnet并在GPU上进行计算
                    face_imgs.append(img_cropped.detach().cpu().numpy())
                    face_embs.append(img_embedding.detach().cpu().numpy())
            except:
                continue

    face_embs = np.stack(face_embs, axis=0) if face_embs else None
    face_imgs = np.stack(face_imgs, axis=0) if face_imgs else None
    face_embs_mean = np.mean(face_embs, axis=0) if face_embs is not None else None

    os.makedirs(os.path.join(save_root,'facesemb'), exist_ok=True)
    np.save(os.path.join(save_root,'facesemb', video_path.split('/')[-1][:-4]+'.npy'), face_embs)
    os.makedirs(os.path.join(save_root,'facesimg'), exist_ok=True)
    np.save(os.path.join(save_root,'facesimg', video_path.split('/')[-1][:-4]+'.npy'), face_imgs)
    os.makedirs(os.path.join(save_root,'facesembmean'), exist_ok=True)
    np.save(os.path.join(save_root,'facesembmean', video_path.split('/')[-1][:-4]+'.npy'), face_embs_mean)

if __name__ == "__main__":
    # 将MTCNN和Resnet模型移动到GPU
    mtcnn = MTCNN(image_size=128, margin=50, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    dataset_root = '/home/gyz/facevc/Dataset/HDTF'  # 更改为第一段代码中的路径格式
    types = ["RD", "WDA", "WRA"]  # 类型，与第一段代码保持一致
    for type_ in types:
        video_root = os.path.join(dataset_root, f"{type_}")
        video_paths = glob(os.path.join(video_root, "*.mp4"))
        save_root = os.path.join(dataset_root, f"{type_}_features")

        Parallel(n_jobs=4)(delayed(extract_face_embs)(video_path, mtcnn, resnet, save_root) for video_path in tqdm(video_paths))
