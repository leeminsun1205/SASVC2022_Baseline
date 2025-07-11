# File: custom_dataloader.py

import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

class VlspDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh dành riêng cho dữ liệu VLSP của bạn.
    """
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        # list_IDs chứa đường dẫn tương đối, vd: 'id00271/bonafide/00000.wav'
        relative_path = self.list_IDs[index]
        
        # Tạo đường dẫn đầy đủ đến file âm thanh
        full_audio_path = self.base_dir / relative_path
        
        # Lấy tên file để dùng làm key trả về
        key_filename = os.path.basename(relative_path)

        # Đọc file âm thanh
        X, _ = sf.read(str(full_audio_path))
        
        X_pad = pad(X, self.cut)
        x_inp = torch.Tensor(X_pad)
        
        return x_inp, key_filename