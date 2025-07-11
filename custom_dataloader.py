# File: custom_dataloader.py

import os
from pathlib import Path

import numpy as np
import torch
import torchaudio  # Thay thế soundfile bằng torchaudio
from torch.utils.data import Dataset

def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    # Dùng torch.tile thay cho np.tile
    padded_x = torch.tile(x, (num_repeats,))[:max_len]
    return padded_x

class VlspDataset(Dataset):
    """
    Lớp Dataset tùy chỉnh sử dụng torchaudio.
    """
    def __init__(self, list_IDs, base_dir):
        self.list_IDs = list_IDs
        self.base_dir = Path(base_dir)
        self.cut = 64600

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        relative_path = self.list_IDs[index]
        full_audio_path = self.base_dir / relative_path
        key_filename = os.path.basename(relative_path)

        # ================== PHẦN THAY ĐỔI CHÍNH ==================
        # Sử dụng torchaudio.load để đọc file âm thanh
        # waveform là một Tensor, sr là sample rate
        waveform, sr = torchaudio.load(str(full_audio_path))
        
        # Lấy kênh đầu tiên nếu là âm thanh stereo
        if waveform.shape[0] > 1:
            waveform = waveform[0]
        else:
            waveform = waveform.squeeze()
        # =========================================================
        
        # Bây giờ waveform đã là Tensor, không cần chuyển đổi nữa
        x_inp_pad = pad(waveform, self.cut)
        
        return x_inp_pad, key_filename