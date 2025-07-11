# file: dataloaders/finetune_loader.py

import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

# --- Hàm trợ giúp ---
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def read_audio(base_dir, spk_id, filename, spoof_type="bonafide"):
    """
    Hàm đọc audio đã được sửa lỗi, có khả năng tái tạo đường dẫn đúng.
    """
    # Tái tạo đường dẫn tương đối từ spk_id và filename
    # Ví dụ: base_dir / "id00271" / "bonafide" / "00000.wav"
    filepath = base_dir / spk_id / spoof_type / filename
    
    # Một số trường hợp, `spk_meta` có thể chứa cả đường dẫn, ta xử lý luôn
    if not os.path.exists(filepath):
        filepath = base_dir / filename # Dùng trực tiếp nếu đường dẫn đã đầy đủ

    wav, _ = sf.read(str(filepath))
    wav_pad = pad(wav)
    return torch.Tensor(wav_pad)

# --- Dataset cho Training ---

class FinetuneTrainDataset(Dataset):
    def __init__(self, spk_meta, base_dir):
        self.spk_meta = {spk: meta for spk, meta in spk_meta.items() if len(meta["bonafide"]) >= 2}
        self.base_dir = Path(base_dir)
        self.spk_ids = list(self.spk_meta.keys())
        # Độ dài của epoch, có thể đặt là một số lớn
        self.length = len(self.spk_ids) * 20 

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Logic tạo cặp train tương tự backend_fusion
        ans_type = random.randint(0, 1)
        
        if ans_type == 1:  # target
            spk = random.choice(self.spk_ids)
            enr_fname, tst_fname = random.sample(self.spk_meta[spk]["bonafide"], 2)
            enr_wav = read_audio(self.base_dir, spk, enr_fname)
            tst_wav = read_audio(self.base_dir, spk, tst_fname)

        else:  # nontarget
            spk1, spk2 = random.sample(self.spk_ids, 2)
            enr_fname = random.choice(self.spk_meta[spk1]["bonafide"])
            enr_wav = read_audio(self.base_dir, spk1, enr_fname)
            
            # 50% cơ hội là nontarget-spoof
            if len(self.spk_meta[spk2]["spoof"]) > 0 and random.random() > 0.5:
                 tst_fname = random.choice(self.spk_meta[spk2]["spoof"])
                 tst_wav = read_audio(self.base_dir, spk2, tst_fname, spoof_type="spoof")
            else: # nontarget-bonafide (zero-effort)
                 tst_fname = random.choice(self.spk_meta[spk2]["bonafide"])
                 tst_wav = read_audio(self.base_dir, spk2, tst_fname)
        
        return enr_wav, tst_wav, torch.tensor(ans_type)

# --- Dataset cho Validation & Evaluation ---

class FinetuneDevEvalDataset(Dataset):
    def __init__(self, trial_list, base_dir):
        self.trial_list = trial_list
        self.base_dir = Path(base_dir)

    def __len__(self):
        return len(self.trial_list)

    def __getitem__(self, index):
        line = self.trial_list[index].strip().split()
        spkmd, key, _, ans = line
        
        # Đọc file âm thanh dựa trên speaker model và test key
        # Giả định speaker model là một file âm thanh enrollment duy nhất
        # (Cần điều chỉnh nếu speaker model được tạo từ nhiều file)
        enr_spk_id = spkmd.split('_')[0] # vd: 'id01073_001' -> 'id01073'
        enr_wav = read_audio(self.base_dir, enr_spk_id, spkmd + ".wav")
        
        # Test file có thể là bonafide hoặc spoof
        tst_spk_id = key.split('/')[0] # vd: 'id01073/105-105-id01073-00000'
        tst_wav = read_audio(self.base_dir, tst_spk_id, key + ".wav")
        
        return enr_wav, tst_wav, ans

# --- Hàm Factory ---

def get_trnset(spk_meta_trn: dict, base_dir: str):
    """Hàm tạo training dataset."""
    return FinetuneTrainDataset(spk_meta=spk_meta_trn, base_dir=base_dir)

def get_dev_evalset(utt_list: list, base_dir: str, spk_model=None):
    """Hàm tạo validation/evaluation dataset."""
    # spk_model không cần thiết khi tải dữ liệu thô
    return FinetuneDevEvalDataset(trial_list=utt_list, base_dir=base_dir)