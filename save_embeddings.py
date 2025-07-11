import argparse
import json
import os
import pickle as pk
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from custom_dataloader import VlspDataset 

from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

# --- CẤU HÌNH ---
KAGGLE_BASE_PATH = "/kaggle/input/vlsp-vsasv-datasets/"
SET_NAME_TO_RUN = "trn"

PROTOCOL_PATH = os.path.join(KAGGLE_BASE_PATH, "train_vlsp_2025_metadata.txt")
DATA_DIR = os.path.join(KAGGLE_BASE_PATH, f"vlsp2025/{SET_NAME_TO_RUN}/")
# --- KẾT THÚC CẤU HÌNH ---


def save_embeddings(set_name, cm_embd_ext, asv_embd_ext, device):
    if not os.path.exists(PROTOCOL_PATH):
        print(f"LỖI: File metadata không tồn tại tại: {PROTOCOL_PATH}")
        return

    meta_lines = open(PROTOCOL_PATH, "r").readlines()
    utt_list = []
    
    print(f"Đã đọc {len(meta_lines)} dòng từ file metadata.")
    print("Bắt đầu xử lý từng dòng...")
    
    for i, line in enumerate(meta_lines):
        parts = line.strip().split(" ")
        if len(parts) != 3:
            print(f"  - Dòng {i+1}: Bỏ qua do không có 3 phần tử.")
            continue
        
        filepath = parts[1]
        split_pattern = f"vlsp2025/{set_name}/"
        
        if split_pattern in filepath:
            relative_path = filepath.split(split_pattern, 1)[1]
            utt_list.append(relative_path)
        else:
            # In ra thông báo lỗi để biết tại sao không tách được
            print(f"  - Dòng {i+1}: Tách thất bại! Không tìm thấy '{split_pattern}' trong '{filepath}'")

    print(f"Xử lý hoàn tất. Tổng số tệp hợp lệ trong danh sách: {len(utt_list)}")
    
    # Kiểm tra nếu danh sách rỗng thì dừng lại
    if not utt_list:
        print("LỖI: Danh sách tệp cần xử lý bị rỗng. Dừng chương trình.")
        return
        
    dataset = VlspDataset(utt_list, Path(DATA_DIR))
    loader = DataLoader(dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True)

    cm_emb_dic = {}
    asv_emb_dic = {}

    print(f"Bắt đầu trích xuất embedding từ tập {set_name}...")

    for batch_x, keys in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb, _ = cm_embd_ext(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for key, cm_emb, asv_emb in zip(keys, batch_cm_emb, batch_asv_emb):
            cm_emb_dic[key] = cm_emb
            asv_emb_dic[key] = asv_emb

    os.makedirs("embeddings", exist_ok=True)
    with open(f"embeddings/cm_embd_{set_name}.pk", "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open(f"embeddings/asv_embd_{set_name}.pk", "wb") as f:
        pk.dump(asv_emb_dic, f)
    print("Trích xuất embedding thành công!")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-aasist_config", type=str, default="./aasist/config/AASIST.conf")
    parser.add_argument("-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth")
    parser.add_argument("-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model")
    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    with open(args.aasist_config, "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    cm_embd_ext = AASISTModel(model_config)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.to(device)
    cm_embd_ext.eval()

    asv_embd_ext = ECAPA_TDNN(C=1024)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.to(device)
    asv_embd_ext.eval()

    save_embeddings(SET_NAME_TO_RUN, cm_embd_ext, asv_embd_ext, device)


if __name__ == "__main__":
    main()