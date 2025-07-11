import argparse
import json
import os
import pickle as pk
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# from aasist.data_utils import Dataset_ASVspoof2019_devNeval
from custom_dataloader import VlspDataset 

from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from utils import load_parameters

# list of dataset partitions
SET_PARTITION = ["train", "dev", "eval"]

KAGGLE_BASE_PATH = "/kaggle/input/vlsp-vsasv-datasets/"

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "train": os.path.join(KAGGLE_BASE_PATH, "train_vlsp_2025_metadata.txt"),
    "dev": os.path.join(KAGGLE_BASE_PATH, "train_vlsp_2025_metadata.txt"),  # Tạm thời dùng train
    "eval": os.path.join(KAGGLE_BASE_PATH, "train_vlsp_2025_metadata.txt") # Tạm thời dùng train
}
# directories of each dataset partition
SET_DIR = {
    "train": os.path.join(KAGGLE_BASE_PATH, "vlsp2025/vlsp2025/train/"),
    "dev": os.path.join(KAGGLE_BASE_PATH, "vlsp2025/vlsp2025/train/"), # Tạm thời dùng train
    "eval": os.path.join(KAGGLE_BASE_PATH, "vlsp2025/vlsp2025/train/")# Tạm thời dùng train
}

# enrolment data list for speaker model calculation
# each speaker model comprises multiple enrolment utterances
# Placeholder cho các tệp protocol, cập nhật khi có
SET_TRN = {
    "dev": [],
    "eval": [],
}

def save_embeddings(
    set_name, cm_embd_ext, asv_embd_ext, device
):
    protocol_path = SET_CM_PROTOCOL[set_name]
    base_dir = SET_DIR[set_name]

    print(f"\n[DEBUG] Bắt đầu xử lý cho set: '{set_name}'")
    print(f"[DEBUG] Đường dẫn file metadata: {protocol_path}")
    print(f"[DEBUG] Thư mục gốc của dữ liệu: {base_dir}")

    if not os.path.exists(protocol_path):
        print(f"[DEBUG] LỖI: Không tìm thấy file metadata tại '{protocol_path}'. Dừng lại.")
        return

    meta_lines = open(protocol_path, "r").readlines()
    print(f"[DEBUG] Đã đọc được {len(meta_lines)} dòng từ file metadata.")

    utt_list = []  # Sẽ chứa các đường dẫn tương đối

    for i, line in enumerate(meta_lines):
        if i < 5: # In ra 5 dòng đầu tiên để kiểm tra
            print(f"[DEBUG] Dòng {i+1}: {line.strip()}")

        parts = line.strip().split(" ")
        if len(parts) != 3:
            continue
        
        filepath = parts[1] # vd: vlsp2025/train/id00271/bonafide/00000.wav
        
        try:
            # Đây là dòng code quan trọng cần kiểm tra
            relative_path = filepath.split(f"vlsp2025/vlsp2025/{set_name}/", 1)[1]
            utt_list.append(relative_path)
        except IndexError:
            # Nếu có lỗi ở đây, nó sẽ được in ra
            if i < 10: # Chỉ in 10 lỗi đầu tiên để tránh spam
                print(f"[DEBUG] LỖI PARSING tại dòng {i+1}: Không thể trích xuất đường dẫn tương đối từ '{filepath}'")
            continue

    print(f"\n[DEBUG] Đã xử lý xong metadata. Tổng số file được thêm vào utt_list: {len(utt_list)}")
    if not utt_list:
        print("[DEBUG] KẾT LUẬN: utt_list rỗng! Đây là nguyên nhân gây ra lỗi '0it'. Vui lòng kiểm tra lại logic split đường dẫn bên trên.")
        return

    dataset = VlspDataset(utt_list, Path(base_dir))
    print(f"[DEBUG] Đã tạo VlspDataset với {len(dataset)} mẫu.")

    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )
    print(f"[DEBUG] Đã tạo DataLoader với số batch là {len(loader)}.")


    cm_emb_dic = {}
    asv_emb_dic = {}

    print(f"\nGetting embeddings from set {set_name}...")

    # Vòng lặp tqdm sẽ cho bạn biết chính xác có bao nhiêu item được xử lý
    for batch_x, keys in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_cm_emb, _ = cm_embd_ext(batch_x)
            batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for key, cm_emb, asv_emb in zip(keys, batch_cm_emb, batch_asv_emb):
            # Key bây giờ là tên file, ví dụ '00000.wav'
            cm_emb_dic[key] = cm_emb
            asv_emb_dic[key] = asv_emb
    
    os.makedirs("new_embeddings", exist_ok=True)
    with open(f"new_embeddings/cm_embd_{set_name}.pk", "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open(f"new_embeddings/asv_embd_{set_name}.pk", "wb") as f:
        pk.dump(asv_emb_dic, f)
    print('Done!')

def save_models(set_name, asv_embd_ext, device):
    if not SET_TRN.get(set_name):
        print(f"No enrollment protocols for '{set_name}'. Skipping speaker model generation.")
        return
    
    utt2spk = {}
    utt_list = []

    for trn in SET_TRN[set_name]:
        meta_lines = open(trn, "r").readlines()

        for line in meta_lines:
            tmp = line.strip().split(" ")

            spk = tmp[0]
            utts = tmp[1].split(",")

            for utt in utts:
                if utt in utt2spk:
                    print("Duplicated utt error", utt)

                utt2spk[utt] = spk
                utt_list.append(utt)

    base_dir = SET_DIR[set_name]
    dataset = VlspDataset(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True
    )
    asv_emb_dic = {}

    print("Getting embedgins from set %s..." % (set_name))

    for batch_x, key in tqdm(loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

        for k, asv_emb in zip(key, batch_asv_emb):
            utt = k
            spk = utt2spk[utt]

            if spk not in asv_emb_dic:
                asv_emb_dic[spk] = []

            asv_emb_dic[spk].append(asv_emb)

    for spk in asv_emb_dic:
        asv_emb_dic[spk] = np.mean(asv_emb_dic[spk], axis=0)

    with open("embeddings/spk_model.pk_%s" % (set_name), "wb") as f:
        pk.dump(asv_emb_dic, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-aasist_config", type=str, default="./aasist/config/AASIST.conf"
    )
    parser.add_argument(
        "-aasist_weight", type=str, default="./aasist/models/weights/AASIST.pth"
    )
    parser.add_argument(
        "-ecapa_weight", type=str, default="./ECAPATDNN/exps/pretrain.model"
    )

    return parser.parse_args()


def main():
    args = get_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))

    with open(args.aasist_config, "r") as f_json:
        config = json.loads(f_json.read())

    model_config = config["model_config"]
    cm_embd_ext = AASISTModel(model_config).to(device)
    load_parameters(cm_embd_ext.state_dict(), args.aasist_weight)
    cm_embd_ext.to(device)
    cm_embd_ext.eval()

    asv_embd_ext = ECAPA_TDNN(C=1024)
    load_parameters(asv_embd_ext.state_dict(), args.ecapa_weight)
    asv_embd_ext.to(device)
    asv_embd_ext.eval()

    # for set_name in SET_PARTITION:
    #     save_embeddings(
    #         set_name,
    #         cm_embd_ext,
    #         asv_embd_ext,
    #         device,
    #     )
    #     if set_name == "train":
    #         continue
        # save_models(set_name, asv_embd_ext, device)
    # temp
    set_to_run = "train"
    save_embeddings(
        set_to_run,
        cm_embd_ext,
        asv_embd_ext,
        device,
    )


if __name__ == "__main__":
    main()
