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

# list of dataset partitions
SET_PARTITION = ["train", "dev", "eval"]

KAGGLE_BASE_PATH = "/kaggle/input/vlsp-vsasv-datasets/"

# list of countermeasure(CM) protocols
SET_CM_PROTOCOL = {
    "train": os.path.join(KAGGLE_BASE_PATH, "train_vlsp_2025_metadata.txt"),
    "dev": os.path.join(KAGGLE_BASE_PATH, "train_vlsp_2025_metadata.txt"), # temp 
    "eval": os.path.join(KAGGLE_BASE_PATH, "train_vlsp_2025_metadata.txt") # temp 
}
# directories of each dataset partition
SET_DIR = {
    "train": os.path.join(KAGGLE_BASE_PATH, "vlsp2025/vlsp2025/train/"),
    "dev": os.path.join(KAGGLE_BASE_PATH, "vlsp2025/vlsp2025/train/"), # temp 
    "eval": os.path.join(KAGGLE_BASE_PATH, "vlsp2025/vlsp2025/train/") # temp 
}

SET_TRN = {
    "dev": [], # temp 
    "eval": [], # temp 
}

def save_embeddings(
    set_name, cm_embd_ext, asv_embd_ext, device
):
    protocol_path = SET_CM_PROTOCOL[set_name]
    base_dir = SET_DIR[set_name]

    if not os.path.exists(protocol_path):
        print(f"Metadata file for '{set_name}' not found at {protocol_path}. Skipping.")
        return

    meta_lines = open(protocol_path, "r").readlines()
    utt_list = []

    for line in meta_lines:
        parts = line.strip().split(" ")
        if len(parts) != 3:
            continue
        
        filepath = parts[1]
        
        prefix_to_remove = f"vlsp2025/{set_name}/"
        if filepath.startswith(prefix_to_remove):
            relative_path = filepath[len(prefix_to_remove):]
            utt_list.append(relative_path)

    if not utt_list:
        print("\n[Error] List of files is empty after processing metadata. Please check the format file!")
        return

    dataset = VlspDataset(utt_list, Path(base_dir))
    loader = DataLoader(
        dataset, batch_size=30, shuffle=False, drop_last=False, pin_memory=True, num_workers=4
    )

    cm_emb_dic = {}
    asv_emb_dic = {}

    print(f"\nStart to extract embedding for '{set_name}' ({len(dataset)} files)...")

    with tqdm(total=len(dataset), desc=f"Processing {set_name}", unit="file") as pbar:
        for batch_x, keys in loader:
            batch_x = batch_x.to(device)
            with torch.no_grad():
                batch_cm_emb, _ = cm_embd_ext(batch_x)
                batch_cm_emb = batch_cm_emb.detach().cpu().numpy()
                batch_asv_emb = asv_embd_ext(batch_x, aug=False).detach().cpu().numpy()

            for key, cm_emb, asv_emb in zip(keys, batch_cm_emb, batch_asv_emb):
                cm_emb_dic[key] = cm_emb
                asv_emb_dic[key] = asv_emb
            
            pbar.update(len(keys))
    
    output_dir = "embeddings"
    os.makedirs(output_dir, exist_ok=True)
    cm_output_path = os.path.join(output_dir, f"cm_embd_{set_name}.pk")
    asv_output_path = os.path.join(output_dir, f"asv_embd_{set_name}.pk")

    with open(cm_output_path, "wb") as f:
        pk.dump(cm_emb_dic, f)
    with open(asv_output_path, "wb") as f:
        pk.dump(asv_emb_dic, f)
        
    print(f"\nEmbeddings were saved!")


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
        
    # === temp ===
    set_to_run = "train"
    save_embeddings(
        set_to_run,
        cm_embd_ext,
        asv_embd_ext,
        device,
    )
    # === temp ===


if __name__ == "__main__":
    main()
