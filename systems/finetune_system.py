import json
import os
import pickle as pk
from importlib import import_module

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aasist.models.AASIST import Model as AASISTModel
from ECAPATDNN.model import ECAPA_TDNN
from metrics import get_all_EERs
from utils import load_parameters


class FinetuneSystem(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters()

        # 1. Tải các model trích xuất đặc trưng
        self.asv_encoder = ECAPA_TDNN(C=1024)
        load_parameters(self.asv_encoder.state_dict(), config.ecapa_weight)

        with open(config.aasist_config, "r") as f:
            aasist_config = json.loads(f.read())
        self.cm_encoder = AASISTModel(aasist_config["model_config"])
        load_parameters(self.cm_encoder.state_dict(), config.aasist_weight)

        # 2. Tải model fusion
        _model_module = import_module("models.{}".format(config.model_arch))
        self.fusion_model = getattr(_model_module, "Model")(config.model_config)

        # 3. Định nghĩa hàm loss
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.FloatTensor(config.loss_weight)
        )

        # 4. Đóng băng các model trích xuất đặc trưng ban đầu
        print("--- Freezing feature extractors initially ---")
        for param in self.asv_encoder.parameters():
            param.requires_grad = False
        for param in self.cm_encoder.parameters():
            param.requires_grad = False

    def forward(self, enr_wav, tst_wav):
        # Trích xuất embedding "on-the-fly"
        # Bỏ `with torch.no_grad()` để gradient có thể được tính khi model rã đông
        embd_asv_enr = self.asv_encoder(enr_wav, aug=False)
        embd_asv_tst = self.asv_encoder(tst_wav, aug=False)
        embd_cm_tst, _ = self.cm_encoder(tst_wav)

        # Đưa embedding vào model fusion
        return self.fusion_model(embd_asv_enr, embd_asv_tst, embd_cm_tst)

    def training_step(self, batch, batch_idx):
        enr_wav, tst_wav, label = batch
        # Gọi thẳng self.forward() để đảm bảo gradient được tính toán đúng
        pred = self(enr_wav, tst_wav)
        loss = self.loss(pred, label)
        self.log("trn_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        enr_wav, tst_wav, key = batch
        # Chuyển model về chế độ eval để có kết quả chính xác
        self.asv_encoder.eval()
        self.cm_encoder.eval()
        self.fusion_model.eval()
        
        with torch.no_grad():
             pred = self(enr_wav, tst_wav)
        pred = torch.softmax(pred, dim=-1)
        return {"pred": pred, "key": key}

    def validation_epoch_end(self, outputs):
        log_dict = {}
        preds, keys = [], []
        for output in outputs:
            preds.append(output["pred"])
            keys.extend(list(output["key"]))

        preds = torch.cat(preds, dim=0)[:, 1].detach().cpu().numpy()
        sasv_eer, sv_eer, spf_eer = get_all_EERs(preds=preds, keys=keys)

        log_dict["sasv_eer_dev"] = sasv_eer
        log_dict["sv_eer_dev"] = sv_eer
        log_dict["spf_eer_dev"] = spf_eer

        self.log_dict(log_dict)
        print(f"Epoch {self.current_epoch}: sasv_eer_dev = {sasv_eer:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            [
                {"params": self.fusion_model.parameters(), "lr": self.config.optim.lr_fusion},
                {"params": self.asv_encoder.parameters(), "lr": self.config.optim.lr_extractor},
                {"params": self.cm_encoder.parameters(), "lr": self.config.optim.lr_extractor},
            ],
            weight_decay=self.config.optim.wd,
        )
        return optimizer

    def on_train_epoch_start(self):
        # Logic cho Gradual Unfreezing
        if self.current_epoch == self.config.get("unfreeze_epoch", 5):
            print("\n--- Unfreezing feature extractors ---")
            for param in self.asv_encoder.parameters():
                param.requires_grad = True
            for param in self.cm_encoder.parameters():
                param.requires_grad = True

    # --- BỔ SUNG CÁC HÀM TẢI DỮ LIỆU ---

    def setup(self, stage=None):
        # Tải metadata và gán dataloader functions
        with open(self.config.dirs.spk_meta + "spk_meta_trn.pk", "rb") as f:
            self.spk_meta_trn = pk.load(f)
        with open(self.config.dirs.spk_meta + "spk_meta_dev.pk", "rb") as f:
            self.spk_meta_dev = pk.load(f)
        
        module = import_module("dataloaders." + self.config.dataloader)
        self.ds_func_trn = getattr(module, "get_trnset")
        self.ds_func_dev = getattr(module, "get_dev_evalset") # Giả định có hàm này

    def train_dataloader(self):
        # Giả định base_dir được cấu hình đúng
        base_dir = "/kaggle/input/vlsp-vsasv-datasets/vlsp2025/vlsp2025/train/"
        self.train_ds = self.ds_func_trn(self.spk_meta_trn, base_dir)
        return DataLoader(
            self.train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.loader.n_workers,
        )

    def val_dataloader(self):
        with open(self.config.dirs.sasv_dev_trial, "r") as f:
            sasv_dev_trial = f.readlines()
        
        # Giả định base_dir cho tập dev
        base_dir = "/kaggle/input/vlsp-vsasv-datasets/vlsp2025/vlsp2025/train/" # Sửa nếu khác
        self.dev_ds = self.ds_func_dev(sasv_dev_trial, base_dir)
        return DataLoader(
            self.dev_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.loader.n_workers,
        )