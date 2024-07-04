import json
import os
from transformers import AutoTokenizer
from sources.models import sBERTRegressor, sBERTRegressorNew
import torch.nn as nn
import torch
from torch.optim import Adam
import re
import random
import numpy as np


def seed_everything(seed):
    random.seed(seed)  # Python 내장 random 모듈
    np.random.seed(seed)  # Numpy 모듈
    os.environ["PYTHONHASHSEED"] = str(seed)  # 환경 변수
    torch.manual_seed(seed)  # CPU를 위한 시드 고정
    torch.cuda.manual_seed(seed)  # GPU를 위한 시드 고정
    torch.cuda.manual_seed_all(seed)  # 멀티 GPU를 위한 시드 고정
    torch.backends.cudnn.deterministic = True  # CUDNN 옵티마이저 고정
    torch.backends.cudnn.benchmark = False  # 벤치마크 모드 비활성화
    torch.backends.cudnn.enabled = False  # cudnn 비활성화


def model_identification(args):
    model_name = args.config["model"]
    cleaned_model_name = re.sub(r"\W+", "", model_name).lower()
    if cleaned_model_name == "sbert":
        return "sBERT"
    elif cleaned_model_name == "sbertnew":
        return "sBERTNew"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_tokenizer(args):
    if args.model == "sBERT":
        model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = sBERTRegressor(model_name, args.config["is_freeze"])
    elif args.model == "sBERTNew":
        model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = sBERTRegressorNew(model_name, args.config["is_freeze"])
    else:
        raise ValueError(
            f"Unknown model: {args.config['model']}\tPossible Option: [sbert]"
        )
    return model.to(args.device), tokenizer


def get_optimizer(config, model):
    if config["optimizer"].lower() == "adam":
        return Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"].lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def get_criterion(config):
    crt = config["criterion"].lower().strip()
    if crt == "mse":
        return nn.MSELoss()
    elif crt == "mae":
        return nn.L1Loss()
    elif crt == "rmse":
        return RMSELoss()
    else:
        raise ValueError(f"Unknown criterion: {config['criterion']}")


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


def start_message(args):
    device_str = str(args.device)
    print(
        f"──────────────────────── TRAINING SETTINGS ───────────────────────\n"
        f" DEVICE: {device_str}   \n"
        f" SEED: {args.config['seed']}\n"
        f" CONFIG FILE: {args.config_path}\n"
        f" ENVIRONMENT PATH: {args.config['env_path']}\n"
        f" MODEL NAME: {args.model}\n"
        f" TIMESTAMP: {args.time}\n"
        f"──────────────────────────────────────────────────────────────────"
    )
