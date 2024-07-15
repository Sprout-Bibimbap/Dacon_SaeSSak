import json
import os
from transformers import AutoTokenizer
from sources.models import (
    sBERTRegressor,
    sBERTRegressorNew,
    RoBERTaRegressor,
    RoBERTaRegressorNew,
)
import torch.nn as nn
import torch
from torch.optim import Adam, SGD, AdamW
import re
import random
import numpy as np
from torch.optim.lr_scheduler import (
    StepLR,
    ExponentialLR,
    ReduceLROnPlateau,
    CosineAnnealingLR,
)


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
        return "sBERT", ""
    elif cleaned_model_name == "sbertnew":
        return "sBERTNew", ""
    elif cleaned_model_name == "sbertnewv2":
        return "sBERTNew", "V2"
    elif cleaned_model_name == "robertabase":
        return "RoBERTa-Base", ""
    elif cleaned_model_name == "robertalarge":
        return "RoBERTa-Large", ""
    elif cleaned_model_name == "robertanewbase":
        return "RoBERTaNew-Base", ""
    elif cleaned_model_name == "robertanewlarge":
        return "RoBERTaNew-Large", ""
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
        model = sBERTRegressorNew(
            model_name, args.config["is_freeze"], version=args.version
        )
    elif "RoBERTa" in args.model and "New" not in args.model:
        if args.model == "RoBERTa-Base":
            model_name = "klue/roberta-base"
        elif args.model == "RoBERTa-Large":
            model_name = "klue/roberta-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = RoBERTaRegressor(model_name, args.config["is_freeze"])
    elif "RoBERTa" in args.model and "New" in args.model:
        if args.model == "RoBERTaNew-Base":
            model_name = "klue/roberta-base"
        elif args.model == "RoBERTaNew-Large":
            model_name = "klue/roberta-large"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = RoBERTaRegressorNew(
            model_name,
            args.config["is_freeze"],
            args.config["sigmoid_scaling"],
            args.config["pooling_method"],
        )
    else:
        raise ValueError(
            f"Unknown model: {args.config['model']}\tPossible Option: [sBERT, sBERTNew, sBERTNewV2, RoBERTa-Base, RoBERTa-Large, RoBERTaNew-Base, RoBERTaNew-Large]"
        )
    return model.to(args.device), tokenizer


def get_optimizer(config, model):
    optimizer_type = config["optimizer"].lower()
    lr = config["learning_rate"]

    if optimizer_type == "adam":
        return Adam(model.parameters(), lr=lr)
    elif optimizer_type == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    elif optimizer_type == "sgd":
        return SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")


def get_criterion(config):
    crt = config["criterion"].lower().strip()

    if crt == "mse":
        criterion = nn.MSELoss()
        print("Initialized Mean Squared Error Loss")

    elif crt == "mae":
        criterion = nn.L1Loss()
        print("Initialized Mean Absolute Error Loss")

    elif crt == "rmse":
        criterion = RMSELoss()  # 가정: RMSELoss는 사용자 정의 손실 함수
        print("Initialized Root Mean Squared Error Loss")

    elif crt == "huber":
        criterion = nn.HuberLoss()
        print("Initialized Huber Loss")

    else:
        raise ValueError(f"Unknown criterion: {config['criterion']}")

    return criterion


def get_scheduler(config, optimizer):
    scheduler_type = config["scheduler"].lower().strip()

    if scheduler_type == "steplr":
        # StepLR: 에폭마다 학습률을 일정 비율로 감소시킵니다.
        # step_size: 학습률을 감소시키는 주기
        # gamma: 학습률 감소율
        step_size = 8
        gamma = 0.1
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        print(f"Initialized StepLR with step_size={step_size} and gamma={gamma}")

    elif scheduler_type == "exponentiallr":
        # ExponentialLR: 에폭마다 학습률을 지수적으로 감소시킵니다.
        # gamma: 각 에폭마다 적용할 감소율
        gamma = 0.97
        scheduler = ExponentialLR(optimizer, gamma=gamma)
        print(f"Initialized ExponentialLR with gamma={gamma}")

    elif scheduler_type == "reducelronplateau":
        # ReduceLROnPlateau: 검증 손실이 개선되지 않을 때 학습률을 감소시킵니다.
        # mode: 'min' 또는 'max' (손실을 최소화하거나 정확도를 최대화할 때)
        # factor: 학습률 감소율
        # patience: 몇 에폭 동안 개선이 없을 때 학습률을 감소시킬지
        mode = "min"
        factor = 0.5
        patience = 6
        verbose = True
        scheduler = ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
        )
        print(
            f"Initialized ReduceLROnPlateau with mode={mode}, factor={factor}, patience={patience}, verbose={verbose}"
        )

    elif scheduler_type == "cosineannealinglr":
        # CosineAnnealingLR: 코사인 함수의 형태로 학습률을 조절합니다.
        # T_max: 하나의 학습률 주기 길이
        # eta_min: 학습률의 하한선
        T_max = 10
        eta_min = 0
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
        print(f"Initialized CosineAnnealingLR with T_max={T_max}, eta_min={eta_min}")

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    return scheduler


class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))


def start_message(args):
    device_str = str(args.device)
    version = ":" + args.version if args.version else args.version
    freeze = "[FREEZED]" if args.config["is_freeze"] else "[NOT FREEZED]"
    print(
        f"──────────────────────── TRAINING SETTINGS ───────────────────────\n"
        f" DEVICE: {device_str}   \n"
        f" SEED: {args.config['seed']}\n"
        f" CONFIG FILE: {args.config_path}\n"
        f" ENVIRONMENT PATH: {args.config['env_path']}\n"
        f" MODEL NAME: {args.model}{version} {freeze}\n"
        f" TIMESTAMP: {args.time}\n"
        f"──────────────────────────────────────────────────────────────────"
    )
