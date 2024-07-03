import json
import os
from transformers import AutoTokenizer, AutoModel
from sources.models import sBERTRegressor
import torch.nn as nn
import torch
from torch.optim import Adam
import re


def model_identification(args):
    model_name = args.config["model"]
    cleaned_model_name = re.sub(r"\W+", "", model_name).lower()
    if cleaned_model_name == "sbert":
        return "sBERT"
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_model_tokenizer(args):
    if args.model == "sBERT":
        model_name = "snunlp/KR-SBERT-V40K-klueNLI-augSTS"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = sBERTRegressor(model_name, args.config["is_freeze"])
    else:
        raise ValueError(
            f"Unknown model: {args.config['model']}\tPossible Option: [sbert]"
        )
    return model.to(args.device), tokenizer


def get_criterion(config):
    crt = config["criterion"].lower().strip()
    if crt == "mse":
        return nn.MSELoss()
    elif crt == "mae":
        return nn.L1Loss()
    else:
        raise ValueError(f"Unknown criterion: {config['criterion']}")


def get_optimizer(config, model):
    if config["optimizer"].lower() == "adam":
        return Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"].lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config["learning_rate"])
    else:
        raise ValueError(f"Unknown optimizer: {config['optimizer']}")
