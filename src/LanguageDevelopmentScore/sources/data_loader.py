import pandas as pd
import os
import torch
from huggingface_hub import hf_hub_download
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset


class CustomDataset(Dataset):
    def __init__(self, config):
        self.data_config = config["data"]
        repo_id = self.data_config["repo_id"]
        file_name = self.data_config["file_name"]
        api_key = os.getenv("HF_API_KEY")
        file_path = hf_hub_download(
            repo_id, file_name, repo_type="dataset", use_auth_token=api_key
        )
        data = pd.read_parquet(file_path)
        self.data = data_preprocessing(data)
        self.X = self.data[self.data_config["input"]].values.squeeze()
        self.y = self.data[self.data_config["output"]].values.squeeze()
        print(
            f"Setup Data: {len(self.X)} data, {len(self.data_config['input'])} features"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X = self.X[idx]
        y = torch.tensor(self.y[idx] * 10, dtype=torch.float32)
        return X, y


def collate_fn(batch, tokenizer, max_length):
    texts, targets = zip(*batch)
    tokenized_inputs = tokenizer(
        list(texts),  # batch 처리
        return_tensors="pt",
        max_length=max_length,
        padding="max_length",
        truncation=True,
    )
    input_ids = tokenized_inputs["input_ids"]
    attention_mask = tokenized_inputs["attention_mask"]
    targets = torch.stack(targets)  # 타겟들을 텐서로 묶기
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }, targets


def get_loader(config, tokenizer):
    dataset = CustomDataset(config)
    y_stratify = dataset.y.astype(int).ravel()

    train_idx, valid_idx = train_test_split(
        range(len(dataset)),
        test_size=config["valid_size"],
        # stratify=y_stratify,
        random_state=config["seed"],
    )

    train_dataset = Subset(dataset, train_idx)
    valid_dataset = Subset(dataset, valid_idx)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, tokenizer, config["max_length"]),
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, tokenizer, config["max_length"]),
    )

    return train_dataloader, valid_dataloader


def data_preprocessing(data):
    """필요하다면 구현"""
    return data[:100]
