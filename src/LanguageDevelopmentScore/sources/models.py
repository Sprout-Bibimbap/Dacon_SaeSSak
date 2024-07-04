from transformers import AutoModel
import torch.nn as nn


class sBERTRegressor(nn.Module):
    def __init__(self, model_name, is_freeze) -> None:
        super(sBERTRegressor, self).__init__()
        self.sbert = AutoModel.from_pretrained(model_name)
        self.regressor = nn.Linear(self.sbert.config.hidden_size, 1)

        if is_freeze:
            for param in self.sbert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.sbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, CLS Index, h]
        regression_output = self.regressor(cls_embedding)
        return regression_output.squeeze()


class sBERTRegressorNew(nn.Module):
    def __init__(self, model_name, is_freeze) -> None:
        super(sBERTRegressorNew, self).__init__()
        self.sbert = AutoModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(self.sbert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.regressor = nn.Linear(64, 1)
        self.activation = nn.ReLU()

        if is_freeze:
            for param in self.sbert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.sbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [B, CLS Index, h]
        x = self.activation(self.fc1(cls_embedding))
        x = self.activation(self.fc2(x))
        regression_output = self.regressor(x)
        return regression_output.squeeze()
