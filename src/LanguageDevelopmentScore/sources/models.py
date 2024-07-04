from transformers import AutoModel
import torch.nn as nn
import torch


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
    def __init__(self, model_name, is_freeze, version) -> None:
        super(sBERTRegressorNew, self).__init__()
        self.version = version
        self.sbert = AutoModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(self.sbert.config.hidden_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.regressor = nn.Linear(64, 1)
        self.activation = nn.GELU()

        if is_freeze:
            print("**PRETRAINED MODEL FREEZE**")
            for param in self.sbert.parameters():
                param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = (
            model_output.last_hidden_state
        )  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.sbert(input_ids=input_ids, attention_mask=attention_mask)
        if self.version == 1:
            one_embedding = outputs.last_hidden_state[:, 0, :]  # [B, CLS Index, h]
        elif self.version == 2:
            one_embedding = self.mean_pooling(outputs, attention_mask)  # [B, All, h]
        x = self.activation(self.fc1(one_embedding))
        x = self.activation(self.fc2(x))
        regression_output = self.regressor(x)
        return regression_output.squeeze()
