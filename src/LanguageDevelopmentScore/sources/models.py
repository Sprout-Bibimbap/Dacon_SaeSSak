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
        if self.version == "V2":
            one_embedding = self.mean_pooling(outputs, attention_mask)  # [B, All, h]
        elif not self.version:
            one_embedding = outputs.last_hidden_state[:, 0, :]  # [B, CLS Index, h]
        else:
            raise ValueError(f"Version is not valid. Version: {self.version}")
        x = self.activation(self.fc1(one_embedding))
        x = self.activation(self.fc2(x))
        regression_output = self.regressor(x)
        return regression_output.squeeze()


class RoBERTaRegressor(nn.Module):
    def __init__(self, model_name, is_freeze) -> None:
        super(RoBERTaRegressor, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.regressor = nn.Linear(64, 1)
        self.activation = nn.GELU()

        if is_freeze:
            print("**PRETRAINED MODEL FREEZE**")
            for param in self.bert.parameters():
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
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        one_embedding = self.mean_pooling(outputs, attention_mask)  # [B, All, h]
        x = self.activation(self.fc1(one_embedding))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        regression_output = self.regressor(x)
        return regression_output.squeeze()


class RoBERTaRegressorNew(nn.Module):
    def __init__(self, model_name, is_freeze=False, is_sigmoid=False):
        super(RoBERTaRegressorNew, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.regressor = nn.Linear(64, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

        self.sigmoid_scaling = is_sigmoid
        if is_freeze:
            print("**PRETRAINED MODEL FREEZE**")
            for param in self.bert.parameters():
                param.requires_grad = False

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        one_embedding = self.mean_pooling(outputs, attention_mask)
        x = self.activation(self.fc1(one_embedding))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        regression_output = self.regressor(x)
        if self.sigmoid_scaling:
            regression_output = torch.sigmoid(regression_output)
        return regression_output.squeeze()
