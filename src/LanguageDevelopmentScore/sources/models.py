from transformers import AutoModel
import torch.nn as nn
import torch
import torch.nn.functional as F
import re


class BERTRegressorDeep(nn.Module):
    def __init__(self, model_name, is_freeze=False, is_sigmoid=False, pooling="mean"):
        super(BERTRegressorDeep, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.attention_pooling = AttentionLayer(
            hidden_size=self.bert.config.hidden_size
        )  # BERT hidden size
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.regressor = nn.Linear(32, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

        self.sigmoid_scaling = is_sigmoid
        self.pooling = re.sub(r"\W+", "", pooling).lower()
        if is_freeze:
            print("**PRETRAINED MODEL FREEZE**")
            for param in self.bert.parameters():
                param.requires_grad = False

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        if "mean" in self.pooling:
            one_embedding = self.mean_pooling(token_embeddings, attention_mask)
        elif "attention" in self.pooling:
            one_embedding = self.attention_pooling(token_embeddings, attention_mask)
        x = self.activation(self.fc1(one_embedding))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        x = self.dropout(x)
        x = self.activation(self.fc5(x))
        x = self.dropout(x)
        x = self.activation(self.fc6(x))
        x = self.dropout(x)
        regression_output = self.regressor(x)
        if self.sigmoid_scaling:
            regression_output = torch.sigmoid(regression_output)
        return regression_output.squeeze()


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
    def __init__(self, model_name, is_freeze=False, is_sigmoid=False, pooling="mean"):
        super(RoBERTaRegressorNew, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.attention_pooling = AttentionLayer(
            hidden_size=self.bert.config.hidden_size
        )  # BERT hidden size
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.regressor = nn.Linear(64, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

        self.sigmoid_scaling = is_sigmoid
        self.pooling = re.sub(r"\W+", "", pooling).lower()
        if is_freeze:
            print("**PRETRAINED MODEL FREEZE**")
            for param in self.bert.parameters():
                param.requires_grad = False

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        if "mean" in self.pooling:
            one_embedding = self.mean_pooling(token_embeddings, attention_mask)
        elif "attention" in self.pooling:
            one_embedding = self.attention_pooling(token_embeddings, attention_mask)
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


class RoBERTaRegressorDeep(nn.Module):
    def __init__(self, model_name, is_freeze=False, is_sigmoid=False, pooling="mean"):
        super(RoBERTaRegressorDeep, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.attention_pooling = AttentionLayer(
            hidden_size=self.bert.config.hidden_size
        )

        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.bn1 = nn.BatchNorm1d(1024)

        self.fc2 = nn.Linear(1024, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)

        self.fc4 = nn.Linear(256, 128)
        self.bn4 = nn.BatchNorm1d(128)

        self.fc5 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)

        self.fc6 = nn.Linear(64, 32)
        self.bn6 = nn.BatchNorm1d(32)

        self.regressor = nn.Linear(32, 1)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

        self.sigmoid_scaling = is_sigmoid
        self.pooling = re.sub(r"\W+", "", pooling).lower()
        if is_freeze:
            print("**PRETRAINED MODEL FREEZE**")
            for param in self.bert.parameters():
                param.requires_grad = False

        self.fc_layers = nn.Sequential(
            self.fc1,
            self.bn1,
            self.activation,
            self.dropout,
            self.fc2,
            self.bn2,
            self.activation,
            self.dropout,
            self.fc3,
            self.bn3,
            self.activation,
            self.dropout,
            self.fc4,
            self.bn4,
            self.activation,
            self.dropout,
            self.fc5,
            self.bn5,
            self.activation,
            self.dropout,
            self.fc6,
            self.bn6,
            self.activation,
            self.dropout,
            self.regressor,
        )

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
        token_embeddings[input_mask_expanded == 0] = float("-inf")
        max_pooled = torch.max(token_embeddings, dim=1)[0]
        return max_pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state

        if "mean" in self.pooling:
            one_embedding = self.mean_pooling(token_embeddings, attention_mask)
        elif "max" in self.pooling:
            one_embedding = self.max_pooling(token_embeddings, attention_mask)
        elif "attention" in self.pooling:
            one_embedding = self.attention_pooling(token_embeddings, attention_mask)

        x = self.fc_layers(one_embedding)

        if self.sigmoid_scaling:
            x = torch.sigmoid(x)
        return x.squeeze()


class RoBERTaRegressorDeep0718(nn.Module):
    def __init__(self, model_name, is_freeze=False, is_sigmoid=False, pooling="mean"):
        super(RoBERTaRegressorDeep0718, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.attention_pooling = AttentionLayer(
            hidden_size=self.bert.config.hidden_size
        )  # BERT hidden size
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 32)
        self.regressor = nn.Linear(32, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

        self.sigmoid_scaling = is_sigmoid
        self.pooling = re.sub(r"\W+", "", pooling).lower()
        if is_freeze:
            print("**PRETRAINED MODEL FREEZE**")
            for param in self.bert.parameters():
                param.requires_grad = False

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand_as(token_embeddings)
        token_embeddings[input_mask_expanded == 0] = float("-inf")
        max_pooled = torch.max(token_embeddings, dim=1)[0]
        return max_pooled

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        if "cls" in self.pooling:
            one_embedding = token_embeddings[:, 0, :]
        if "mean" in self.pooling:
            one_embedding = self.mean_pooling(token_embeddings, attention_mask)
        elif "max" in self.pooling:
            one_embedding = self.max_pooling(token_embeddings, attention_mask)
        elif "attention" in self.pooling:
            one_embedding = self.attention_pooling(token_embeddings, attention_mask)
        x = self.activation(self.fc1(one_embedding))
        x = self.dropout(x)
        x = self.activation(self.fc2(x))
        x = self.dropout(x)
        x = self.activation(self.fc3(x))
        x = self.dropout(x)
        x = self.activation(self.fc4(x))
        x = self.dropout(x)
        x = self.activation(self.fc5(x))
        x = self.dropout(x)
        x = self.activation(self.fc6(x))
        x = self.dropout(x)
        regression_output = self.regressor(x)
        if self.sigmoid_scaling:
            regression_output = torch.sigmoid(regression_output)
        return regression_output.squeeze()


class RoBERTaRegressor0709(nn.Module):
    def __init__(self, model_name, is_freeze=False):
        super(RoBERTaRegressor0709, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.fc1 = nn.Linear(self.bert.config.hidden_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        self.regressor = nn.Linear(64, 1)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

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
        return regression_output.squeeze()


# class RoBERTaRegressorNew(nn.Module):
#     def __init__(self, model_name, is_freeze=False, is_sigmoid=False):
#         super(RoBERTaRegressorNew, self).__init__()
#         self.bert = AutoModel.from_pretrained(model_name)
#         self.fc1 = nn.Linear(self.bert.config.hidden_size, 1024)
#         self.fc2 = nn.Linear(1024, 512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc4 = nn.Linear(256, 128)
#         self.fc5 = nn.Linear(128, 64)
#         self.fc6 = nn.Linear(64, 32)
#         self.regressor = nn.Linear(32, 1)
#         self.activation = nn.GELU()
#         self.dropout = nn.Dropout(0.3)

#         self.sigmoid_scaling = is_sigmoid
#         if is_freeze:
#             print("**PRETRAINED MODEL FREEZE**")
#             for param in self.bert.parameters():
#                 param.requires_grad = False

#     def mean_pooling(self, model_output, attention_mask):
#         token_embeddings = model_output.last_hidden_state
#         input_mask_expanded = (
#             attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         )
#         sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
#         sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
#         return sum_embeddings / sum_mask

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         one_embedding = self.mean_pooling(outputs, attention_mask)
#         x = self.activation(self.fc1(one_embedding))
#         x = self.dropout(x)
#         x = self.activation(self.fc2(x))
#         x = self.dropout(x)
#         x = self.activation(self.fc3(x))
#         x = self.dropout(x)
#         x = self.activation(self.fc4(x))
#         x = self.dropout(x)
#         x = self.activation(self.fc5(x))
#         x = self.dropout(x)
#         x = self.activation(self.fc6(x))
#         x = self.dropout(x)
#         regression_output = self.regressor(x)
#         if self.sigmoid_scaling:
#             regression_output = torch.sigmoid(regression_output)
#         return regression_output.squeeze()


class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.randn(hidden_size, 1))

    def forward(self, token_embeddings, attention_mask):
        attention_scores = torch.matmul(
            token_embeddings, self.attention_weights
        ).squeeze(-1)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)
        attention_weights = F.softmax(attention_scores, dim=1)

        weighted_sum = torch.matmul(
            attention_weights.unsqueeze(1), token_embeddings
        ).squeeze(1)
        return weighted_sum
