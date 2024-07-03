from transformers import AutoModel, AutoTokenizer
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
