from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

import src.utils as utils

# self.bert = BertModel.from_pretrained("bert-base-uncased")


def create_bert_classifier(
    bert_pretrained_name: str,
    classifier_type: str,
    classifier_hidden_size: int,
    classifier_drop_out: float = 0,
    freeze_bert=True,
    classifier_init_state_path: str = None,
    random_state: int = 42,
) -> BertClassifier:
    if random_state is not None:
        utils.set_seed(random_state)  # Set seed for reproducibility
    device = utils.get_device()

    bert_classifier = BertClassifier(
        pretrained_name=bert_pretrained_name,
        freeze_bert=freeze_bert,
        logistic_hidden_size=classifier_hidden_size,
        classifier_drop_out=classifier_drop_out,
        classifier_type=classifier_type,
    )

    if classifier_init_state_path:
        bert_classifier.classifier.load_state_dict(
            torch.load(classifier_init_state_path)
        )

    return bert_classifier.to(device)


class BertClassifier(nn.Module):
    loss_fn = nn.CrossEntropyLoss()

    def __init__(
        self,
        pretrained_name: str = "distilbert-base-uncased",
        freeze_bert=True,
        logistic_hidden_size=20,
        classifier_drop_out: float = 0,
        classifier_type: str = "single-fc",
    ):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, logistic_hidden_size, 2

        self.bert = AutoModel.from_pretrained(pretrained_name)

        if classifier_type == "single-fc":
            self.classifier = nn.Sequential(
                nn.Dropout(p=classifier_drop_out),
                nn.Linear(D_in, D_out),
            )
            nn.init.xavier_uniform_(self.classifier[-1].weight)
        elif classifier_type == "double-fc":
            self.classifier = nn.Sequential(
                nn.Linear(D_in, H),
                nn.Tanh(),
                nn.Dropout(p=classifier_drop_out),
                nn.Linear(H, D_out),
            )
            nn.init.xavier_uniform_(self.classifier[0].weight)
        else:
            raise ValueError(f"{classifier_type} not recognized")

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None):
        if input_ids is None:
            outputs = self.bert(
                inputs_embeds=inputs_embeds, attention_mask=attention_mask
            )
        else:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        last_hidden_state = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state)
        return logits

    def compute_loss(self, logits, labels) -> float:
        return self.loss_fn(logits, labels)

    def compute_accuracy(self, logits, labels) -> float:
        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()
        return (preds == labels).cpu().numpy().mean() * 100
