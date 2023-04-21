import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

# self.bert = BertModel.from_pretrained("bert-base-uncased")


class BertClassifier(nn.Module):
    def __init__(
        self, pretrained_name: str = "distilbert-base-uncased", freeze_bert=False
    ):
        super(BertClassifier, self).__init__()
        D_in, H, D_out = 768, 20, 2

        self.bert = AutoModel.from_pretrained(pretrained_name)
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H), nn.ReLU(), nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state)
        return logits


def bert_predict(model, test_dataloader, device):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        if hasattr(logits, "logits"):
            logits = logits.logits
        all_logits.append(logits)

    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs
