"""modeling.py — MultimodalClassifier for Long COVID tweets.

Extracted from hparam_search.py so it can be imported without triggering
that script's module-level argument parsing and data loading.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

TEXT_MODEL = "cardiffnlp/twitter-xlm-roberta-base"


class TabHead(nn.Module):
    """MLP with variable depth; out_dim is the last hidden width."""
    def __init__(self, d_in, hidden_dims, dropout):
        super().__init__()
        layers, in_dim = [], d_in
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        self.net     = nn.Sequential(*layers)
        self.out_dim = in_dim

    def forward(self, x):
        return self.net(x)


class MultimodalClassifier(nn.Module):
    def __init__(self, d_tab, n_classes, tab_hidden_dims, text_hidden_dim, dropout):
        super().__init__()

        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
        text_enc_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Sequential(
            nn.Linear(text_enc_dim, text_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.tab_head  = TabHead(d_tab, tab_hidden_dims, dropout)
        fusion_dim     = text_hidden_dim + self.tab_head.out_dim
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, input_ids, attention_mask, X_static):
        cls_emb  = self.text_encoder(input_ids=input_ids,
                                     attention_mask=attention_mask).last_hidden_state[:, 0]
        text_emb = self.text_proj(cls_emb)
        tab_emb  = self.tab_head(X_static)
        return self.classifier(torch.cat([text_emb, tab_emb], dim=1))
