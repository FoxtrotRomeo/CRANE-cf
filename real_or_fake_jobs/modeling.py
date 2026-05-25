"""modeling.py — MultimodalClassifier for real-or-fake job postings.

Extracted from hparam_search.py so it can be imported without triggering
that script's module-level argument parsing and data loading.
"""

import torch
import torch.nn as nn
from transformers import AutoModel

TEXT_MODEL = "distilbert-base-uncased"


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

        # Shared encoder — one forward pass per text field, weights shared
        self.text_encoder = AutoModel.from_pretrained(TEXT_MODEL)
        enc_dim = self.text_encoder.config.hidden_size

        # Independent projection head per text field
        def _proj():
            return nn.Sequential(
                nn.Linear(enc_dim, text_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        self.proj_description     = _proj()
        self.proj_company_profile = _proj()
        self.proj_requirements    = _proj()

        self.tab_head   = TabHead(d_tab, tab_hidden_dims, dropout)
        fusion_dim      = 3 * text_hidden_dim + self.tab_head.out_dim
        self.classifier = nn.Linear(fusion_dim, n_classes)

    def forward(self, desc_ids, desc_mask, profile_ids, profile_mask,
                reqs_ids, reqs_mask, X_static):
        # Stack all three texts along the batch dim → one encoder call (3×B, seq_len)
        all_ids  = torch.cat([desc_ids,  profile_ids,  reqs_ids],  dim=0)
        all_mask = torch.cat([desc_mask, profile_mask, reqs_mask], dim=0)
        cls_all  = self.text_encoder(input_ids=all_ids,
                                     attention_mask=all_mask).last_hidden_state[:, 0]
        # Split back into per-field CLS tokens, then project independently
        B = desc_ids.size(0)
        desc_emb, profile_emb, reqs_emb = cls_all.split(B, dim=0)
        desc_emb    = self.proj_description(desc_emb)
        profile_emb = self.proj_company_profile(profile_emb)
        reqs_emb    = self.proj_requirements(reqs_emb)
        tab_emb     = self.tab_head(X_static)
        return self.classifier(torch.cat([desc_emb, profile_emb, reqs_emb, tab_emb], dim=1))
