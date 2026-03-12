from dataclasses import dataclass
from typing import Optional
from transformers import TimesFm2_5ModelForPrediction

import torch
import torch.nn as nn

@dataclass
class TimesFM25Config:
    model_name: str = "google/timesfm-2.5-200m-transformers"

@dataclass
class TimesFM25Output:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None


class TimesFM25(nn.Module):
    def __init__(self, cfg: TimesFM25Config):
        super().__init__()
        self.cfg = cfg
        self.timesfm25 = TimesFm2_5ModelForPrediction.from_pretrained("google/timesfm-2.5-200m-transformers")
    
    def forward(self,):
        raise NotImplementedError
    
    def generate(self,):
        pass
