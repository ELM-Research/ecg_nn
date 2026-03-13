import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DecoderTransformerConfig:
    vocab_size: int
    pad_id: int
    d_model: int = 512
    n_heads: int = 8
    dim_ff: int = 2048
    num_layers: int = 12
    dropout: float = 0.1
    max_seq_len: int = 512
    flow_matching_head: bool = False
    flow_matching_loss_weight: float = 1.0


@dataclass
class DecoderTransformerOutput:
    loss: Optional[torch.Tensor]
    logits: torch.Tensor
    hidden_states: Optional[torch.Tensor] = None
    token_loss: Optional[torch.Tensor] = None
    flow_loss: Optional[torch.Tensor] = None
    flow_prediction: Optional[torch.Tensor] = None


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dim_ff, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.lin1 = nn.Linear(d_model, dim_ff)
        self.lin2 = nn.Linear(dim_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(
        self,
        x,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        y, _ = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = self.norm1(x + self.dropout(y))
        y = self.lin2(self.dropout_ff(F.gelu(self.lin1(x))))
        x = self.norm2(x + self.dropout(y))
        return x


class DecoderTransformer(nn.Module):
    def __init__(self, cfg: DecoderTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList([DecoderBlock(cfg.d_model, cfg.n_heads, cfg.dim_ff, cfg.dropout) for _ in range(cfg.num_layers)])
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.flow_matching_head = None
        self.flow_time_mlp = None
        if cfg.flow_matching_head:
            self.flow_in = nn.Linear(1, cfg.d_model)
            self.flow_hid = nn.Linear(cfg.d_model, cfg.d_model)
            self.flow_out = nn.Linear(cfg.d_model, 1)
            self.flow_time_mlp = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model),
                nn.SiLU(),
                nn.Linear(cfg.d_model, cfg.d_model),
            )
            self.flow_matching_head = nn.LayerNorm(cfg.d_model)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def _causal_mask(self, L: int, device: torch.device):
        return torch.triu(torch.ones(L, L, dtype=torch.bool, device=device), diagonal=1)

    def _embed(self, input_ids: torch.Tensor):
        bsz, seq_len = input_ids.size()
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        return self.dropout(x)

    @staticmethod
    def _timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(
            half, device=t.device, dtype=torch.float32
        ) / max(half, 1))
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding

    def _make_key_padding_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.eq(self.cfg.pad_id)

    def _decode_hidden(self, tgt_ids: torch.Tensor, tgt_key_padding_mask: torch.Tensor) -> torch.Tensor:
        x = self._embed(tgt_ids)
        causal_mask = self._causal_mask(tgt_ids.size(1), x.device)
        for layer in self.layers:
            x = layer(x, attn_mask=causal_mask, key_padding_mask=tgt_key_padding_mask)
        return x

    def _flow_predict(self, hidden_states: torch.Tensor, signal_values: torch.Tensor,
                      tgt_key_padding_mask: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        pooled_hidden = hidden_states.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        valid_counts = (~tgt_key_padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
        pooled_hidden = pooled_hidden.sum(dim=1) / valid_counts
        time_emb = self.flow_time_mlp(self._timestep_embedding(t, self.cfg.d_model))
        cond = self.flow_matching_head(pooled_hidden + time_emb)
        flow_h = self.flow_in(signal_values.unsqueeze(-1)) + cond.unsqueeze(1)
        flow_h = F.gelu(self.flow_hid(flow_h))
        return self.flow_out(flow_h).squeeze(-1)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        signal_values: Optional[torch.Tensor] = None,
        signal_mask: Optional[torch.Tensor] = None,
    ) -> DecoderTransformerOutput:
        if tgt_key_padding_mask is None:
            tgt_key_padding_mask = self._make_key_padding_mask(tgt_ids)
        x = self._decode_hidden(tgt_ids, tgt_key_padding_mask)
        logits = self.lm_head(x)
        token_loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        flow_loss, flow_prediction = None, None
        if self.flow_matching_head is not None and signal_values is not None:
            signal_values = signal_values.to(logits.device)
            if signal_values.dim() == 3:
                signal_values = signal_values.squeeze(-1)
            x1 = signal_values
            x0 = torch.randn_like(x1)
            t = torch.rand(x1.size(0), device=x1.device)
            xt = (1 - t[:, None]) * x0 + t[:, None] * x1
            flow_prediction = self._flow_predict(x, xt, tgt_key_padding_mask, t)
            flow_target = x1 - x0
            if signal_mask is not None:
                signal_mask = signal_mask.to(logits.device).float()
                sq = (flow_prediction - flow_target).pow(2) * signal_mask
                flow_loss = sq.sum() / signal_mask.sum().clamp(min=1.0)
            else:
                flow_loss = F.mse_loss(flow_prediction, flow_target)

        loss = token_loss
        if flow_loss is not None:
            loss = flow_loss * self.cfg.flow_matching_loss_weight if loss is None else loss + flow_loss * self.cfg.flow_matching_loss_weight
        return DecoderTransformerOutput(
            loss=loss,
            logits=logits,
            hidden_states=x,
            token_loss=token_loss,
            flow_loss=flow_loss,
            flow_prediction=flow_prediction,
        )

    @torch.inference_mode()
    def generate(
        self,
        tgt_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        do_sample: bool = False,
        return_new_only: bool = False,
        return_logits: bool = True,):
        out = tgt_ids
        start_len = out.size(1)
        all_logits = [] if return_logits else None

        for _ in range(max_new_tokens):
            if out.size(1) >= self.cfg.max_seq_len:
                break

            key_padding_mask = self._make_key_padding_mask(out)
            x = self._embed(out)
            causal_mask = self._causal_mask(out.size(1), x.device)
            for layer in self.layers:
                x = layer(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
            logits = self.lm_head(x[:, -1, :])

            if return_logits:
                all_logits.append(logits.clone())

            if temperature != 1.0:
                logits = logits / max(temperature, 1e-8)

            if top_k is not None and 0 < top_k < logits.size(-1):
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits = logits.masked_fill(logits < v[:, -1:], float("-inf"))

            if do_sample:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            out = torch.cat([out, next_id], dim=1)

        tokens = out[:, start_len:] if return_new_only else out

        if return_logits:
            out_logits = torch.stack(all_logits, dim=1)
            return tokens, out_logits
        return tokens

    @torch.inference_mode()
    def generate_signal(self, tgt_ids: torch.Tensor, signal_len: int, num_steps: int = 10) -> torch.Tensor:
        if self.flow_matching_head is None:
            raise ValueError("flow matching head is disabled")
        key_padding_mask = self._make_key_padding_mask(tgt_ids)
        hidden_states = self._decode_hidden(tgt_ids, key_padding_mask)
        signal = torch.randn(tgt_ids.size(0), signal_len, device=tgt_ids.device)
        dt = 1.0 / max(num_steps, 1)
        for i in range(num_steps):
            t = torch.full((tgt_ids.size(0),), (i + 0.5) * dt, device=tgt_ids.device)
            v = self._flow_predict(hidden_states, signal, key_padding_mask, t)
            signal = signal + v * dt
        return signal

    def resize_embeddings(self, new_vocab_size: int):
        print("Resizing Embeddings")
        old_vocab_size = self.cfg.vocab_size
        print("Old Vocab Size", old_vocab_size)
        if new_vocab_size == old_vocab_size:
            return

        dtype = self.token_emb.weight.dtype
        old_token_emb = self.token_emb
        self.token_emb = nn.Embedding(new_vocab_size, self.cfg.d_model,
                                      padding_idx=self.cfg.pad_id, dtype = dtype)
        nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_emb.weight[:old_vocab_size] = old_token_emb.weight

        old_lm_head = self.lm_head
        self.lm_head = nn.Linear(self.cfg.d_model, new_vocab_size,
                                 bias=False, dtype= dtype)
        nn.init.xavier_uniform_(self.lm_head.weight)
        with torch.no_grad():
            self.lm_head.weight[:old_vocab_size] = old_lm_head.weight
        print("New Vocab Size", new_vocab_size)
        self.cfg.vocab_size = new_vocab_size
