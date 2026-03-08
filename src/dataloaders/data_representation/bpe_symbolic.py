import numpy as np
from typing import Tuple, Dict, List, Any

import bpe

class BPESymbolic:
    def __init__(self, vocab, merges, args):
        self.vocab, self.merges = vocab, merges
        self.args = args
        self.symbols = list("abcdefghijklmnopqrstuvwxyz")
        self.len_symbols = len(self.symbols)
        self.symbols_arr = np.asarray(self.symbols, dtype="U1")

        base_vocab_size = len(self.vocab)

        base_special = ["pad_id"]
        if self.args.objective == "autoregressive":
            extra_special = ["bos_id", "eos_id"]
        else:
            extra_special = []

        self.special_tokens = base_special + extra_special
        self.special_tokens_map = set()

        for i, token_name in enumerate(self.special_tokens):
            token_id = base_vocab_size + i
            setattr(self.args, token_name, token_id)
            self.special_tokens_map.update([token_id])

        # print("SPECIAL TOKENS", self.special_tokens)
        # print("SPECIAL TOKEN MAP", self.special_tokens_map)
        self.vocab_size = base_vocab_size + len(self.special_tokens)

    def __call__(self, data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        report = data.get("report", "")
        if self.args.condition:
            data["ecg"] = data["ecg"][self.args.condition_lead]
        per_mod_tokens, mn, mx = self.signal_to_bpe_tokens(data["ecg"])
        skip_pad = "eval" in self.args.mode
        seq = self.build_autoregressive_sequence(per_mod_tokens)
        seq = seq if skip_pad else self.pad_tokenized_data(seq)
        return {
            "transformed_data": seq,
            "min": mn,
            "max": mx,
            "report": report,
        }

    def signal_to_bpe_tokens(self, ecg: np.ndarray):
        per_mod_tokens: List[List[int]] = []
        clipped_arr, mn, mx = self.normalize(ecg)
        quantized_arr = self.quantize(clipped_arr)
        symbols = self.quantized_to_symbol(quantized_arr)
        joined_symbols = "".join(symbols.ravel())
        bpe_tokens = bpe.encode_symbol(joined_symbols, self.merges)
        per_mod_tokens.append(bpe_tokens)
        return per_mod_tokens, mn, mx

    def build_autoregressive_sequence(self, per_mod_tokens: List[List[int]]) -> List[int]:
        flat: List[int] = []
        for tokens in per_mod_tokens:
            flat.extend(tokens)
        return [self.args.bos_id] + flat + [self.args.eos_id]

    def pad_tokenized_data(self, tokenized_data: List[int]) -> List[int]:
        n = len(tokenized_data)
        if n <= self.args.bpe_symbolic_len:
            return [self.args.pad_id] * (self.args.bpe_symbolic_len - n) + tokenized_data
        return tokenized_data[:self.args.bpe_symbolic_len -1] + [tokenized_data[-1]]

    def quantize(self, clipped_normalized: np.ndarray) -> np.ndarray:
        return np.minimum(
            np.floor(clipped_normalized * self.len_symbols),
            self.len_symbols - 1,
        ).astype(np.uint8)

    def quantized_to_symbol(self, quantized_signal: np.ndarray) -> np.ndarray:
        return self.symbols_arr[quantized_signal]

    def normalize(self, arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
        arr = arr.astype(np.float32, copy=False)
        mn = float(arr.min())
        mx = float(arr.max())
        denom = mx - mn + self.args.norm_eps
        return np.clip((arr - mn) / denom, 0.0, 1.0), mn, mx

    def denormalize(self, arr: np.ndarray, mn: float, mx: float) -> np.ndarray:
        arr = arr.astype(np.float32, copy=False)
        denom = mx - mn + self.args.norm_eps
        return arr * denom + mn

    def decode(self, tokens, mn: float, mx: float, shape=None) -> np.ndarray:
        t = [x for x in tokens if x not in self.special_tokens_map]
        s = bpe.decode_symbol(t, self.vocab)
        sym2idx = {c: i for i, c in enumerate(self.symbols)}
        q = np.fromiter((sym2idx.get(c, 0) for c in s), dtype=np.int32)
        x = np.clip((q.astype(np.float32) + 0.5) / float(self.len_symbols), 0.0, 1.0)

        if shape is not None:
            n = int(np.prod(shape))
            if x.size < n:
                x = np.pad(x, (0, n - x.size))
            else:
                x = x[:n]
            x = x.reshape(shape)

        return self.denormalize(x, mn, mx)
