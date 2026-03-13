import numpy as np
import torch
from typing import Tuple, Any, Dict


class Forecasting:
    def __init__(self, args):
        self.args = args

    def __call__(self, transformed_data):
        if self.args.data_representation == "bpe_symbolic":
            return self.bpe_symbolic(transformed_data)
        elif self.args.data_representation == "signal":
            return self.signal(transformed_data)

    def signal(self, transformed_data: Dict[str, Any]):
        inputs = np.asarray(transformed_data["transformed_data"])
        padding_mask = transformed_data.get("padding_mask")
        modality_mask = transformed_data.get("modality_mask")
        if self.args.objective == "autoregressive":
            return {
                "signal": inputs,
                "padding_mask": padding_mask,
                "modality_mask": modality_mask,
            }

    def bpe_symbolic(self, transformed_data: Dict[str, Any]):
        inputs = np.asarray(transformed_data["transformed_data"])
        orig_len = inputs.shape[0]
        labels = inputs.copy()

        if "train" in self.args.mode and self.args.objective == "autoregressive":
            labels = self.autoregressive(labels)
        else:
            labels, inputs = self.prepare_eval(inputs, labels)

        inputs = torch.as_tensor(inputs, dtype=torch.long)
        labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else None

        if self.args.neural_network in {"trans_discrete_decoder", "trans_discrete_decoder_fm"}:
            out = {"tgt_ids": inputs, "labels": labels}
            if self.args.neural_network == "trans_discrete_decoder_fm":
                signal_values = transformed_data["normalized_signal"]
                signal_mask = (labels != -100).float().cpu().numpy()
                signal_values = signal_values[:signal_mask.shape[0]]
                if signal_values.shape[0] < signal_mask.shape[0]:
                    pad = signal_mask.shape[0] - signal_values.shape[0]
                    signal_values = np.pad(signal_values, (0, pad))
                out["signal_values"] = torch.as_tensor(signal_values, dtype=torch.float32)
                out["signal_mask"] = torch.as_tensor(signal_mask, dtype=torch.float32)
            if "eval" in self.args.mode:
                out.update({"min": transformed_data["min"], "max": transformed_data["max"],
                            "report": transformed_data["report"], "orig_len" : orig_len})
            return out

    def autoregressive(self, labels: np.ndarray) -> np.ndarray:
        non_pad_mask = labels != self.args.pad_id
        if non_pad_mask.any():
            first_non_pad = non_pad_mask.argmax()
            non_pad_len = non_pad_mask.sum()
            num_to_mask = int(non_pad_len * (1 - self.args.forecast_ratio))
            labels[first_non_pad : first_non_pad + num_to_mask] = -100
        labels[labels == self.args.pad_id] = -100
        labels[labels == self.args.bos_id] = -100
        return labels

    def prepare_eval(self, inputs: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        split_idx = int(len(labels) * (1 - self.args.forecast_ratio))
        return labels[split_idx:], inputs[:split_idx]