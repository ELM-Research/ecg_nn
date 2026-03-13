import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest
import tempfile
from argparse import Namespace

from neural_networks.transformer.discrete.decoder import DecoderTransformerConfig, DecoderTransformer
from neural_networks.transformer.discrete.signal_head import SignalFlowHeadConfig, SignalFlowHead, DecoderWithSignalHead
from neural_networks.build_nn import BuildNN


def _make_args(**overrides):
    defaults = dict(
        neural_network="trans_discrete_decoder", nn_ckpt=None, bpe_symbolic_len=64,
        pad_id=0, bos_id=1, eos_id=2, signal_head=False, bfloat_16=False,
        signal_head_layers=2, signal_head_num_steps=10, freeze_decoder=False,
        flow_loss_weight=1.0, condition=None, segment_len=100,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


class _FakeDataRepr:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size


def _save_signal_head_checkpoint(path, vocab_size=50):
    dec_cfg = DecoderTransformerConfig(vocab_size=vocab_size, pad_id=0, d_model=512, n_heads=8, dim_ff=2048, num_layers=12, max_seq_len=64)
    decoder = DecoderTransformer(dec_cfg)
    sh_cfg = SignalFlowHeadConfig(signal_dim=12, d_model=512, n_heads=8, dim_ff=2048, num_layers=2, max_signal_len=100, num_steps=10)
    model = DecoderWithSignalHead(decoder, SignalFlowHead(sh_cfg))
    torch.save({"model_state_dict": model.state_dict()}, path)


def _save_plain_decoder_checkpoint(path, vocab_size=50):
    dec_cfg = DecoderTransformerConfig(vocab_size=vocab_size, pad_id=0, d_model=512, n_heads=8, dim_ff=2048, num_layers=12, max_seq_len=64)
    decoder = DecoderTransformer(dec_cfg)
    torch.save({"model_state_dict": decoder.state_dict()}, path)


def test_prepare_transformer_loads_vocab_from_signal_head_checkpoint():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        _save_signal_head_checkpoint(f.name, vocab_size=50)
        args = _make_args(nn_ckpt=f.name, signal_head=True)
        builder = BuildNN(args)
        data_repr = _FakeDataRepr(vocab_size=50)
        nn_components = builder.build_nn(data_repr)
        assert nn_components["neural_network"].cfg.vocab_size == 50
    os.unlink(f.name)


def test_prepare_transformer_loads_vocab_from_plain_decoder_checkpoint():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        _save_plain_decoder_checkpoint(f.name, vocab_size=50)
        args = _make_args(nn_ckpt=f.name, signal_head=True)
        builder = BuildNN(args)
        data_repr = _FakeDataRepr(vocab_size=50)
        nn_components = builder.build_nn(data_repr)
        assert nn_components["neural_network"].decoder.cfg.vocab_size == 50
    os.unlink(f.name)


def test_prepare_transformer_loads_plain_decoder_checkpoint_no_signal_head():
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        _save_plain_decoder_checkpoint(f.name, vocab_size=50)
        args = _make_args(nn_ckpt=f.name, signal_head=False)
        builder = BuildNN(args)
        data_repr = _FakeDataRepr(vocab_size=50)
        nn_components = builder.build_nn(data_repr)
        assert nn_components["neural_network"].cfg.vocab_size == 50
    os.unlink(f.name)
