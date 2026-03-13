import torch

from neural_networks.transformer.discrete.decoder import DecoderTransformer
from neural_networks.transformer.discrete.decoder import DecoderTransformerConfig


def test_decoder_flow_matching_forward_combines_losses():
    torch.manual_seed(0)
    cfg = DecoderTransformerConfig(
        vocab_size=32,
        pad_id=0,
        d_model=32,
        n_heads=4,
        dim_ff=64,
        num_layers=2,
        max_seq_len=16,
        flow_matching_head=True,
        flow_matching_loss_weight=0.5,
    )
    model = DecoderTransformer(cfg)

    tgt_ids = torch.tensor([[0, 0, 1, 4, 5, 6, 2, 0]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, -100, 4, 5, 6, 2, -100]], dtype=torch.long)
    signal_values = torch.randn(1, tgt_ids.size(1))
    signal_mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]], dtype=torch.float32)

    out = model(
        tgt_ids=tgt_ids,
        labels=labels,
        signal_values=signal_values,
        signal_mask=signal_mask,
    )

    assert out.loss is not None
    assert out.token_loss is not None
    assert out.flow_loss is not None
    expected = out.token_loss + cfg.flow_matching_loss_weight * out.flow_loss
    assert torch.allclose(out.loss, expected)
    assert out.flow_prediction.shape == signal_values.shape


def test_decoder_flow_matching_signal_generation_shape():
    torch.manual_seed(0)
    cfg = DecoderTransformerConfig(
        vocab_size=32,
        pad_id=0,
        d_model=32,
        n_heads=4,
        dim_ff=64,
        num_layers=2,
        max_seq_len=16,
        flow_matching_head=True,
    )
    model = DecoderTransformer(cfg)
    tgt_ids = torch.tensor([[0, 1, 4, 5, 6, 2]], dtype=torch.long)

    signal = model.generate_signal(tgt_ids=tgt_ids, signal_len=12, num_steps=4)
    assert signal.shape == (1, 12)


def test_decoder_without_flow_head_keeps_original_path():
    cfg = DecoderTransformerConfig(
        vocab_size=32,
        pad_id=0,
        d_model=32,
        n_heads=4,
        dim_ff=64,
        num_layers=2,
        max_seq_len=16,
    )
    model = DecoderTransformer(cfg)
    tgt_ids = torch.tensor([[0, 1, 4, 5, 6, 2]], dtype=torch.long)
    labels = torch.tensor([[-100, -100, 4, 5, 6, 2]], dtype=torch.long)

    out = model(tgt_ids=tgt_ids, labels=labels)
    assert out.loss is not None
    assert out.flow_loss is None
