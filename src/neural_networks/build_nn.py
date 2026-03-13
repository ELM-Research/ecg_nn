import argparse
import torch
import numpy as np

from utils.gpu_setup import is_main

from configs.constants import TRANSFORMER_MODELS, MAE_MODELS, MERL_MODEL, MLAE_MODELS, MTAE_MODELS, ST_MEM_MODELS


class BuildNN:
    def __init__(self, args: argparse.Namespace):
        self.args = args

    def build_nn(self, data_representation):
        nn_components = None

        if "trans" in self.args.neural_network:
            nn_components = self.prepare_transformer(data_representation)
            nn_components["find_unused_parameters"] = TRANSFORMER_MODELS[self.args.neural_network]["find_unused_parameters"]
        if "mae" in self.args.neural_network:
            nn_components = self.prepare_mae()
            nn_components["find_unused_parameters"] = MAE_MODELS[self.args.neural_network]["find_unused_parameters"]
        if "merl" in self.args.neural_network:
            nn_components = self.prepare_merl()
            nn_components["find_unused_parameters"] = MERL_MODEL[self.args.neural_network]["find_unused_parameters"]
        if self.args.neural_network == "mlae":
            nn_components = self.prepare_mlae()
            nn_components["find_unused_parameters"] = MLAE_MODELS[self.args.neural_network]["find_unused_parameters"]
        if self.args.neural_network == "mtae":
            nn_components = self.prepare_mtae()
            nn_components["find_unused_parameters"] = MTAE_MODELS[self.args.neural_network]["find_unused_parameters"]
        if self.args.neural_network == "st_mem":
            nn_components = self.prepare_st_mem()
            nn_components["find_unused_parameters"] = ST_MEM_MODELS[self.args.neural_network]["find_unused_parameters"]
        assert nn_components is not None, print("NN Components is None")
        if self.args.nn_ckpt:
            self.load_nn_checkpoint(nn_components, data_representation)
        return nn_components

    def prepare_transformer(self, data_representation):
        if "trans_discrete" in self.args.neural_network:
            vocab_size = data_representation.vocab_size
            if self.args.nn_ckpt:
                ckpt = torch.load(self.args.nn_ckpt, map_location="cpu", weights_only=False)
                vocab_size = ckpt["model_state_dict"]["token_emb.weight"].shape[0]
            if self.args.neural_network in {"trans_discrete_decoder", "trans_discrete_decoder_fm"}:
                from neural_networks.transformer.discrete.decoder import DecoderTransformerConfig, DecoderTransformer
                cfg = DecoderTransformerConfig(
                    vocab_size=vocab_size,
                    pad_id=self.args.pad_id,
                    max_seq_len=self.args.bpe_symbolic_len,
                    flow_matching_head=self.args.neural_network == "trans_discrete_decoder_fm",
                    flow_matching_loss_weight=self.args.flow_matching_loss_weight,
                )
                model = DecoderTransformer(cfg)
                if self.args.bfloat_16:
                    model = model.to(torch.bfloat16) # decoder only usually bfloat16
        elif "trans_continuous" in self.args.neural_network:
            if self.args.neural_network == "trans_continuous_nepa":
                from neural_networks.transformer.continuous.nepa import NEPAConfig, NEPATransformer
                cfg = NEPAConfig(max_seq_len=self.args.segment_len)
                model = NEPATransformer(cfg)
            elif self.args.neural_network == "trans_continuous_dit":
                from neural_networks.transformer.continuous.dit import DiTConfig, DiT
                num_steps = {"rectified_flow": 50, "ddpm": 1000}[self.args.objective]
                if self.args.condition == "lead":
                    input_dim = 7
                else:
                    input_dim = 12
                cfg = DiTConfig(input_dim = input_dim, loss_type=self.args.objective, num_steps=num_steps,
                                condition = self.args.condition, text_feature_extractor=self.args.text_feature_extractor)
                model = DiT(cfg)
        return {"neural_network": model}

    def prepare_mae(self, ):
        if self.args.neural_network == "mae_vit":
            from neural_networks.mae.mae_vit import MAEViTConfig, MAEViT

            cfg = MAEViTConfig(patch_dim=self.args.patch_dim)
            model = MAEViT(cfg)
        return {"neural_network": model}
    
    def prepare_merl(self,):
        from neural_networks.merl.merl import MerlConfig, Merl
        cfg = MerlConfig(distributed=self.args.distributed)
        model = Merl(cfg)
        return {"neural_network": model}

    def prepare_mlae(self):
        from neural_networks.mlae.mlae import MLAEConfig, MLAE
        cfg = MLAEConfig(seq_len=self.args.segment_len)
        model = MLAE(cfg)
        return {"neural_network": model}

    def prepare_mtae(self):
        from neural_networks.mtae.mtae import MTAEConfig, MTAE
        cfg = MTAEConfig(seq_len=self.args.segment_len, patch_size=self.calculate_patch_size())
        model = MTAE(cfg)
        return {"neural_network": model}

    def prepare_st_mem(self):
        from neural_networks.st_mem.st_mem import ST_MEMConfig, ST_MEM
        cfg = ST_MEMConfig(seq_len=self.args.segment_len, patch_size=self.calculate_patch_size())
        model = ST_MEM(cfg)
        return {"neural_network": model}

    def load_nn_checkpoint(self, nn_components, data_representation):
        ckpt = torch.load(self.args.nn_ckpt, map_location="cpu", weights_only=False)
        use_ema = getattr(self.args, "ema", False) and "ema_state_dict" in ckpt
        if use_ema:
            state = ckpt["ema_state_dict"]
            if is_main():
                print("Loading EMA weights from checkpoint")
        else:
            state = ckpt["model_state_dict"]

        if "trans_discrete" in self.args.neural_network:
            old_vocab = state["token_emb.weight"].shape[0]
            new_vocab = data_representation.vocab_size
            strict = self.args.neural_network != "trans_discrete_decoder_fm"
            nn_components["neural_network"].load_state_dict(state, strict=strict)
            if new_vocab > old_vocab:
                nn_components["neural_network"].resize_embeddings(new_vocab)
                if is_main():
                    print(f"Resized vocab from {old_vocab} to {new_vocab}")
        else:
            nn_components["neural_network"].load_state_dict(state, strict=False)
        if is_main():
            print(f"Loaded NN checkpoint from {self.args.nn_ckpt}")

    def calculate_patch_size(self):
        min_patches = 16
        max_patches = 64
        factors = [i for i in range(1, self.args.segment_len + 1) if self.args.segment_len % i == 0]
        patch_candidates = []
        for patch_size in factors:
            num_patches = self.args.segment_len // patch_size
            if min_patches <= num_patches <= max_patches:
                patch_candidates.append(patch_size)
        if not patch_candidates:
            target = int(np.sqrt(self.args.segment_len / 32))
            patch_size = min(factors, key=lambda x: abs(x - target))
        else:
            patch_size = min(patch_candidates, key=lambda x: abs(self.args.segment_len // x - 32))
        return patch_size
