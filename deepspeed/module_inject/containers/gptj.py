# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from .base import *
from .features.meta_tensor import MetaTensorContainer
from deepspeed.model_implementations.transformers.ds_gpt import DeepSpeedGPTInference
import torch
from torch.nn.parameter import Parameter
from ..policy import TransformerPolicy
from ..policy import transformer_param_names
from ..policy import maybe_copy
from ..policy import maybe_copy_qkv
from ..policy import maybe_get_lora


class DS_GPTJContainer(MetaTensorContainer, BaseTransformerContainer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # All model specific things should be defined here instead of the base class.

    def create_module(self, config=None):
        _config = config if config is not None else self.ds_model_config
        self.module = DeepSpeedGPTInference(_config, mp_group=self.mp_group)
        self.module.config.scale_attention = self.scale_attention
        return self.module

    def load_params(self, module, sd, weight_quantizer, mp_replace, prefix):
        param_names = (
            'attn.q_proj.weight', \
            'attn.k_proj.weight', \
            'attn.v_proj.weight', \
            'attn.out_proj.weight', \
            'mlp.fc_in.weight', \
            'mlp.fc_in.bias', \
            'mlp.fc_out.weight', \
            'mlp.fc_out.bias', \
            'ln_1.weight', \
            'ln_1.bias'
        )
        maybe_copy_qkv(module.attention,
                       sd,
                       weight_quantizer,
                       mp_replace,
                       'attn_qkvw', [prefix + param_names[0], prefix + param_names[1], prefix + param_names[2]],
                       split_qkv=self.policy.split_qkv)
        for i in range(3, 4):
            maybe_copy(module.attention, sd, weight_quantizer, mp_replace, transformer_param_names[i - 1],
                       prefix + param_names[i])
        for i in range(4, 8):
            maybe_copy(module.mlp, sd, weight_quantizer, mp_replace, transformer_param_names[i],
                       prefix + param_names[i])
        for i in range(8, 10):
            maybe_copy(module, sd, weight_quantizer, mp_replace, transformer_param_names[i + 2],
                       prefix + param_names[i])

    def attention_q_k_v_mp(self, mp_replace, reversed_dim=False):
        self.module.attention.attn_qw = mp_replace.copy(self.module.attention.attn_qw[:self.qw.shape[0] //
                                                                                      mp_replace.mp_size],
                                                        self.qw,
                                                        int8=reversed_dim,
                                                        allocat_tensor=reversed_dim)
        self.module.attention.attn_kw = mp_replace.copy(self.module.attention.attn_kw[:self.qw.shape[0] //
                                                                                      mp_replace.mp_size],
                                                        self.kw,
                                                        int8=reversed_dim,
                                                        allocat_tensor=reversed_dim)
        self.module.attention.attn_vw = mp_replace.copy(self.module.attention.attn_vw[:self.qw.shape[0] //
                                                                                      mp_replace.mp_size],
                                                        self.vw,
                                                        int8=reversed_dim,
                                                        allocat_tensor=reversed_dim)
        self.module.attention.attn_qb = mp_replace.copy(
            self.module.attention.attn_qb[:self.qw.shape[0] // mp_replace.mp_size],
            self.qb,
            int8=reversed_dim,
            allocat_tensor=reversed_dim) if self.module.attention.attn_qb is not None else None
        self.module.attention.attn_kb = mp_replace.copy(
            self.module.attention.attn_kb[:self.qw.shape[0] // mp_replace.mp_size],
            self.kb,
            int8=reversed_dim,
            allocat_tensor=reversed_dim) if self.module.attention.attn_kb is not None else None
        self.module.attention.attn_vb = mp_replace.copy(
            self.module.attention.attn_vb[:self.qw.shape[0] // mp_replace.mp_size],
            self.vb,
            int8=reversed_dim,
            allocat_tensor=reversed_dim) if self.module.attention.attn_vb is not None else None

    def mlp_inter_mp(self, mp_replace, reversed_dim=False):
        if reversed_dim:
            self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w[:self._h4h_w.shape[0] //
                                                                              mp_replace.mp_size],
                                                      self._h4h_w,
                                                      int8=reversed_dim,
                                                      allocat_tensor=reversed_dim)
            self.module.mlp.inter_b = mp_replace.copy(
                    self.module.mlp.inter_b[:self._h4h_w.shape[0] // mp_replace.mp_size],
                    self._h4h_b,
                    int8=reversed_dim,
                    allocat_tensor=reversed_dim) if self.module.mlp.inter_b is not None else None
        else:
            self.module.mlp.inter_w = mp_replace.copy(self.module.mlp.inter_w, self._h4h_w, int8=reversed_dim)
            self.module.mlp.inter_b = mp_replace.copy(self.module.mlp.inter_b, self._h4h_b, int8=reversed_dim)

    def release_qkv(self):
        del self.module.attention.attn_qkvw
        del self.module.attention.attn_qkvb
        self.module.attention.attn_qkvw = None
        self.module.attention.attn_qkvb = None

        qkv_data = [self.module.attention.attn_qw.data, \
                self.module.attention.attn_qb.data if self.module.attention.attn_qb is not None else None, \
                self.module.attention.attn_kw.data, \
                self.module.attention.attn_kb.data if self.module.attention.attn_kb is not None else None, \
                self.module.attention.attn_vw.data, \
                self.module.attention.attn_vb.data if self.module.attention.attn_vb is not None else None]
        for data in qkv_data:
            del data

        self.module.attention.attn_qw = self.qw
        self.module.attention.attn_qb = self.qb
        self.module.attention.attn_kw = self.kw
        self.module.attention.attn_kb = self.kb
        self.module.attention.attn_vw = self.vw
        self.module.attention.attn_vb = self.vb

    def reset_qkv(self):
        self.qkvw.data[:self.qw.shape[0]] = self.qw.data
        self.qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kw.data
        self.qkvw.data[2 * self.qw.shape[0]:] = self.vw.data
        if self.qkvb is not None:
            self.qkvb.data[:self.qw.shape[0]] = self.qb.data
            self.qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]] = self.kb.data
            self.qkvb.data[2 * self.qw.shape[0]:] = self.vb.data

        qkv_data = [self.qw.data, \
                    self.qb.data if self.qb is not None else None, \
                    self.kw.data, \
                    self.kb.data if self.kb is not None else None, \
                    self.vw.data, \
                    self.vb.data if self.vb is not None else None]

        self.qw.data = self.qkvw.data[:self.qw.shape[0]]
        self.kw.data = self.qkvw.data[self.qw.shape[0]:2 * self.qw.shape[0]]
        self.vw.data = self.qkvw.data[2 * self.qw.shape[0]:]

        if self.qkvb is not None:
            self.qb.data = self.qkvb.data[:self.qw.shape[0]]
            self.kb.data = self.qkvb.data[self.qw.shape[0]:2 * self.qw.shape[0]]
            self.vb.data = self.qkvb.data[2 * self.qw.shape[0]:]

        for data in qkv_data:
            del data

    def set_params_wo_copy(self, Z3_enabled=False):
        self.module.mlp.attn_nw = self.attn_nw
        self.module.mlp.attn_nb = self.attn_nb
        self.module.norm_w = self.input_nw
        self.module.norm_b = self.input_nb
        self.module.mlp.inter_w = self._h4h_w
        self.module.mlp.inter_b = self._h4h_b
        self.module.mlp.output_w = self._4hh_w
        self.module.mlp.output_b = self._4hh_b
        self.module.attention.attn_ow = self.dense_w
        self.module.attention.attn_ob = self.dense_b
        if not Z3_enabled or self.q_k_v is None:
            self.module.attention.attn_qkvw = self.qkvw
            self.module.attention.attn_qkvb = self.qkvb
        if self.q_k_v is not None:
            if Z3_enabled:
                self.module.attention.attn_qw = self.qw
                self.module.attention.attn_qb = self.qb
                self.module.attention.attn_kw = self.kw
                self.module.attention.attn_kb = self.kb
                self.module.attention.attn_vw = self.vw
                self.module.attention.attn_vb = self.vb
            else:
                self.qw.data = self.qkvw[:self.qw.shape[0], :]
                self.kw.data = self.qkvw[self.qw.shape[0]:2 * self.qw.shape[0], :]
                self.vw.data = self.qkvw[self.qw.shape[0] * 2:, :]
                if self.qkvb is not None:
                    self.qb.data = self.qkvb[:self.qw.shape[0]]
                    self.kb.data = self.qkvb[self.qw.shape[0]:2 * self.qw.shape[0]]
                    self.vb.data = self.qkvb[self.qw.shape[0] * 2:]


class HFGPTJLayerPolicy(TransformerPolicy):
    _orig_layer_class = None

    def __init__(self, client_module, inference=True):
        super().__init__(inference, scale_attention=True)
        self.client_module = client_module
        try:
            import transformers
            HFGPTJLayerPolicy._orig_layer_class = transformers.models.gptj.modeling_gptj.GPTJBlock
        except:
            HFGPTJLayerPolicy._orig_layer_class = None

    def get_hidden_heads(self):
        return self.client_module.attn.embed_dim, \
                self.client_module.attn.num_attention_heads, \
                self.client_module.ln_1.eps

    def get_q_k_v(self):
        return self.client_module.attn.q_proj.weight, \
            None, \
            self.client_module.attn.k_proj.weight, \
            None, \
            self.client_module.attn.v_proj.weight, \
            None

    def attention(self, enable_training=False):
        qw = self.client_module.attn.q_proj.weight
        kw = self.client_module.attn.k_proj.weight
        vw = self.client_module.attn.v_proj.weight

        qkvw = Parameter(torch.cat((qw, kw, vw), dim=0), requires_grad=enable_training)

        return qkvw, \
               None, \
               self.client_module.attn.out_proj.weight, \
               None,

    def mlp(self):
        return self.client_module.mlp.fc_in.weight, \
               self.client_module.mlp.fc_in.bias, \
               self.client_module.mlp.fc_out.weight, \
               self.client_module.mlp.fc_out.bias

    def layernorm(self):
        return None, \
               None, \
               self.client_module.ln_1.weight, \
               self.client_module.ln_1.bias

    def get_lora_params(self):
        all_lora_params = []
        for p in [
            self.client_module.mlp.fc_in, \
            self.client_module.mlp.fc_out, \
            self.client_module.attn.q_proj, \
            self.client_module.attn.k_proj, \
            self.client_module.attn.v_proj, \
            self.client_module.attn.out_proj, \
            ]:
            all_lora_params.append(maybe_get_lora(p))
        return all_lora_params
