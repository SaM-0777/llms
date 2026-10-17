import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen38.config import Config
from qwen38.utils import is_torchdynamo_exporting
from qwen38.norms import RMSNorm, RMSNormGated, l2_norm
from qwen38.rope import apply_rotary_embeddings


class FullAttention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.use_cache = cfg.use_cache

        self.head_dim = cfg.head_dim
        self.num_attention_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_kv_heads

        self.W_q = nn.Linear(
            cfg.embedding_dim,
            2 * self.num_attention_heads * self.head_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.W_k = nn.Linear(
            cfg.embedding_dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.W_v = nn.Linear(
            cfg.embedding_dim,
            self.num_kv_heads * self.head_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.W_o = nn.Linear(
            self.num_attention_heads * self.head_dim,
            cfg.embedding_dim,
            bias=False,
            dtype=cfg.dtype,
        )

        self.q_norm = RMSNorm(dim=self.head_dim, eps=cfg.norm_eps)
        self.k_norm = RMSNorm(dim=self.head_dim, eps=cfg.norm_eps)

        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)

        if self.use_cache:
            self.cache_k = torch.zeros(
                cfg.batch_size,
                self.num_kv_heads,
                cfg.max_seqlen,
                self.head_dim,
                device=cfg.device,
                dtype=cfg.dtype,
            )
            self.cache_v = torch.zeros(
                cfg.batch_size,
                self.num_kv_heads,
                cfg.max_seqlen,
                self.head_dim,
                device=cfg.device,
                dtype=cfg.dtype,
            )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: tuple[torch.Tensor, torch.Tensor],
        start_pos: int = 0,
        mask: torch.Tensor | None = None,
    ):
        B, seqlen, _ = x.shape
        hidden_shape = (B, seqlen, -1, self.head_dim)

        q = self.W_q(x)  # B, seqlen, q_proj_dim
        k = self.W_k(x)  # B, seqlen, kv_proj_dim
        v = self.W_v(x)  # B, seqlen, kv_proj_dim

        query_states, gate = torch.chunk(
            q.view(B, seqlen, -1, 2 * self.head_dim), 2, dim=-1
        )  # (B, seqlen, num_attention_heads, head_dim), (B, seqlen, num_attention_heads, head_dim)
        gate = gate.reshape(B, seqlen, -1)  # B, seqlen, num_attention_heads * head_dim

        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(
            1, 2
        )  # B, num_attention_heads, seqlen, head_dim
        key_states = self.k_norm(k.view(hidden_shape)).transpose(
            1, 2
        )  # B, num_kv_heads, seqlen, head_dim
        value_states = v.view(hidden_shape).transpose(
            1, 2
        )  # B, num_kv_heads, seqlen, head_dim

        query_states, key_states = apply_rotary_embeddings(
            q=query_states,
            k=key_states,
            freqs=freqs_cis,
        )

        if self.use_cache:
            self.cache_k[:B, :, start_pos : start_pos + seqlen, :] = key_states  # type: ignore
            self.cache_v[:B, :, start_pos : start_pos + seqlen, :] = value_states  # type: ignore

            key_states = self.cache_k[:B, :, : start_pos + seqlen, :]  # type: ignore
            value_states = self.cache_v[:B, :, : start_pos + seqlen, :]  # type: ignore

        output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            mask,
            dropout_p=0.0,
            scale=self.head_dim**-0.5,
            enable_gqa=True,
        )  # B, num_attention_heads, seqlen, head_dim

        output = output.transpose(1, 2)  # B, seqlen, num_attention_heads, head_dim
        gate = gate.view(
            B, seqlen, self.num_attention_heads, self.head_dim
        )  # B, seqlen, num_attention_heads, head_dim

        output = output * torch.sigmoid(gate)
        output = output.reshape(
            B, seqlen, self.num_attention_heads * self.head_dim
        )  # B, seqlen, num_attention_heads * head_dim
        output = self.W_o(output)  # B, seqlen, embedding_dim

        return output


class GatedDeltaNetAttention(nn.Module):
    """
    Implementation from
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py
    """

    def __init__(self, cfg: Config, layer_idx: int) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.use_cache = cfg.use_cache

        self.num_k_heads = cfg.linear_num_key_heads
        self.num_v_heads = cfg.linear_num_value_heads
        self.k_head_dim = cfg.linear_key_head_dim
        self.v_head_dim = cfg.linear_value_head_dim

        self.k_dim = self.k_head_dim * self.num_k_heads
        self.v_dim = self.v_head_dim * self.num_v_heads

        self.conv_kernel_size: int = cfg.linear_conv_kernel_dim
        self.conv_dim = 2 * self.k_dim + self.v_dim

        self.Conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))

        A = torch.empty(self.num_v_heads).uniform_(0.01, 16)
        self.A_log = nn.Parameter(torch.log(A))

        self.norm = RMSNormGated(dim=self.v_head_dim, eps=cfg.norm_eps)

        self.out_proj = nn.Linear(
            self.v_dim,
            cfg.embedding_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.in_proj_qkv = nn.Linear(
            cfg.embedding_dim,
            (2 * self.k_dim) + self.v_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.in_proj_z = nn.Linear(
            cfg.embedding_dim,
            self.v_dim,
            bias=False,
            dtype=cfg.dtype,
        )
        self.in_proj_b = nn.Linear(
            cfg.embedding_dim,
            self.num_v_heads,
            bias=False,
            dtype=cfg.dtype,
        )
        self.in_proj_a = nn.Linear(
            cfg.embedding_dim,
            self.num_v_heads,
            bias=False,
            dtype=cfg.dtype,
        )

        self.register_buffer("recurrent_state", None, persistent=False)
        self.register_buffer("conv_state", None, persistent=False)

        self.recurrent_state = torch.zeros(
            cfg.batch_size,
            self.num_v_heads,
            self.k_head_dim,
            self.v_head_dim,
            device=cfg.device,
            dtype=torch.float32,
        )
        self.conv_state = torch.zeros(
            cfg.batch_size,
            self.conv_dim,
            self.conv_kernel_size - 1,
            device=cfg.device,
            dtype=torch.float32,
        )

    def _apply_zero_padding_mask(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        if mask is None:
            return x

        # NOTE mask is boolean or binary Tensor of shape B, seqlen
        # with values must be
        # True / 1, for valid tokens
        # False / 0, for padding tokens
        return x.masked_fill(
            ~mask.to(device=x.device, dtype=torch.bool).unsqueeze(-1), 0
        )

    def _apply_causal_convolution(self, mixed_qkv: torch.Tensor):
        _, _, seqlen = mixed_qkv.shape
        state_len = self.conv_state.shape[-1]

        weights = self.Conv1d.weight
        bias = self.Conv1d.bias

        # Previous conv state is available and during decoding stage
        if self.use_cache and seqlen == 1:
            conv_input = torch.cat([self.conv_state, mixed_qkv], dim=-1).to(
                dtype=weights.dtype
            )
            self.conv_state.copy_(conv_input[:, :, -state_len:])

            out = F.conv1d(
                conv_input,
                weight=weights,
                bias=bias,
                padding=0,
                groups=self.conv_dim,
            )
            out = out[:, :, -seqlen:]
        # Prefill stage
        else:
            if self.use_cache:
                conv_input = torch.cat([self.conv_state, mixed_qkv], dim=-1).to(
                    dtype=weights.dtype
                )
                self.conv_state.copy_(conv_input[:, :, -state_len:])
            else:
                conv_input = mixed_qkv
                conv_input = F.pad(
                    conv_input.to(dtype=weights.dtype), (self.conv_kernel_size - 1, 0)
                )

            out = F.conv1d(
                conv_input,
                weight=weights,
                bias=bias,
                padding=0,
                groups=self.conv_dim,
            )
            out = out[:, :, -seqlen:]

        return F.silu(out).to(dtype=mixed_qkv.dtype)

    def _apply_recurrent_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
    ):
        dtype = q.dtype

        B, seqlen, _, k_head_dim = k.shape
        _, _, num_v_heads, v_head_dim = v.shape
        decay = g

        q = q.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen, k_head_dim
        k = k.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen, k_head_dim
        v = v.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen, v_head_dim
        beta = beta.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen
        decay = decay.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen

        q = l2_norm(q)
        k = l2_norm(k)

        q = q / (q.shape[-1] ** 0.5)

        last_recurrent_state = self.recurrent_state.to(
            v
        )  # B, num_v_heads, k_head_dim, v_head_dim
        core_attn_out = torch.zeros_like(v)  # B, num_v_heads, seqlen, v_head_dim

        for i in range(seqlen):
            q_t, k_t, v_t = (
                q[:, :, i],
                k[:, :, i],
                v[:, :, i],
            )  # (B, num_v_heads, k_head_dim), (B, num_v_heads, k_head_dim), (B, num_v_heads, v_head_dim)

            decay_t = decay[:, :, i].exp()[..., None, None]  # B, num_v_heads, 1, 1
            last_recurrent_state = last_recurrent_state * decay_t

            beta_t = beta[:, :, i].unsqueeze(-1)
            kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            last_recurrent_state = last_recurrent_state + k_t.unsqueeze(
                -1
            ) * delta.unsqueeze(-2)

            core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(
                dim=-2
            )

        core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(dtype=dtype)
        return core_attn_out, last_recurrent_state

    def _apply_chunk_gated_delta_rule(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 64,
    ):
        dtype = q.dtype

        B, seqlen, _, k_head_dim = k.shape
        _, _, num_v_heads, v_head_dim = v.shape
        decay = g

        q = q.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen, k_head_dim
        k = k.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen, k_head_dim
        v = v.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen, v_head_dim
        beta = beta.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen
        decay = decay.transpose(1, 2).float().contiguous()  # B, num_v_heads, seqlen

        q = l2_norm(q)
        k = l2_norm(k)

        q = q / (q.shape[-1] ** 0.5)

        # NOTE
        # Pad sequence length to be a multiple of chunk_size. Padding is described as (left_pad, right_pad) for each dim.
        # Check https://github.com/huggingface/transformers/blob/09281b8d56b08199ce9f397c694f38853cc45278/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L336
        pad_size = (chunk_size - seqlen % chunk_size) % chunk_size
        q = F.pad(q, (0, 0, 0, pad_size))
        k = F.pad(k, (0, 0, 0, pad_size))
        v = F.pad(v, (0, 0, 0, pad_size))
        beta = F.pad(beta, (0, pad_size))
        decay = F.pad(decay, (0, pad_size))

        total_seqlen = seqlen + pad_size
        num_chunks = total_seqlen // chunk_size

        v_beta = v * beta.unsqueeze(-1)
        k_beta = k * beta.unsqueeze(-1)

        q = q.reshape(q.shape[0], q.shape[1], -1, chunk_size, q.shape[-1])
        k = k.reshape(k.shape[0], k.shape[1], -1, chunk_size, k.shape[-1])
        k_beta = k_beta.reshape(
            k_beta.shape[0], k_beta.shape[1], -1, chunk_size, k_beta.shape[-1]
        )
        v_beta = v_beta.reshape(
            v_beta.shape[0], v_beta.shape[1], -1, chunk_size, v_beta.shape[-1]
        )
        decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)

        strictly_upper_mask = torch.ones(
            chunk_size, chunk_size, dtype=torch.bool, device=q.device
        ).triu(1)

        cum_decay = decay.cumsum(dim=3)

        pairwise_decay = cum_decay.unsqueeze(4) - cum_decay.unsqueeze(3)
        pairwise_decay = pairwise_decay.masked_fill(strictly_upper_mask, float("-inf"))
        pairwise_decay = pairwise_decay.exp()

        ut_system = (k_beta @ k.transpose(-1, -2)) * pairwise_decay
        intra_chunk_attn = (q @ k.transpose(-1, -2)) * pairwise_decay
        decayed_k_beta = k_beta * cum_decay.exp().unsqueeze(-1)

        if not is_torchdynamo_exporting():
            new_values = torch.linalg.solve_triangular(
                ut_system, v_beta, upper=False, unitriangular=True
            )
            k_cumdecay = torch.linalg.solve_triangular(
                ut_system, decayed_k_beta, upper=False, unitriangular=True
            )
        else:
            ut_system = -ut_system.tril(-1)
            for i in range(1, chunk_size):
                row = ut_system[..., i, :i].clone()
                sub = ut_system[..., :i, :i].clone()
                ut_system[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
            ut_system = ut_system + torch.eye(
                chunk_size, dtype=ut_system.dtype, device=ut_system.device
            )
            new_values, k_cumdecay = ut_system @ v_beta, ut_system @ decayed_k_beta

        last_recurrent_state = self.recurrent_state.to(new_values)
        core_attn_out = torch.zeros_like(new_values)

        q = q * cum_decay.exp().unsqueeze(-1)
        k = k * (cum_decay[..., -1:] - cum_decay).exp().unsqueeze(-1)
        chunk_decay = cum_decay[..., -1].exp()[..., None, None]

        for i in range(num_chunks):
            v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ last_recurrent_state
            inter_chunk_attn = q[:, :, i] @ last_recurrent_state
            core_attn_out[:, :, i] = (
                inter_chunk_attn + intra_chunk_attn[:, :, i] @ v_new
            )
            last_recurrent_state = (
                last_recurrent_state * chunk_decay[:, :, i]
                + k[:, :, i].transpose(-1, -2) @ v_new
            )

        padded_output_shape = (B, num_v_heads, -1, v_head_dim)
        core_attn_out = core_attn_out.reshape(padded_output_shape)
        core_attn_out = core_attn_out[:, :, :seqlen]
        core_attn_out = core_attn_out.transpose(1, 2).to(
            dtype=dtype, memory_format=torch.contiguous_format
        )
        return core_attn_out, last_recurrent_state

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ):
        x = self._apply_zero_padding_mask(x, mask)
        B, seqlen, _ = x.shape

        mixed_qkv = self.in_proj_qkv(x)  # B, seqlen, 2 * k_dim + v_dim
        mixed_qkv = mixed_qkv.transpose(1, 2)  # B, 2 * k_dim + v_dim, seqlen

        z = self.in_proj_z(x)
        z = z.reshape(
            B, seqlen, -1, self.v_head_dim
        )  # B, seqlen, v_dim, v_dim * num_v_heads

        a = self.in_proj_a(x)  # B, seqlen, num_v_heads
        b = self.in_proj_b(x)  # B, seqlen, num_v_heads

        mixed_qkv = self._apply_causal_convolution(mixed_qkv)
        mixed_qkv = mixed_qkv.transpose(1, 2)  # B, seqlen, 2 * k_dim + v_dim

        q, k, v = torch.split(
            mixed_qkv,
            [
                self.k_dim,
                self.k_dim,
                self.v_dim,
            ],
            dim=-1,
        )

        q = q.reshape(
            B, seqlen, -1, self.k_head_dim
        )  # B, seqlen, num_k_heads, k_head_dim
        k = k.reshape(
            B, seqlen, -1, self.k_head_dim
        )  # B, seqlen, num_k_heads, k_head_dim
        v = v.reshape(
            B, seqlen, -1, self.v_head_dim
        )  # B, seqlen, num_v_heads, v_head_dim

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)

        if self.num_v_heads // self.num_k_heads > 1:
            q = q.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2
            )  # B, seqlen, num_v_heads, k_head_dim
            k = k.repeat_interleave(
                self.num_v_heads // self.num_k_heads, dim=2
            )  # B, seqlen, num_v_heads, k_head_dim

        # Decoding stage
        if self.use_cache and seqlen == 1:
            core_attn_out, last_recurrent_state = (
                self._apply_recurrent_gated_delta_rule(q, k, v, g, beta)
            )
        # Prefill stage
        else:
            core_attn_out, last_recurrent_state = self._apply_chunk_gated_delta_rule(
                q, k, v, g, beta
            )

        if self.use_cache:
            self.recurrent_state.copy_(last_recurrent_state)

        core_attn_out = core_attn_out.reshape(-1, self.v_head_dim)
        z = z.reshape(-1, self.v_head_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(B, seqlen, -1)

        output = self.out_proj(core_attn_out)
        return output
