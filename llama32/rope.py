import torch


def compute_rope_params(
    dim: int,
    context_length: int = 4096,
    theta_base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
):
    assert dim % 2 == 0, "Embedding dimension must be even"
    freqs = 1.0 / (theta_base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(context_length, device=freqs.device, dtype=dtype)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freq_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freq_cis = reshape_for_broadcast(freq_cis, q_)
    q_out = torch.view_as_real(q_ * freq_cis).flatten(3)
    k_out = torch.view_as_real(k_ * freq_cis).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)
