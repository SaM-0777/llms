import torch


def compute_rope_params(
    dim: int,
    context_length: int = 4096,
    theta_base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
):
    assert dim % 2 == 0, "Embedding dimension must be even"
    inv_freqs = 1.0 / (
        theta_base ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)].float() / dim)
    )
    t = torch.arange(context_length, device=inv_freqs.device, dtype=dtype)
    freqs = torch.outer(t, inv_freqs)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    return freqs_cis.view(1, x.shape[1], 1, x.shape[-1])


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs
    q_shape, k_shape = q.shape, k.shape
    q_dtype, k_dtype = q.dtype, k.dtype

    q = q.float().reshape(*q.shape[:-1], -1, 2)
    k = k.float().reshape(*k.shape[:-1], -1, 2)

    q0 = q[..., 0]
    q1 = q[..., 1]
    k0 = k[..., 0]
    k1 = k[..., 1]
    
    cos = reshape_for_broadcast(cos, q0)
    sin = reshape_for_broadcast(sin, q0)

    q_out = torch.stack([q0 * cos - q1 * sin, q0 * sin + q1 * cos], dim=-1)
    k_out = torch.stack([k0 * cos - k1 * sin, k0 * sin + k1 * cos], dim=-1)

    q_out = q_out.flatten(-2).reshape(q_shape)
    k_out = k_out.flatten(-2).reshape(k_shape)

    return q_out.to(q_dtype), k_out.to(k_dtype)


def compute_polar_rope_params(
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


def reshape_polar_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_polar_rotary_emb(
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
