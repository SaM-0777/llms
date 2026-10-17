import torch


def calculate_rope_params(
    dim: int = 256,
    context_len: int = 32768,
    theta_base: float = 1000000.0,
    device: torch.device = torch.device("cuda"),
    dtype: torch.dtype = torch.float32,
):
    assert dim % 2 == 0, f"Embedding dimension must be even"
    indices = torch.arange(0, dim, 2, dtype=dtype, device=device)
    inv_freqs = 1.0 / (theta_base ** (indices / dim))
    t = torch.arange(context_len, device=inv_freqs.device, dtype=dtype)
    freqs = torch.outer(t, inv_freqs)
    freqs = torch.cat((freqs, freqs), dim=-1)
    cos = freqs.cos().to(dtype=dtype)
    sin = freqs.sin().to(dtype=dtype)
    return cos, sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    return freqs_cis.view(1, x.shape[1], 1, x.shape[-1])


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embeddings(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = freqs
    q_dtype, k_dtype = q.dtype, k.dtype

    cos = cos[None, None, :, :].float()
    sin = sin[None, None, :, :].float()

    q_out = q.float() * cos + rotate_half(q.float()) * sin
    k_out = k.float() * cos + rotate_half(k.float()) * sin

    return q_out.to(q_dtype), k_out.to(k_dtype)
