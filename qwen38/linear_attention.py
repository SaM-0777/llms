import math
import torch
import torch.nn as nn

dtype = torch.bfloat16
device = torch.device("mps")

seed = 42
torch.manual_seed(seed)

B = 1

vocab_size = 62656
seqlen = 16
embedding_dim = 1024

# Batch of sequence of tokens
x = torch.randint(
    0,
    vocab_size,
    (
        B,
        seqlen,
    ),
    device=device,
)

embeddings = nn.Embedding(vocab_size, embedding_dim, device=device)
embedding_tokens = embeddings(x)  # B, seqlen, embedding_dim

num_head = 8
dk = embedding_dim // num_head

Wq = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device, dtype=dtype)
Wk = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device, dtype=dtype)
Wv = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device, dtype=dtype)
Wo = nn.Linear(embedding_dim, embedding_dim, bias=False, device=device, dtype=dtype)

Q = Wq(embedding_tokens)  # B, seqlen, embedding_dim
K = Wk(embedding_tokens)  # B, seqlen, embedding_dim
V = Wv(embedding_tokens)  # B, seqlen, embedding_dim

Q = Q.view(B, seqlen, num_head, dk)  # B, seqlen, num_heads, dk
K = K.view(B, seqlen, num_head, dk)  # B, seqlen, num_heads, dk
V = V.view(B, seqlen, num_head, dk)  # B, seqlen, num_heads, dk

Q = Q.transpose(1, 2)  # B, num_heads, seqlen, dk
K = K.transpose(1, 2)  # B, num_heads, seqlen, dk
V = V.transpose(1, 2)  # B, num_heads, seqlen, dk

QKT = Q @ K.transpose(-2, -1)  # B, num_heads, seqlen, seqlen

mask = None
if seqlen > 1:
    mask = torch.full((seqlen, seqlen), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # upper triangular matrix
    QKT = QKT + mask

scores = torch.softmax(QKT / math.sqrt(dk), dim=-1)  # B, num_heads, seqlen, seqlen

out = scores @ V  # B, num_heads, seqlen, dk
out = out.transpose(1, 2).contiguous()  # B, seqlen, num_heads, dk
out = out.view(B, seqlen, -1)  # B, seqlen, num_heads * dk
out = Wo(out)  # B, seqlen, embedding_dim

attention = embedding_tokens + out


# Linear Attention
KV = K.transpose(-2, -1).unsqueeze(-1) @ V.unsqueeze(-2)  # B, num_heads, dk, dk
state = torch.cumsum(KV, dim=2)  # B, num_heads, dk, dk
linear_out = (Q.unsqueeze(-2) @ state).squeeze(-2)  # B, num_heads, seqlen, dk
linear_out = (
    linear_out.transpose(1, 2).contiguous().view(B, seqlen, -1)
)  # B, seqlen, num_heads * dk
