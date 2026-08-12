#!/usr/bin/env python3
"""Mint an H3CD precomputed-conditioning file for h3.c's H3_CONDITIONING_FILE.

Replays h3_text_encoder.c exactly: raw BPE tokenization (no chat template, no
special tokens), embedding lookup, the released 50 Qwen3-VL language layers
(RMSNorm eps 1e-6, per-head q/k RMSNorm, rotate-half RoPE theta 5e6, causal
GQA 64Q/8KV, SwiGLU), NO final norm, output cast to BF16. Weights stream
layer-by-layer from the Comfy-Org single-file BF16 text encoder, compute in
float32 on CUDA (stronger than the M4's BF16 Metal path).

Usage:
  mint_h3_conditioning.py --prompt "..." --out cond.h3cd \
      --te qwen3vl_32b_minimax_h3_bf16.safetensors --tokenizer tokenizer.json
"""

import argparse
import struct

import torch
from safetensors import safe_open
from tokenizers import Tokenizer

HIDDEN = 5120
LAYERS = 50
Q_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 128
ROPE_THETA = 5_000_000.0
RMS_EPS = 1e-6
PAD_TOKEN_ID = 151643
MAGIC = 0x44434833  # "H3CD"


def rms_norm(x, weight, eps=RMS_EPS):
    variance = x.pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps) * weight


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--te", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    ids = tokenizer.encode(args.prompt, add_special_tokens=False).ids
    if not ids:
        ids = [PAD_TOKEN_ID]
    print(f"tokens ({len(ids)}): {ids}")

    device = torch.device(args.device)
    tokens = torch.tensor(ids, dtype=torch.long, device=device)
    seq = len(ids)

    positions = torch.arange(seq, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (ROPE_THETA ** (
        torch.arange(0, HEAD_DIM, 2, dtype=torch.float32, device=device)
        / HEAD_DIM))
    angles = positions[:, None] * inv_freq[None, :]          # [seq, 64]
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)     # [seq, 128]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)

    causal = torch.full((seq, seq), float("-inf"), device=device)
    causal = torch.triu(causal, diagonal=1)

    with safe_open(args.te, framework="pt", device=str(device)) as f:
        def load(name):
            return f.get_tensor(name).to(torch.float32)

        hidden = load("model.embed_tokens.weight")[tokens]
        # h3 keeps activations BF16 between ops; pin the embedding boundary.
        hidden = hidden.to(torch.bfloat16).to(torch.float32)

        for layer in range(LAYERS):
            p = f"model.layers.{layer}."
            normed = rms_norm(hidden, load(p + "input_layernorm.weight"))
            q = normed @ load(p + "self_attn.q_proj.weight").T
            k = normed @ load(p + "self_attn.k_proj.weight").T
            v = normed @ load(p + "self_attn.v_proj.weight").T
            q = q.view(seq, Q_HEADS, HEAD_DIM)
            k = k.view(seq, KV_HEADS, HEAD_DIM)
            v = v.view(seq, KV_HEADS, HEAD_DIM)
            q = rms_norm(q, load(p + "self_attn.q_norm.weight"))
            k = rms_norm(k, load(p + "self_attn.k_norm.weight"))
            q = q * cos[:, None, :] + rotate_half(q) * sin[:, None, :]
            k = k * cos[:, None, :] + rotate_half(k) * sin[:, None, :]
            q = q.transpose(0, 1)                              # [H, seq, D]
            k = k.repeat_interleave(Q_HEADS // KV_HEADS, dim=1).transpose(0, 1)
            v = v.repeat_interleave(Q_HEADS // KV_HEADS, dim=1).transpose(0, 1)
            scores = q @ k.transpose(-2, -1) / (HEAD_DIM ** 0.5) + causal
            attention = torch.softmax(scores, dim=-1) @ v      # [H, seq, D]
            attention = attention.transpose(0, 1).reshape(seq, Q_HEADS * HEAD_DIM)
            hidden = hidden + attention @ load(p + "self_attn.o_proj.weight").T
            normed = rms_norm(hidden, load(p + "post_attention_layernorm.weight"))
            gate = normed @ load(p + "mlp.gate_proj.weight").T
            up = normed @ load(p + "mlp.up_proj.weight").T
            fused = torch.nn.functional.silu(gate) * up
            hidden = hidden + fused @ load(p + "mlp.down_proj.weight").T
            print(f"layer {layer + 1}/{LAYERS} |h|={hidden.norm().item():.3f}",
                  flush=True)

    values = hidden.to(torch.bfloat16).view(torch.uint16).cpu().numpy()
    with open(args.out, "wb") as out:
        out.write(struct.pack("<IIQQII", MAGIC, 1, seq, HIDDEN, 0, 0))
        out.write(values.tobytes())
    print(f"wrote {args.out}: {seq} tokens x {HIDDEN}, "
          f"{values.nbytes + 32} bytes")


if __name__ == "__main__":
    main()
