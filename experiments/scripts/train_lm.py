"""Train a small Qwen2-architecture language model on public-domain books (CPU).

The real target is Qwen2.5-0.5B, but Hugging Face is unreachable from this machine, so
the experiments run on a model trained here with the same architecture family
(RMSNorm, rotary attention with grouped KV heads, SwiGLU FFN).

usage: train_lm.py DATA_DIR OUT_DIR MINUTES [--bench]
"""
import json
import math
import os
import sys
import time

import torch
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

data_dir, out_dir, minutes = sys.argv[1], sys.argv[2], float(sys.argv[3])
bench = "--bench" in sys.argv
torch.set_num_threads(2)
torch.manual_seed(0)
os.makedirs(out_dir, exist_ok=True)

tok = PreTrainedTokenizerFast.from_pretrained(os.path.join(data_dir, "tokenizer"))
ids = torch.load(os.path.join(data_dir, "train_ids.pt")).long()
alice = torch.load(os.path.join(data_dir, "carroll-alice_ids.pt")).long()

cfg = Qwen2Config(vocab_size=len(tok), hidden_size=256, intermediate_size=768, num_hidden_layers=12,
                  num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=512,
                  tie_word_embeddings=True, rope_theta=10000.0)
model = Qwen2ForCausalLM(cfg)
n_params = sum(p.numel() for p in model.parameters())
decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
opt = torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                        lr=2e-3, betas=(0.9, 0.95), eps=1e-8)
T, B, peak, warm = 256, 16, 2e-3, 100
budget = minutes * 60
eval_x = alice[: 16 * T].view(16, T)


@torch.no_grad()
def held_out_loss():
    model.eval()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(input_ids=eval_x, labels=eval_x).loss.item()
    model.train()
    return loss


log = open(os.path.join(out_dir, "train_log.txt"), "a")
start, step, last_save = time.time(), 0, time.time()
model.train()
while True:
    elapsed = time.time() - start
    if elapsed > budget or (bench and step == 20):
        break
    frac = elapsed / budget
    lr = peak * min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    for g in opt.param_groups:
        g["lr"] = lr
    ix = torch.randint(0, ids.numel() - T - 1, (B,))
    x = torch.stack([ids[i:i + T] for i in ix])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        loss = model(input_ids=x, labels=x).loss
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    step += 1
    if step % 50 == 0:
        rate = step * B * T / (time.time() - start)
        print(f"step {step} loss {loss.item():.3f} lr {lr:.2e} {rate:.0f} tok/s {elapsed / 60:.1f} min", file=log, flush=True)
    if time.time() - last_save > 300:
        print(f"step {step} held-out Alice loss {held_out_loss():.3f}", file=log, flush=True)
        model.save_pretrained(out_dir)
        last_save = time.time()

rate = step * B * T / (time.time() - start)
if bench:
    print(f"params {n_params / 1e6:.2f}M | {rate:.0f} tokens/s with bf16 autocast on 2 threads")
else:
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    final = held_out_loss()
    json.dump({"params": n_params, "steps": step, "tokens_seen": step * B * T, "minutes": (time.time() - start) / 60,
               "held_out_alice_loss": final, "uniform_loss": math.log(len(tok)), "config": cfg.to_dict()},
              open(os.path.join(out_dir, "train_summary.json"), "w"), indent=1, default=str)
    print(f"done: {step} steps, held-out Alice loss {final:.3f} (uniform {math.log(len(tok)):.2f})", file=log, flush=True)
