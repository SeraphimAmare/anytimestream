"""Shared helpers for the anytime-LLM starter kit.

Works with SwiGLU decoder models in Hugging Face transformers: Llama 2/3,
Mistral, Qwen2/2.5 (anything whose layers look like model.model.layers[i]
with .mlp.{gate_proj, up_proj, down_proj, act_fn} and .post_attention_layernorm).
Use --model tiny for a seconds-long smoke test with a small random model.
"""
from contextlib import contextmanager

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------- loading

def pick_device(pref=None):
    if pref:
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_tiny_model(device, seed=0):
    """A 4-layer random Llama (no download) for checking that everything runs."""
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=256,
                      num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=1024, initializer_range=0.1)
    return LlamaForCausalLM(cfg).to(device).eval()


def load_model(name, device, dtype=None):
    """Return (model, tokenizer). name='tiny' gives a random smoke-test model and no tokenizer.
    Default precision: bfloat16 on GPU, float32 on CPU (CPU bf16 is slow on most machines and noisier)."""
    if name == "tiny":
        return build_tiny_model(device), None
    if dtype is None:
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    if dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        dtype = torch.float16  # older GPUs (e.g. T4) lack fast bf16
    import transformers
    from packaging import version
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # transformers >= 4.56 renamed torch_dtype -> dtype; passing the wrong one can silently load fp32.
    kw = "dtype" if version.parse(transformers.__version__) >= version.parse("4.56") else "torch_dtype"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, **{kw: dtype})
    return model.to(device).eval(), tok


# ----------------------------------------------------------------------------- model anatomy

def decoder_layers(model):
    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise ValueError("Expected model.model.layers (Llama/Mistral/Qwen2-style decoder).")
    return layers


def check_mlp(mlp):
    for attr in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        if not hasattr(mlp, attr):
            raise ValueError(f"MLP has no '{attr}'. This kit expects SwiGLU MLPs (Llama, Mistral, Qwen2/2.5).")
    return mlp


@torch.no_grad()
def ffn_hidden(mlp, x):
    """h = act(gate(x)) * up(x): one value per FFN neuron, shape [..., d_ff].
    The FFN output is down_proj(h) = sum_i h_i * W_down[:, i], so dropping neuron i
    drops exactly one term of that sum."""
    check_mlp(mlp)
    return mlp.act_fn(mlp.gate_proj(x)) * mlp.up_proj(x)


@torch.no_grad()
def down_col_norms(mlp):
    """||W_down[:, i]|| for every neuron i (float32, on the weight's device)."""
    return mlp.down_proj.weight.float().norm(dim=0)


def neuron_scores(h, col_norms):
    """Contribution size of each neuron: |h_i| * ||W_down[:, i]|| (float32)."""
    return h.abs().float() * col_norms


def model_dims(cfg):
    heads = cfg.num_attention_heads
    kv = getattr(cfg, "num_key_value_heads", None) or heads
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // heads
    return dict(d_model=cfg.hidden_size, d_ff=cfg.intermediate_size, n_layers=cfg.num_hidden_layers,
                heads=heads, kv_heads=kv, head_dim=head_dim, vocab=cfg.vocab_size)


@contextmanager
def swap_mlps(model, make):
    """Temporarily replace each layer's MLP with make(layer_idx, original_mlp)."""
    layers = decoder_layers(model)
    originals = [layer.mlp for layer in layers]
    try:
        for i, layer in enumerate(layers):
            layer.mlp = make(i, originals[i])
        yield originals
    finally:
        for layer, mlp in zip(layers, originals):
            layer.mlp = mlp


# ----------------------------------------------------------------------------- data

def eval_batches(tokenizer, n_seqs, seq_len, text_file=None, vocab_size=None, seed=0):
    """List of [1, seq_len] token tensors. Default text: WikiText-2 test split."""
    if tokenizer is None:  # tiny smoke-test model: random tokens are fine
        g = torch.Generator().manual_seed(seed)
        return [torch.randint(0, vocab_size, (1, seq_len), generator=g) for _ in range(n_seqs)]
    if text_file:
        with open(text_file, encoding="utf-8", errors="ignore") as f:
            text = f.read()
    else:
        from datasets import load_dataset
        try:
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        except Exception:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    n_avail = ids.numel() // seq_len
    if n_avail < n_seqs:
        raise ValueError(f"Text only has {n_avail} chunks of {seq_len} tokens; lower --n-seqs or --seq-len.")
    starts = torch.linspace(0, (n_avail - 1) * seq_len, n_seqs).long().tolist()  # spread across the text
    return [ids[s:s + seq_len].unsqueeze(0) for s in starts]


# ----------------------------------------------------------------------------- FFN statistics

@torch.no_grad()
def collect_ffn_stats(model, batches, device, active_frac=0.2, keep_fracs=()):
    """Dense pass with hooks. Returns per-layer:
      freq[l, i]      fraction of tokens where neuron i is in that token's top-`active_frac` by score
      rel_err[l, j]   mean ||y - y_keep|| / ||y|| when only the top keep_fracs[j] neurons are kept
    """
    layers = decoder_layers(model)
    L = len(layers)
    d_ff = layers[0].mlp.down_proj.weight.shape[1]
    col = [down_col_norms(layer.mlp) for layer in layers]
    freq = torch.zeros(L, d_ff, dtype=torch.float64)
    err = torch.zeros(L, len(keep_fracs), dtype=torch.float64)
    n_tok = torch.zeros(L, dtype=torch.float64)
    k_act = max(1, round(active_frac * d_ff))

    def make_hook(l):
        def hook(mod, inputs, output):
            x = inputs[0]
            h = ffn_hidden(mod, x).reshape(-1, d_ff)
            s = neuron_scores(h, col[l])
            top = s.topk(k_act, dim=-1).indices
            freq[l] += torch.bincount(top.flatten(), minlength=d_ff).double().cpu()
            if keep_fracs:
                y = output.reshape(-1, output.shape[-1]).float()
                yn = y.norm(dim=-1).clamp_min(1e-12)
                order = s.argsort(dim=-1, descending=True)
                for j, f in enumerate(keep_fracs):
                    k = max(1, round(f * d_ff))
                    mask = torch.zeros_like(h).scatter_(-1, order[:, :k], 1)
                    y_k = mod.down_proj(h * mask).float()
                    err[l, j] += ((y - y_k).norm(dim=-1) / yn).sum().double().cpu()
            n_tok[l] += h.shape[0]
        return hook

    handles = [layer.mlp.register_forward_hook(make_hook(l)) for l, layer in enumerate(layers)]
    try:
        for b in batches:
            model(input_ids=b.to(device), use_cache=False)
    finally:
        for hd in handles:
            hd.remove()
    freq /= n_tok[:, None]
    err /= n_tok[:, None]
    return freq.float(), err.float()


# ----------------------------------------------------------------------------- quality metrics

@torch.no_grad()
def compare_logits(ref_logits, logits, input_ids=None):
    """Per-position KL(ref || test), top-1 agreement, and next-token NLL under the test model."""
    ref_lp = F.log_softmax(ref_logits.float(), dim=-1)
    lp = F.log_softmax(logits.float(), dim=-1)
    out = {"kl": (ref_lp.exp() * (ref_lp - lp)).sum(-1),
           "agree": (ref_logits.argmax(-1) == logits.argmax(-1)).float()}
    if input_ids is not None:
        out["nll"] = -lp[:, :-1].gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return out
