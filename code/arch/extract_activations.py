"""
Extract hidden activations from Qwen3-1.7B for a movie script.

Usage:
    python extract_activations.py <script.txt> [options]

Options:
    --output PATH        Output JSON file (default: activations.json)
    --layers LAYERS      Comma-separated layer indices, e.g. "0,6,12" (default: all)
    --pooling METHOD     How to pool token activations: mean|last|none (default: mean)
    --chunk-size N       Characters per chunk fed to model (default: 512)
    --chunk-overlap N    Overlap between chunks in chars (default: 64)
    --device DEVICE      cpu or cuda (default: auto-detect)

Example:
    python extract_activations.py my_script.txt --layers 0,8,15 --pooling mean --output out.json
"""

import argparse
import json
import sys
import textwrap

try:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with:  pip install transformers accelerate numpy")
    sys.exit(1)


MODEL_ID = "Qwen/Qwen3-1.7B"
MAX_TOKENS_PER_CHUNK = 512  # keep well under the model's 32k context for speed


# ── helpers ──────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_chars, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_chars - overlap_chars
    return chunks


def pool_hidden(hidden: torch.Tensor, mask: torch.Tensor, method: str) -> list[float]:
    """
    hidden : (1, seq_len, hidden_size)
    mask   : (1, seq_len)  — attention mask (1 = real token, 0 = pad)
    Returns a flat Python list of floats.
    """
    h = hidden.squeeze(0)  # (seq_len, hidden_size)
    m = mask.squeeze(0).bool()  # (seq_len,)

    if method == "last":
        # last real token
        last_idx = m.nonzero(as_tuple=False)[-1].item()
        vec = h[last_idx]
    elif method == "mean":
        real = h[m]  # (n_real, hidden_size)
        vec = real.mean(dim=0)
    elif method == "none":
        # return all real tokens as a 2-D list
        return h[m].float().cpu().numpy().tolist()
    else:
        raise ValueError(f"Unknown pooling method: {method!r}")

    return vec.float().cpu().numpy().tolist()


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract Qwen3-1.7B hidden activations from a movie script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(__doc__),
    )
    p.add_argument("script", help="Path to the movie script .txt file")
    p.add_argument("--output", default="activations.json", help="Output JSON path")
    p.add_argument("--layers", default=None,
                   help="Comma-separated layer indices (default: all layers)")
    p.add_argument("--pooling", default="mean", choices=["mean", "last", "none"],
                   help="Token pooling strategy (default: mean)")
    p.add_argument("--chunk-size", type=int, default=1500,
                   help="Characters per chunk (default: 1500)")
    p.add_argument("--chunk-overlap", type=int, default=150,
                   help="Overlap between chunks in chars (default: 150)")
    p.add_argument("--device", default=None,
                   help="Device: cpu or cuda (default: auto)")
    return p.parse_args()


def main():
    args = parse_args()

    # ── device ──
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[info] Using device: {device}")

    # ── load script ──
    print(f"[info] Reading script from: {args.script}")
    with open(args.script, "r", encoding="utf-8", errors="replace") as fh:
        script_text = fh.read()
    print(f"[info] Script length: {len(script_text):,} characters")

    # ── load model ──
    print(f"[info] Loading tokenizer and model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        output_hidden_states=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()

    n_layers = model.config.num_hidden_layers  # e.g. 28 for 1.7B
    print(f"[info] Model has {n_layers} hidden layers + embedding layer = {n_layers + 1} total")

    # ── resolve which layers to extract ──
    if args.layers is not None:
        requested = [int(x.strip()) for x in args.layers.split(",")]
        # layer 0 = embedding output, layers 1..n = transformer block outputs
        valid = set(range(n_layers + 1))
        bad = [l for l in requested if l not in valid]
        if bad:
            print(f"[warn] Ignoring out-of-range layer indices: {bad}")
        layer_indices = sorted(set(requested) & valid)
    else:
        layer_indices = list(range(n_layers + 1))
    print(f"[info] Extracting {len(layer_indices)} layer(s): {layer_indices}")

    # ── chunk script ──
    chunks = chunk_text(script_text, args.chunk_size, args.chunk_overlap)
    print(f"[info] Split script into {len(chunks)} chunks "
          f"(chunk_size={args.chunk_size}, overlap={args.chunk_overlap})")

    # ── extract activations ──
    results = []
    for i, chunk in enumerate(chunks):
        print(f"[info] Processing chunk {i + 1}/{len(chunks)} ...", end="\r", flush=True)

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKENS_PER_CHUNK,
            padding=False,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (n_layers+1) tensors, each (1, seq, hidden)
        hidden_states = outputs.hidden_states  # tuple length = n_layers + 1

        chunk_record: dict = {
            "chunk_index": i,
            "text_preview": chunk[:120].replace("\n", " "),
            "n_tokens": inputs["input_ids"].shape[1],
            "layers": {},
        }

        for layer_idx in layer_indices:
            h = hidden_states[layer_idx]   # (1, seq_len, hidden_size)
            pooled = pool_hidden(h, inputs["attention_mask"], args.pooling)
            chunk_record["layers"][str(layer_idx)] = pooled

        results.append(chunk_record)

    print()  # newline after \r progress

    # ── build output document ──
    output_doc = {
        "model": MODEL_ID,
        "script_file": args.script,
        "script_length_chars": len(script_text),
        "n_chunks": len(chunks),
        "chunk_size_chars": args.chunk_size,
        "chunk_overlap_chars": args.chunk_overlap,
        "pooling": args.pooling,
        "layer_indices": layer_indices,
        "hidden_size": model.config.hidden_size,
        "activations": results,
    }

    # ── write JSON ──
    print(f"[info] Writing activations to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(output_doc, fh)

    size_mb = __import__("os").path.getsize(args.output) / 1e6
    print(f"[done] Saved {len(results)} chunk records to {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
