"""
Train Qwen-2.5-0.5B on a *processed* SFT dataset folder produced by
data/prep_ultra_sft.py   (or any folder with train/ val/ and config.json).

Usage
-----
    python scripts/run_sft.py \
            --data data/processed/ultra_sft \
            --epochs 1 \
            --bsz 4 \
            --lr 2e-5 \
            --grad-acc 8 \
            --log-every 100
"""
from pathlib import Path
import json, math, argparse, torch
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader

# ───────────────── collate util ─────────────────
def collate_fn(batch, pad_id, label_pad_id=-100):
    """Right-pad `input_ids`, `labels`, `attention_mask`."""
    import torch
    keys = batch[0].keys()           # input_ids / labels / attention_mask
    cols = {k: [torch.tensor(b[k]) for b in batch] for k in keys}
    pad = lambda seqs, pad_val: torch.nn.utils.rnn.pad_sequence(
        seqs, batch_first=True, padding_value=pad_val
    )
    out = {
        "input_ids":      pad(cols["input_ids"],      pad_id),
        "attention_mask": pad(cols["attention_mask"], 0),
        "labels":         pad(cols["labels"],         label_pad_id),
    }
    return out

# ───────────────── training loop ─────────────────
def train(data_dir: Path,
          epochs: int,
          bsz: int,
          lr: float,
          grad_acc: int,
          log_every: int):

    cfg   = json.load(open(data_dir/"config.json"))
    tok   = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
+             cfg["model"], torch_dtype=torch.float16   # force fp16
            ).cuda()

    ds_tr = load_from_disk(data_dir/"train")
    ds_va = load_from_disk(data_dir/"val")

    dl_tr = DataLoader(
        ds_tr, shuffle=True, batch_size=bsz, num_workers=4,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id)
    )
    dl_va = DataLoader(
        ds_va, batch_size=bsz, num_workers=2,
        collate_fn=lambda b: collate_fn(b, tok.pad_token_id)
    )

    opt   = AdamW(model.parameters(), lr=lr)
    steps = epochs * math.ceil(len(ds_tr) / bsz)
    sched = get_linear_schedule_with_warmup(opt, 0.03 * steps, steps)
    scaler = torch.amp.GradScaler('cuda')

    global_step = 0
    for ep in range(epochs):
        model.train()
        for batch in dl_tr:
            batch = {k:v.cuda() for k,v in batch.items()}
            with torch.amp.autocast('cuda', dtype=torch.float16):
                loss = model(**batch).loss / grad_acc
            scaler.scale(loss).backward()

            if (global_step + 1) % grad_acc == 0:
                scaler.step(opt); scaler.update()
                opt.zero_grad();  sched.step()

            if global_step % log_every == 0:
                print(f"ep{ep} step{global_step} train-loss {loss.item()*grad_acc:.3f}")
            global_step += 1

        # ── validation perplexity ─────────────────────
        model.eval(); tot, n = 0.0, 0
        with torch.no_grad():
            for batch in dl_va:
                batch = {k:v.cuda() for k,v in batch.items()}
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    tot += batch["labels"].ne(-100).sum().item() * \
                           model(**batch).loss.item()
                    n   += batch["labels"].ne(-100).sum().item()
        ppl = math.exp(tot / n)
        print(f"*** epoch {ep}  validation PPL {ppl:.2f} ***")

        model.save_pretrained(f"checkpoints/ultra_sft_ep{ep}")

    return model, tok

# ───────────────── sample decode ─────────────────
def demo_decode(model, tok, prompt, max_new=256):
    device = next(model.parameters()).device
    ids = tok(prompt, return_tensors="pt").to(device)
    out = model.generate(**ids, max_new_tokens=max_new,
                         temperature=0.7, top_p=0.95)
    print(tok.decode(out[0], skip_special_tokens=True))

# ───────────────── CLI ─────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--bsz", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--grad-acc", type=int, default=8)
    ap.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    mdl, tok = train(data_dir=args.data,
                     epochs=args.epochs,
                     bsz=args.bsz,
                     lr=args.lr,
                     grad_acc=args.grad_acc,
                     log_every=args.log_every)

    # quick sanity generation
    print("\n=== SAMPLE ===")
    demo_decode(
        mdl, tok,
        "You are an expert barista. Give me three quick tips for latte art.",
        max_new=120
    )
