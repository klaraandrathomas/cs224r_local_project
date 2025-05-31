"""
train_sft.py
────────────
Splits the Warm-Start dataset into train / validation
Logs train & val loss + perplexity every `--log_every` updates
Writes a CSV in ./results/ named after the hyper-params
"""

import argparse, csv, math, os, time, textwrap, torch
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, Dataset
from countdown_dataloader import (
    build_tokenizer,
    tok_sft,
    collate_sft,      
)


def make_loaders(model_name, bsz, max_len):
    tok = build_tokenizer(model_name)

    train_ds = load_dataset("Asap7772/cog_behav_all_strategies", split="train")
    val_ds   = load_dataset("Asap7772/cog_behav_all_strategies", split="test")

    for name, subset in (("train", train_ds), ("val", val_ds)):
        subset = subset.map(
            tok_sft,
            fn_kwargs={"tok": tok, "max_len": max_len},
            batched=True,
            remove_columns=subset.column_names,
            load_from_cache_file=False,
            num_proc=8,
        )
        if name == "train":
            train_tok = subset
        else:
            val_tok = subset

    loader_kw = dict(
        batch_size=bsz,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda b, pad=tok.pad_token_id: collate_sft(b, pad),
    )
    return tok, \
           torch.utils.data.DataLoader(train_tok, shuffle=True,  **loader_kw), \
           torch.utils.data.DataLoader(val_tok,   shuffle=False, **loader_kw)



def eval_loss(model, loader, device):
    model.eval()
    total, cnt = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            loss  = model(**batch).loss
            total += loss.item()
            cnt   += 1
    model.train()
    return total / cnt


def log_row(writer, step, epoch, phase, loss):
    ppl = math.exp(min(loss, 20))   # cap exp overflow
    writer.writerow({"step": step, "epoch": epoch,
                     "phase": phase, "loss": loss, "ppl": ppl})


def train_loop(model, train_loader, val_loader, tok,
               epochs, lr, device, log_every, csv_path):
    optim = AdamW(model.parameters(), lr=lr)
    sched = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(0.05 * epochs * len(train_loader)),
        num_training_steps=epochs * len(train_loader),
    )
    scaler = GradScaler()
    Path(csv_path).parent.mkdir(exist_ok=True, parents=True)
    csv_file = open(csv_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=["step","epoch","phase","loss","ppl"])
    writer.writeheader()

    global_step = 0
    for ep in range(1, epochs + 1):
        print(f"\n===== Epoch {ep}/{epochs} =====")

        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with autocast(dtype=torch.float16):
                loss = model(**batch).loss

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optim); scaler.update(); optim.zero_grad(set_to_none=True)
            sched.step()

            if global_step % log_every == 0:
                log_row(writer, global_step, ep, "train", loss.item())
                csv_file.flush()

                # validation
                val_loss = eval_loss(model, val_loader, device)
                log_row(writer, global_step, ep, "val", val_loss)
                csv_file.flush()

                print(f"[step {global_step:6d}] "
                      f"train_loss={loss.item():.4f} "
                      f"val_loss={val_loss:.4f}")

            global_step += 1
    csv_file.close()
    print(f"log saved to {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--epochs",     type=int,   default=1)
    ap.add_argument("--bsz",        type=int,   default=1)
    ap.add_argument("--lr",         type=float, default=2e-5)
    ap.add_argument("--log_every",  type=int,   default=100)
    ap.add_argument("--max_len",    type=int,   default=1024)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok, train_loader, val_loader = make_loaders(
        args.model_name, args.bsz, args.max_len
    )

    model = (AutoModelForCausalLM
             .from_pretrained(args.model_name, trust_remote_code=True)
             .to(device))
    model.gradient_checkpointing_enable()
    model.resize_token_embeddings(len(tok))


    fname = (f"{args.model_name.replace('/','_')}"
             f"_bs{args.bsz}_lr{args.lr}_len{args.max_len}.csv")
    log_path = Path("results") / fname

    train_loop(model, train_loader, val_loader, tok,
               epochs=args.epochs, lr=args.lr, device=device,
               log_every=args.log_every, csv_path=log_path)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir); tok.save_pretrained(args.out_dir)
    print(f"checkpoint saved to {args.out_dir}")



if __name__ == "__main__":
    main()
